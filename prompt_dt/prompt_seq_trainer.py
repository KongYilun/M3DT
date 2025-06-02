# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time
from wandb import env
from .prompt_utils import flatten_prompt, parameters_to_gradvector, parameters_to_gradvector_expert, parameters_to_gradvector_router
import copy


class PromptSequenceTrainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn,
                 scheduler=None, eval_fns=None, get_prompt=None, get_prompt_batch=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.get_prompt = get_prompt
        self.prompt = self.get_prompt() # sample prompt data when initialization
        self.get_prompt_batch = get_prompt_batch

        self.start_time = time.time()


    def pure_train_iteration_mix(self, num_steps, no_prompt=False, env_name=None, action_mask_dim=8):

        train_losses = []
        gradient_norms=[]
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss,gradient_norm = self.train_step_mix(no_prompt, env_name, action_mask_dim)
            train_losses.append(train_loss)
            gradient_norms.append(gradient_norm)
            if self.scheduler is not None:
                self.scheduler.step()
        
        
        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs[f'training/{env_name}_gradient_norm'] = np.mean(gradient_norms)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs


    def train_step_mix(self, no_prompt=False, index=None, action_mask_dim=None):
        prompt, batch = self.get_prompt_batch(index=index)
        states, actions, rewards, dones, rtg, timesteps, attention_mask, env_name = batch
        action_target = torch.clone(actions)
        if no_prompt:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=None
            )
        else:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=prompt
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        action_preds=action_preds[:,:action_mask_dim]
        action_target=action_target[:,:action_mask_dim]
        
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        # gradient = parameters_to_gradvector(self.model)

        self.optimizer.step()

        gradients_norm=[]
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    gradients_norm.append(param.grad.norm().cpu())

        gradient_norm=np.mean(gradients_norm)

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(), gradient_norm

    def eval_iteration_multienv(self, get_prompt, prompt_trajectories_list, eval_episodes, env_name_list, info, prompt_info, 
                                variant, env_list, iter_num=0, print_logs=False, no_prompt=False, group='test'):
        print('=' * 80)
        print('evaluate at tasks: ', env_name_list)
        logs = dict()
        print('start evaluating...')
        self.model.eval()

        eval_start = time.time()
        for env_id, env_name in enumerate(env_name_list):
            # need to sample eval_fn and prompt together 
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            # print(prompt_info[env_name])
            self.get_prompt = get_prompt(prompt_trajectories_list[env_id], prompt_info[env_name], variant)
            if not no_prompt:
                self.prompt = flatten_prompt(self.get_prompt(index=-1), batch_size=1)
                # prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = self.prompt
                # print('======get trainer.prompt', prompt_states.shape)
            else:
                self.prompt = None
            for eval_fn in self.eval_fns:
                # print('env_name : ', env_list[env_id])
                outputs = eval_fn(self.model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        total_return_mean = {}
        total_normalized_score_mean = {}
        for k, v in logs.items():
            # self.logger.record_tabular(k, float(v))
            if 'return_mean' in k:
                env = k.split('/')[1].split('_')[0]
                if 'target' not in k.split('/')[1].split('_')[1]:
                    env = env + k.split('/')[1].split('_')[1]
                if env not in total_return_mean.keys():
                    total_return_mean[env] = float(v)
                elif total_return_mean[env] < float(v):
                    total_return_mean[env] = float(v)
            if 'normalized_score_mean' in k:
                env = k.split('/')[1].split('_')[0]
                if 'target' not in k.split('/')[1].split('_')[1]:
                    env = env + k.split('/')[1].split('_')[1]
                if env not in total_normalized_score_mean.keys():
                    total_normalized_score_mean[env] = float(v)
                elif total_normalized_score_mean[env] < float(v):
                    total_normalized_score_mean[env] = float(v)
        total_mean = []
        total_normalized_score = []
        for k, v in total_return_mean.items():
            logs[f'{group}-{k}-Success']=float(total_normalized_score_mean[k])
            total_mean.append(v)
            total_normalized_score.append(total_normalized_score_mean[k])

        logs[f'{group}-Total-Return-Mean'] = np.mean(total_mean)
        logs[f'{group}-Total-Normalized-Score-Mean'] = np.mean(total_normalized_score)

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def eval_iteration_multienv_for_test(self, model, get_prompt, prompt_trajectories_list, eval_episodes, env_name_list, info, prompt_info, 
                                variant, env_list, iter_num=0, print_logs=False, no_prompt=False, group='test'):
        print('=' * 80)
        print('evaluate at tasks: ', env_name_list)
        logs = dict()
        print('start evaluating...')
        model.eval()

        eval_start = time.time()
        for env_id, env_name in enumerate(env_name_list):
            # need to sample eval_fn and prompt together 
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            # print(prompt_info[env_name])
            self.get_prompt = get_prompt(prompt_trajectories_list[env_id], prompt_info[env_name], variant)
            if not no_prompt:
                self.prompt = flatten_prompt(self.get_prompt(index=-1), batch_size=1)
                # prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = self.prompt
                # print('======get trainer.prompt', prompt_states.shape)
            else:
                self.prompt = None
            for eval_fn in self.eval_fns:
                # print('env_name : ', env_list[env_id])
                outputs = eval_fn(model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        total_return_mean = {}
        total_normalized_score_mean = {}
        for k, v in logs.items():
            # self.logger.record_tabular(k, float(v))
            if 'return_mean' in k:
                env = k.split('/')[1].split('_')[0]
                if 'target' not in k.split('/')[1].split('_')[1]:
                    env = env + k.split('/')[1].split('_')[1]
                if env not in total_return_mean.keys():
                    total_return_mean[env] = float(v)
                elif total_return_mean[env] < float(v):
                    total_return_mean[env] = float(v)
            if 'normalized_score_mean' in k:
                env = k.split('/')[1].split('_')[0]
                if 'target' not in k.split('/')[1].split('_')[1]:
                    env = env + k.split('/')[1].split('_')[1]
                if env not in total_normalized_score_mean.keys():
                    total_normalized_score_mean[env] = float(v)
                elif total_normalized_score_mean[env] < float(v):
                    total_normalized_score_mean[env] = float(v)
        total_mean = []
        total_normalized_score = []
        for k, v in total_return_mean.items():
            logs[f'{group}-{k}-Success']=float(total_normalized_score_mean[k])
            total_mean.append(v)
            total_normalized_score.append(total_normalized_score_mean[k])

        logs[f'{group}-Total-Return-Mean'] = np.mean(total_mean)
        logs[f'{group}-Total-Normalized-Score-Mean'] = np.mean(total_normalized_score)

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

 
    def save_model(self, env_name, postfix, folder, seed):
        model_name = '/prompt_model_' + env_name + postfix + f'_seed{seed}'
        torch.save(self.model.state_dict(),folder+model_name)  # model save
        print('model saved to ', folder+model_name)

    
    def update_gradient(self, no_prompt=False, env_name=None, action_mask_dim=None):

        prompt, batch = self.get_prompt_batch(index=env_name)
        states, actions, rewards, dones, rtg, timesteps, attention_mask_, env_name_ = batch
        action_target = torch.clone(actions)

        original_param = copy.deepcopy(self.model.state_dict())
        # self.model.load_state_dict(original_param)
        # print(attention_mask_.shape)
        
        self.model.train()

        if no_prompt:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask_, prompt=None
            )
        else:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask_, prompt=prompt
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask_.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask_.reshape(-1) > 0]

        action_preds=action_preds[:,:action_mask_dim]
        action_target=action_target[:,:action_mask_dim]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)

        gradient,shapes = parameters_to_gradvector(self.model)

        self.model.load_state_dict(original_param)

        return gradient,shapes
    
    def update_gradient_expert(self, no_prompt=False, env_name=None, action_mask_dim=None,seed=None):

        prompt, batch = self.get_prompt_batch(index=env_name)
        states, actions, rewards, dones, rtg, timesteps, attention_mask_, env_name_ = batch
        action_target = torch.clone(actions)

        original_param = copy.deepcopy(self.model.state_dict())
        # self.model.load_state_dict(original_param)
        # print(attention_mask_.shape)
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.model.train()

            if no_prompt:
                state_preds, action_preds, reward_preds = self.model.forward(
                    states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask_, prompt=None
                )
            else:
                state_preds, action_preds, reward_preds = self.model.forward(
                    states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask_, prompt=prompt
                )

            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask_.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask_.reshape(-1) > 0]

            action_preds=action_preds[:,:action_mask_dim]
            action_target=action_target[:,:action_mask_dim]

            loss = self.loss_fn(
                None, action_preds, None,
                None, action_target, None,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)

            expert_gradient = parameters_to_gradvector_expert(self.model)

            self.model.load_state_dict(original_param)

        return expert_gradient
    
    def update_gradient_router(self, no_prompt=False, env_name=None, action_mask_dim=None,seed=None):

        prompt, batch = self.get_prompt_batch(index=env_name)
        states, actions, rewards, dones, rtg, timesteps, attention_mask_, env_name_ = batch
        action_target = torch.clone(actions)

        original_param = copy.deepcopy(self.model.state_dict())
        # self.model.load_state_dict(original_param)
        # print(attention_mask_.shape)
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.model.train()

            if no_prompt:
                state_preds, action_preds, reward_preds = self.model.forward(
                    states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask_, prompt=None
                )
            else:
                state_preds, action_preds, reward_preds = self.model.forward(
                    states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask_, prompt=prompt
                )

            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask_.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask_.reshape(-1) > 0]

            action_preds=action_preds[:,:action_mask_dim]
            action_target=action_target[:,:action_mask_dim]

            loss = self.loss_fn(
                None, action_preds, None,
                None, action_target, None,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)

            router_gradient = parameters_to_gradvector_router(self.model)

            self.model.load_state_dict(original_param)

        return router_gradient
