# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import random
import torch
import time
from wandb import env
from .prompt_utils import flatten_prompt, parameters_to_gradvector, run_kmeans_multiple_times
import copy
from sklearn.cluster import KMeans

class PromptGroupMergeTrainer:

    def __init__(self, model, num_env, batch_size, get_batch, loss_fn,
                 eval_fns=None, get_prompt=None, get_prompt_batch=None, variant=None):
        
        self.merged_model = model
        self.merged_optimizer = torch.optim.AdamW(
            self.merged_model.parameters(),
            lr=variant['learning_rate'],
            weight_decay=variant['weight_decay'],
        )
        self.merged_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.merged_optimizer,
            lambda steps: min((steps + 1) / variant['warmup_steps'], 1)
        )

        self.num_env=num_env
        self.grouped_model = []
        self.grouped_envs = []
        self.group_labels = None
        self.group_num=None
        self.grouped_optimizer=[]
        self.grouped_scheduler=[]

        self.lr = variant['learning_rate']
        self.weight_decay = variant['weight_decay']
        self.warmup_steps = variant['warmup_steps']
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.get_prompt = get_prompt
        self.prompt = self.get_prompt() # sample prompt data when initialization
        self.get_prompt_batch = get_prompt_batch
        self.start_time = time.time()

    def group(self, iter, env_gradients, cosine_sim, k, random_group=False):
        self.group_num=k ####other mothod for nums
        # kmeans = run_kmeans_multiple_times(n_clusters=self.group_num)
        if random_group==False:
            env_gradients=torch.tensor(env_gradients).cuda()
            _ , group_labels=run_kmeans_multiple_times(env_gradients, self.group_num)
            self.group_labels = group_labels          
        else:
            group_labels = [0] * 55 + [1] * 55 + [2] * 50
            random.shuffle(group_labels)
            self.group_labels = group_labels
        self.grouped_envs=[[] for _ in range(self.group_num)]
        for i in range(len(group_labels)):
            self.grouped_envs[group_labels[i]].append(i)
        # original_param = copy.deepcopy(self.merged_model.state_dict())
        # self.grouped_model=[self.model]*self.group_num
        self.grouped_model.clear()
        for _ in range(self.group_num):
            self.grouped_model.append(copy.deepcopy(self.merged_model))

        self.grouped_optimizer.clear()
        self.grouped_scheduler.clear()
        for i in range(self.group_num):
            self.grouped_optimizer.append(torch.optim.AdamW(
                self.grouped_model[i].parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            ))
            self.grouped_scheduler.append(torch.optim.lr_scheduler.LambdaLR(
                self.grouped_optimizer[i],
                lambda steps: min((steps + 1) / self.warmup_steps, 1)
            ))
    
    def merge(self,iter,weight_merge=True):
        no_sum=[]
        if weight_merge==False:
            sum_weight=[1/self.group_num for _ in range(self.group_num)]
        else:
            sum_weight=[len(self.grouped_envs[i])/self.num_env for i in range(self.group_num)]
        theta={}
        for key in self.merged_model.state_dict().keys():
            if not any(ns in key for ns in no_sum):
                theta[key]=0
                for i in range(self.group_num):
                    theta[key] += sum_weight[i] * self.grouped_model[i].state_dict()[key]
            else:
                theta[key]=self.merged_model.state_dict()[key]  #########直接用分组训练前的参数
        
        self.merged_model.load_state_dict(theta)
        self.merged_optimizer = torch.optim.AdamW(
            self.merged_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.merged_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.merged_optimizer,
            lambda steps: min((steps + 1) / self.warmup_steps, 1)
        )
        self.grouped_model.clear()
        self.grouped_optimizer.clear()
        self.grouped_scheduler.clear()



    def merge_train_iteration(self, num_steps, no_prompt=False, env_name=None, action_mask_dim=8):
        train_losses = []
        gradient_norms=[]
        logs = dict()

        train_start = time.time()

        self.merged_model.train()
        for _ in range(num_steps):
            train_loss,gradient_norm = self.merge_train_step(no_prompt, env_name, action_mask_dim)
            train_losses.append(train_loss)
            gradient_norms.append(gradient_norm)
            if self.merged_scheduler is not None:
                self.merged_scheduler.step()
        
        logs['time/training'] = time.time() - train_start
        logs['training/merged_train_loss_mean'] = np.mean(train_losses)
        logs['training/merged_train_loss_std'] = np.std(train_losses)
        logs[f'training/merged_{env_name}_gradient_norm'] = np.mean(gradient_norms)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs
    
    def merge_train_step(self, no_prompt=False, index=None, action_mask_dim=None):
        prompt, batch = self.get_prompt_batch(index=index)
        states, actions, rewards, dones, rtg, timesteps, attention_mask, env_name = batch
        action_target = torch.clone(actions)
        if no_prompt:
            state_preds, action_preds, reward_preds = self.merged_model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=None
            )
        else:
            state_preds, action_preds, reward_preds = self.merged_model.forward(
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

        self.merged_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.merged_model.parameters(), .25)
        # gradient = parameters_to_gradvector(self.model)

        self.merged_optimizer.step()

        gradients_norm=[]
        for name, param in self.merged_model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    gradients_norm.append(param.grad.norm().cpu())

        gradient_norm=np.mean(gradients_norm)

        with torch.no_grad():
            self.diagnostics['training/merged_action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(), gradient_norm
    
    def group_train_iteration(self, num_steps, no_prompt=False, env_name=None, env_id=None, action_mask_dim=8):
        train_losses = []
        gradient_norms = []
        logs = dict()
        group_id=self.group_labels[env_id]
        train_start = time.time()
        
        self.grouped_model[group_id].train()
        for _ in range(num_steps):
            train_loss,gradient_norm = self.group_train_step(group_id, no_prompt, env_name, action_mask_dim)
            train_losses.append(train_loss)
            gradient_norms.append(gradient_norm)
            if self.grouped_scheduler[group_id] is not None:
                self.grouped_scheduler[group_id].step()
        
        logs['time/grouped_training'] = time.time() - train_start
        logs['training/grouped_train_loss_mean'] = np.mean(train_losses)
        logs['training/grouped_train_loss_std'] = np.std(train_losses)
        logs[f'training/grouped_{env_name}_gradient_norm'] = np.mean(gradient_norms)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs
    
    def group_train_step(self, group_id, no_prompt=False, index=None, action_mask_dim=None):
        prompt, batch = self.get_prompt_batch(index=index)
        states, actions, rewards, dones, rtg, timesteps, attention_mask, env_name = batch
        action_target = torch.clone(actions)
        if no_prompt:
            state_preds, action_preds, reward_preds = self.grouped_model[group_id].forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=None
            )
        else:
            state_preds, action_preds, reward_preds = self.grouped_model[group_id].forward(
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

        self.grouped_optimizer[group_id].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.grouped_model[group_id].parameters(), .25)
        # gradient = parameters_to_gradvector(self.model)

        self.grouped_optimizer[group_id].step()

        gradients_norm=[]
        for name, param in self.grouped_model[group_id].named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    gradients_norm.append(param.grad.norm().cpu())

        gradient_norm=np.mean(gradients_norm)

        with torch.no_grad():
            self.diagnostics['training/grouped_action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(), gradient_norm



    def eval_grouped_models_for_test(self, model, get_prompt, group_prompt_trajectories_list, eval_episodes, group_env_name_list, info, prompt_info, 
                                variant, group_env_list, iter_num=0, print_logs=False, no_prompt=False, group='test'):
        print('=' * 80)
        print('evaluate at task group: ', group_env_name_list)
        print('eval task num in group:', len(group_env_name_list))
        logs = dict()
        print('start evaluating...')
        model.eval()

        eval_start = time.time()
        for env_id, env_name in enumerate(group_env_name_list):
            # need to sample eval_fn and prompt together 
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, group_env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            # print(prompt_info[env_name])
            self.get_prompt = get_prompt(group_prompt_trajectories_list[env_id], prompt_info[env_name], variant)
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

 
    def save_model(self, env_name, postfix, folder, merge_training):
        if merge_training==True:
            model_name = '/prompt_model_' + env_name + postfix + '_merge'
            torch.save(self.merged_model.state_dict(),folder+model_name)  # model save
        else:
            model_name = '/prompt_model_' + env_name + postfix + '_group'
            model_dicts = {f'group_{i}': self.grouped_model[i].state_dict() for i in range(self.group_num)}
            task_dicts = {f'group_{i}': self.grouped_envs[i] for i in range(self.group_num)}
            save_dict = {"model_dict": model_dicts, "task_dicts": task_dicts}
            torch.save(save_dict, folder+model_name)
        print('model saved to ', folder+model_name)

    
    def update_gradient(self, no_prompt=False, env_name=None, action_mask_dim=None):

        prompt, batch = self.get_prompt_batch(index=env_name)
        states, actions, rewards, dones, rtg, timesteps, attention_mask_, env_name_ = batch
        action_target = torch.clone(actions)

        original_param = copy.deepcopy(self.merged_model.state_dict())
        # self.model.load_state_dict(original_param)
        # print(attention_mask_.shape)
        self.merged_model.train()

        if no_prompt:
            state_preds, action_preds, reward_preds = self.merged_model.forward(
                states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask_, prompt=None
            )
        else:
            state_preds, action_preds, reward_preds = self.merged_model.forward(
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

        self.merged_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.merged_model.parameters(), .25)

        gradient = parameters_to_gradvector(self.merged_model)

        self.merged_model.load_state_dict(original_param)

        return gradient
