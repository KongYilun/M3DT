import os
# os.environ["WANDB_MODE"] = "offline"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from ast import parse
import gym
import numpy as np
import torch
import wandb
import copy
import argparse
import pickle
import random
import sys
import time
from tqdm import tqdm
import itertools
from transformers import GPT2Config
from prompt_dt.prompt_decision_transformer import PromptDecisionTransformer
from prompt_dt.prompt_decision_transformer_with_moe import PromptDTMoE
from prompt_dt.prompt_seq_trainer import PromptSequenceTrainer
from prompt_dt.prompt_utils import get_env_list, report_parameters, get_env_list_for_long_task, get_action_mask_dim, plot_gradient_cluster, report_training_parameters
from prompt_dt.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune
from prompt_dt.prompt_utils import process_total_data_mean, load_data_prompt, process_info
from prompt_dt.prompt_utils import eval_episodes
from prompt_dt.prompt_utils import get_avg_gradient, gradvector_to_parameters

from collections import namedtuple
import json, pickle, os
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import torch.nn.functional as F

import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def experiment_mix_env(
        variant,
):
    seed = variant['seed']
    cur_dir = os.getcwd()
    save_path = os.path.join(cur_dir, f'model_saved_moe/')
    if not os.path.exists(save_path): os.mkdir(save_path)
    timestr = time.strftime("%y%m%d-%H%M%S")
    print(timestr)
    print(seed)
    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal') 
    set_seed(variant['seed'])
    env_name_ = variant['env']
    model_scale=variant['model_scale']
    state_dim=39
    act_dim=8
    data_save_path = variant['data_path']
    device = variant['device']

    model = PromptDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=1000,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    report_parameters(model)
    model.load_state_dict(torch.load(f'model_saved_backbone/prompt_model_mt160_iter_400000_seed{seed}'))
    print('load backbone:',f'model_saved_backbone/prompt_model_mt160_iter_400000_seed{seed}')
    model = model.to(device=device)

    expert_params=[]
    expert_num=variant['expert_num']
    for i in range(expert_num):
        expert_params.append(torch.load(f'model_saved_expert/gradient_group_{expert_num}_seed{seed}/expert_{i}_iter_200000'))
        print('load expert:', f'model_saved_expert/gradient_group_{expert_num}_seed{seed}/expert_{i}_iter_200000')

    model_with_moe=PromptDTMoE(
        pretrained_model=model,
        experts_params=expert_params,
        top_k=None,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=1000,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    report_parameters(model_with_moe)
    report_training_parameters(model_with_moe)
    model_with_moe = model_with_moe.to(device=device)
    for name, param in model_with_moe.named_parameters():
        if param.requires_grad==True:
            print(name)

    task_config = os.path.join('config', f"SuperLong/task_160_seed{seed}.json")
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    # train_env_name_list, test_env_name_list = task_config.group_task_list_52, task_config.group_task_list_52
    if '160' in variant['prefix_name']:
        print('loading 160 tasks!')
        train_env_name_list, test_env_name_list = task_config.mt160_task_list, task_config.mt160_task_list
    elif '10' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.mt10_task_list, task_config.mt10_task_list
    else:
        raise NotImplementedError("please use the 160, 80, 40, 20, 10, 5")
    info = get_env_list_for_long_task(train_env_name_list, device, seed=seed,total_env='train')
    # test_info = get_env_list_for_long_task(test_env_name_list, device, seed=seed,total_env='test')

    exp_prefix = f"{variant['prefix_name']}_expert{expert_num}_seed{seed}_{timestr}"
    group_name = variant['prefix_name']


    print('#########loading data##########')
    start_time=time.time()
    trajectories_list, prompt_trajectories_list = load_data_prompt(train_env_name_list, data_save_path)
    print('load data cost time:',time.time()-start_time,'\n')
    prompt_info = copy.deepcopy(info)
    prompt_info = process_info(train_env_name_list, prompt_trajectories_list, prompt_info, mode, pct_traj, variant)
    with open('test_info_mt160_new.pkl', 'rb') as f:
        info = pickle.load(f)
    # info = process_info(train_env_name_list, trajectories_list, info, mode, pct_traj, variant)
    # with open('test_info_mt160_used.pkl', 'wb') as f:
    #     pickle.dump(info, f)
    # exit()
    
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model_with_moe.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    num_env = len(train_env_name_list)
    env_name = train_env_name_list[0]
    trainer = PromptSequenceTrainer(
            model=model_with_moe,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch(trajectories_list[0], info[env_name], variant),
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=None,
            get_prompt=get_prompt(prompt_trajectories_list[0], prompt_info[env_name], variant),
            get_prompt_batch=get_prompt_batch(trajectories_list, prompt_trajectories_list, info, prompt_info, variant, train_env_name_list)
        )
    log_to_wandb = variant['log_to_wandb']
    if log_to_wandb:
            # wandb.init(
            #     name=exp_prefix,
            #     group=group_name,
            #     project='slt',
            #     config=variant,
            #     dir='wandb',
            #     settings=wandb.Settings(silent="true")
            # )
            save_path += exp_prefix
            os.makedirs(save_path, exist_ok=True)

    
    start_time=time.time()
    with tqdm(total=variant['max_iters'], desc="Training Progress", unit="iter") as pbar:
        for iter in range(variant['max_iters']):#variant['max_iters']variant['max_iters']
            env_id = iter % num_env
            env_name = train_env_name_list[env_id]
            action_mask_dim=get_action_mask_dim(env_name)

            outputs = trainer.pure_train_iteration_mix(
                num_steps=1, #variant['num_steps_per_iter']
                no_prompt=args.no_prompt,
                env_name=env_name,
                action_mask_dim=action_mask_dim
                )
            outputs.update({"global_step": iter+1}) # set global step as iteration
            # 更新进度条
            pbar.update(1)
            # 估计剩余时间
            elapsed_time = time.time() - start_time
            iterations_done = iter + 1
            iterations_left = variant['max_iters'] - iterations_done
            estimated_total_time = (elapsed_time / iterations_done) * variant['max_iters']
            estimated_remaining_time = estimated_total_time - elapsed_time

            # 更新进度条的描述
            pbar.set_description(f"Training Progress (ETA: {estimated_remaining_time:.2f}s)")
            if (iter+1) % variant['save_interval'] == 0:
                model_with_moe.save_moe(postfix='_iter_'+str(iter+1),folder=save_path)
            # if log_to_wandb:
            #     wandb.log(outputs)
    
    # trainer.save_model(env_name='total_model', postfix='_iter_'+str(100000),folder=save_path)
    # wandb.finish()
    model_with_moe.save_moe(postfix='_iter_'+str(iter+1),folder=save_path)
    print('total_time:', time.time()-start_time)
    # trainer.save_model(env_name=args.prefix_name,  postfix='_iter_'+str(iter+1),  folder=save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Super Long') # ['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']
    parser.add_argument('--prefix_name', type=str, default='mt5')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--prompt-episode', type=int, default=1)
    parser.add_argument('--prompt-length', type=int, default=5)
    parser.add_argument('--stochastic-prompt', action='store_true', default=False)
    parser.add_argument('--no-prompt', action='store_true', default=False)
    parser.add_argument('--no-r', action='store_true', default=False)
    parser.add_argument('--no-rtg', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--finetune_steps', type=int, default=10)
    parser.add_argument('--finetune_batch_size', type=int, default=256)
    parser.add_argument('--finetune_opt', action='store_true', default=True)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--no_state_normalize', action='store_true', default=False) 
    parser.add_argument('--average_state_mean', action='store_true', default=False) 
    parser.add_argument('--evaluation', action='store_true', default=False) 
    parser.add_argument('--render', action='store_true', default=False) 
    parser.add_argument('--load-path', type=str, default= None) # choose a model when in evaluation mode

    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000) # 10000*(number of environments)
    parser.add_argument('--num_eval_episodes', type=int, default=10) 
    parser.add_argument('--max_iters', type=int, default=400000) 
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--test_eval_interval', type=int, default=10000)
    parser.add_argument('--save-interval', type=int, default=100000)
    parser.add_argument('--conflict-interval', type=int, default=10000)
    parser.add_argument('--group-interval', type=int, default=200000)
    parser.add_argument('--expert_num', type=int, default=16)
    parser.add_argument('--model_scale', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='/amax/datasets/kyl_data/mt160_used')

    args = parser.parse_args()
    variant=vars(args)

    experiment_mix_env(variant=vars(args))