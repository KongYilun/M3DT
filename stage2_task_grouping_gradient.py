import os
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
import itertools
from transformers import GPT2Config
from prompt_dt.prompt_decision_transformer import PromptDecisionTransformer
from prompt_dt.prompt_seq_trainer import PromptSequenceTrainer
from prompt_dt.prompt_group_trainer import PromptGroupMergeTrainer
from prompt_dt.prompt_utils import get_env_list, report_parameters, get_env_list_for_long_task, get_action_mask_dim, plot_gradient_cluster
from prompt_dt.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune
from prompt_dt.prompt_utils import process_total_data_mean, load_data_prompt, process_info
from prompt_dt.prompt_utils import eval_episodes
from prompt_dt.prompt_utils import get_avg_gradient, run_kmeans_multiple_times, KMeans

from collections import namedtuple
import json, pickle, os
import time
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
    timestr = time.strftime("%y%m%d-%H%M%S")
    print(timestr)
    print(seed)
    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal') 
    set_seed(variant['seed'])
    group_num=variant['group_num']
    state_dim=39
    act_dim=8
    data_save_path = variant['data_path']
    device = variant['device']
    model_scale=variant['model_scale']
    # group_method=variant['group_method']

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
    # report_parameters(model)
    model_path=f'model_saved_backbone/prompt_model_mt160_iter_400000_seed{seed}'
    print(model_path)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device=device)

    task_config = os.path.join('config', f"SuperLong/task_160_seed{seed}.json")
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    if '160' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.mt160_task_list, task_config.mt160_task_list
    elif '80' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.mt80_task_list, task_config.mt80_task_list
    elif '40' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.mt40_task_list, task_config.mt40_task_list
    elif '20' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.mt20_task_list, task_config.mt20_task_list
    elif '10' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.mt10_task_list, task_config.mt10_task_list
    elif '5' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.mt5_task_list, task_config.mt5_task_list
    else:
        raise NotImplementedError("please use the 160, 80, 40, 20, 10, 5")
    info = get_env_list_for_long_task(train_env_name_list, device, seed=seed,total_env='train')


    # random_group=variant['random_group']
    # print('random group:', random_group)
    print('#########loading data##########')
    start_time=time.time()
    trajectories_list, prompt_trajectories_list = load_data_prompt(train_env_name_list, data_save_path)
    print('load data cost time:',time.time()-start_time,'\n')


    prompt_info = copy.deepcopy(info)
    prompt_info = process_info(train_env_name_list, prompt_trajectories_list, prompt_info, mode, pct_traj, variant)
    info = process_info(train_env_name_list, trajectories_list, info, mode, pct_traj, variant)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=batch_size,
            get_batch=get_batch(trajectories_list[0], info[env_name], variant),
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=None,
            get_prompt=get_prompt(prompt_trajectories_list[0], prompt_info[env_name], variant),
            get_prompt_batch=get_prompt_batch(trajectories_list, prompt_trajectories_list, info, prompt_info, variant, train_env_name_list),
        )
 
    gradient_set={}
    gradient_list=[]
    for i,_env_name in enumerate(train_env_name_list):
        tmp_action_mask_dim=get_action_mask_dim(_env_name)
        avg_gradient=[]
        for _ in range(10): 
            gradient,shapes = trainer.update_gradient(
                no_prompt=args.no_prompt,
                env_name = _env_name,
                action_mask_dim=tmp_action_mask_dim
            )
            avg_gradient.append(gradient)
        avg_gradient = torch.stack(avg_gradient)
        avg_gradient = torch.mean(avg_gradient, dim=0)
        gradient_set[_env_name]=avg_gradient
        gradient_list.append(avg_gradient.cpu().numpy())
    gradient_list=np.array(gradient_list)
    gradient_list=torch.tensor(gradient_list)
    cols_all_zeros = torch.all(gradient_list == 0, dim=0)

    # 创建一个布尔掩码，选择非全0的列
    mask = ~cols_all_zeros

    # 使用布尔掩码选择非全0的列
    gradient_list = gradient_list[:, mask]
    avg_gradient=get_avg_gradient(gradient_list)
    conflicts=[]
    for i in gradient_list:
        conflicts.append(avg_gradient*i)
    conflicts=torch.stack(conflicts).cuda()
    min_std=10000
    final_group=None
    for _ in range(40):
        final_centroids, final_labels = run_kmeans_multiple_times(conflicts.cuda(), group_num)
        l=[[] for _ in range(group_num)]
        for i in range(len(final_labels)):
            l[final_labels[i]].append(i)
        ll=[]
        for i in l:
            ll.append(len(i))
        std=np.var(ll)
        if std< min_std:
            min_std=std
            final_group=copy.deepcopy(l)

    for i in final_group:
        print(len(i),i)
    dic={}
    for i in range(len(final_group)):
        dic[f'expert_{i}']=[train_env_name_list[j] for j in final_group[i]]

    with open(f'config/SuperLong/moe/gradient/gradient_grouped_task_{group_num}_seed{seed}.json','w') as file:
        json.dump(dic,file,indent=2)





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
    parser.add_argument('--max_iters', type=int, default=1000000) 
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--test_eval_interval', type=int, default=10000)
    parser.add_argument('--save-interval', type=int, default=10000)
    parser.add_argument('--conflict-interval', type=int, default=10000)
    parser.add_argument('--group-interval', type=int, default=200000)
    parser.add_argument('--group_num', type=int, default=16)
    parser.add_argument('--model_scale', type=int, default=5)
    parser.add_argument('--random_group', action='store_true', default=False)
    parser.add_argument('--data_path', type=str, default='mt160_used')
    parser.add_argument('--group_method', type=str, default='gradient')

    args = parser.parse_args()
    variant=vars(args)

    experiment_mix_env(variant=vars(args))