import json
import random
import os
import numpy as np
import torch
import argparse
import random

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
    set_seed(variant['seed'])
    group_num=variant['group_num']
    seed = variant['seed']
    with open(f'config/SuperLong/task_160_seed{seed}.json','r') as file:
        data=json.load(file)
    task_list=data['mt160_task_list']
    random.shuffle(task_list)
    sublist_length = len(task_list) // group_num
    remainder = len(task_list) % group_num

    random_group={}
    index = 0
    expert_id=0
    for i,j in enumerate(range(remainder)):
        random_group[f'expert_{expert_id}']=task_list[index:index+sublist_length+1]
        index += sublist_length + 1
        expert_id+=1
    for i,j in enumerate(range(group_num-remainder)):
        random_group[f'expert_{expert_id}']=task_list[index:index+sublist_length]
        index += sublist_length
        expert_id+=1 

    with open(f'config/SuperLong/moe/random/random_grouped_task_{group_num}_seed{seed}.json','w') as file:
        json.dump(random_group,file,indent=2)

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
    parser.add_argument('--embed_dim', type=int, default=128)
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
    parser.add_argument('--model_path', type=str, default='model_saved_backbone/mt5-1-240904-115618')
    parser.add_argument('--group_method', type=str, default='gradient')

    args = parser.parse_args()
    variant=vars(args)

    experiment_mix_env(variant=vars(args))