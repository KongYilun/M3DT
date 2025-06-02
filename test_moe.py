import os
os.environ['WANDB_API_KEY']='227446d2817003989abaf27aff2159a908b483e2'
os.environ["WANDB_MODE"] = "offline"
from ast import parse
import gc
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
from prompt_dt.prompt_decision_transformer_with_moe import PromptDTMoE
from prompt_dt.prompt_seq_trainer import PromptSequenceTrainer
from prompt_dt.prompt_utils import get_env_list, report_parameters, load_prompt_for_test
from prompt_dt.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune
from prompt_dt.prompt_utils import process_total_data_mean, load_data_prompt, process_info
from prompt_dt.prompt_utils import eval_episodes
from prompt_dt.prompt_utils import get_avg_gradient, gradvector_to_parameters

from collections import namedtuple
import json, pickle, os
import time
from sklearn.manifold import TSNE



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
    print(seed)
    cur_dir = os.getcwd()
    timestr = time.strftime("%y%m%d-%H%M%S")

    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal')
    
    set_seed(variant['seed'])
    env_name_ = variant['env']

    state_dim=39
    act_dim=8
    data_save_path = variant['data_path']
    device = variant['device']
    model_scale = variant['model_scale']
    
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
    model.load_state_dict(torch.load(f'model_saved_backbone/prompt_model_mt160_iter_400000_seed{seed}'))
    model = model.to(device=device)

    expert_params=[]
    expert_num=variant['expert_num']
    for i in range(expert_num):
        expert_params.append(torch.load(f'model_saved_expert/random_group_{expert_num}_seed{seed}/expert_{i}_iter_200000'))#
        print('load expert:', f'model_saved_expert/random_group_{expert_num}_seed{seed}/expert_{i}_iter_200000')

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
    moe_path=f'model_saved_moe/mt160_random_group56_expert56_5M_seed0_250126-134044/moe__iter_400000'
    model_with_moe.load_moe(moe_path)#model_expert_saved/experts_seed1/expert_{expert_id}_iter_100000
    model_with_moe = model_with_moe.to(device=device)
    print('load moe:', moe_path)
    subset_id=variant['subset_id']
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
    test_info, test_env_list = get_env_list(test_env_name_list, device, seed=seed,total_env='test')
    
    exp_prefix = f'{env_name_}-{seed}-{timestr}'
    group_name = variant['prefix_name']


    print('#########loading data##########')
    start_time=time.time()
    # trajectories_list, prompt_trajectories_list = load_data_prompt(train_env_name_list, data_save_path)
    test_prompt_trajectories_list = load_prompt_for_test(test_env_name_list, data_save_path)
    # test_trajectories_list, test_prompt_trajectories_list = load_data_prompt(test_env_name_list, data_save_path)
    print('load data cost time:',time.time()-start_time,'\n')



    prompt_test_info = copy.deepcopy(test_info)
    prompt_test_info = process_info(test_env_name_list, test_prompt_trajectories_list, prompt_test_info, mode, pct_traj, variant)
    with open('test_info_mt160_new.pkl', 'rb') as f:
        test_info = pickle.load(f)
    

    env_name = train_env_name_list[0]
    trainer = PromptSequenceTrainer(
            model=model_with_moe,
            optimizer=None,
            batch_size=batch_size,
            get_batch=None,
            scheduler=None,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=None,
            get_prompt=get_prompt(test_prompt_trajectories_list[0], prompt_test_info[env_name], variant),
            get_prompt_batch=None
        )
    

    num_env = len(train_env_name_list)

    best_ret = -10000
    best_iter = 0
    outputs = dict()
    start_time=time.time()
    test_eval_logs = trainer.eval_iteration_multienv_for_test(
        model_with_moe,
        get_prompt, test_prompt_trajectories_list,
        eval_episodes, test_env_name_list, test_info, prompt_test_info, variant, test_env_list, iter_num=0, 
        print_logs=True, no_prompt=args.no_prompt, group='test')
    outputs.update(test_eval_logs)
    total_normalized_score_mean = test_eval_logs[f'test-Total-Normalized-Score-Mean']
    print('#'*80)
    print("Normalized Score:", total_normalized_score_mean)
    print("Cost Time:", time.time()-start_time)
    print('#'*80)

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
    parser.add_argument('--conflict-interval', type=int, default=100)
    parser.add_argument('--expert_num', type=int, default=16)
    parser.add_argument('--subset_id', type=int, default=0)
    parser.add_argument('--model_scale', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='mt160_used')
    parser.add_argument('--model_path', type=str, default='mt5-1-240904-115618')

    args = parser.parse_args()
    variant=vars(args)

    experiment_mix_env(variant=vars(args))