import numpy as np
import re
import gym
import json, pickle, random, os, torch
from collections import namedtuple
from .prompt_evaluate_episodes import prompt_evaluate_episode, prompt_evaluate_episode_rtg
import sys
# for mujoco tasks
from mujoco_control_envs.mujoco_control_envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv
# for jacopinpad
from jacopinpad.jacopinpad_gym import jacopinpad_multi
# for metaworld
import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'envs')))
from envs.dmcontrol import make_dmcontrol_env

import matplotlib.pyplot as plt



class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        # Randomly initialize centroids
        self.centroids = X[torch.randperm(X.size(0))[:self.n_clusters], :]

        for i in range(self.max_iter):
            # Assign each data point to the nearest centroid
            distances = torch.cdist(X, self.centroids)#(X[:, None] - self.centroids[None, :]).pow(2).sum(2)
            closest_centroids = distances.argmin(1)

            # Update centroids
            new_centroids = torch.stack([X[closest_centroids == k].mean(0) for k in range(self.n_clusters)])
            
            # Check for convergence
            if torch.all(torch.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

    def predict(self, X):
        # Assign data points to the nearest centroid
        distances = torch.cdist(X, self.centroids)#(X[:, None] - self.centroids[None, :]).pow(2).sum(2)
        return distances.argmin(1)


# class KMeans:
#     def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, batch_size=None):
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter
#         self.tol = tol
#         self.centroids = None
#         self.batch_size = batch_size  # 可以设置批次大小

#     def fit(self, X):
#         # Randomly initialize centroids
#         self.centroids = X[torch.randperm(X.size(0))[:self.n_clusters], :].clone()

#         if self.batch_size is None or self.batch_size > X.size(0):
#             self.batch_size = X.size(0)

#         for i in range(self.max_iter):
#             # Assign each data point to the nearest centroid
#             distances = torch.cdist(X, self.centroids)#(X[:, None] - self.centroids[None, :]).pow(2).sum(1)
#             closest_centroids = distances.argmin(0)

#             # Update centroids using batches to avoid OOM
#             new_centroids = torch.zeros_like(self.centroids)
#             for j in range(0, X.size(0), self.batch_size):
#                 batch_closest_centroids = closest_centroids[j:j + self.batch_size]
#                 for k in range(self.n_clusters):
#                     indices = batch_closest_centroids == k
#                     if torch.any(indices):
#                         new_centroids[k] = torch.mean(X[j:j + self.batch_size][indices], dim=0)

#             # Check for convergence
#             if torch.all(torch.abs(new_centroids - self.centroids) < self.tol):
#                 break
#             self.centroids = new_centroids

#     def predict(self, X):
#         # Assign data points to the nearest centroid
#         distances = (X[:, None] - self.centroids[None, :]).pow(2).sum(2)
#         return distances.argmin(1)

def kmeans(data, K, centroids, max_iters=100, tol=1e-4):
    for i in range(max_iters):
        # 计算每个点到每个中心的距离
        distances = torch.cdist(data, centroids)
        
        # 找到最近的中心点
        labels = torch.argmin(distances, dim=1)
        
        # 更新中心点
        new_centroids = torch.stack([data[labels == k].mean(dim=0) for k in range(K)])
        # print(new_centroids.shape)
        # print(centroids.shape)
        # if new_centroids.shape!=centroids.shape:
        #     continue
        # 检查中心点是否收敛
        if torch.all(torch.abs(new_centroids - centroids) < tol):
            break
        
        centroids = new_centroids
    
    return centroids, labels

def kmeans_plusplus_init(data, K):
    centroids = []
    # 随机选择第一个中心点
    centroids.append(data[torch.randint(0, data.size(0), (1,))].squeeze())
    
    for _ in range(1, K):
        # 计算每个点到当前所有中心点的最小距离的平方
        distances = torch.cdist(data, torch.stack(centroids))
        min_distances = distances.min(dim=1)[0]
        min_distances_sq = min_distances ** 2
        min_distances_sq[min_distances_sq == 0] = 1e-10
        # 根据距离的平方作为权重，随机选择下一个中心点
        prob = min_distances_sq / min_distances_sq.sum()
        has_nan = torch.isnan(prob).any().item()
        if has_nan==True:
            break
        next_centroid_idx = torch.multinomial(prob, 1)
        centroids.append(data[next_centroid_idx].squeeze())
    
    return torch.stack(centroids)

def run_kmeans_multiple_times(data, K, num_runs=40, max_iters=100, tol=1e-4):
    best_centroids = None
    best_labels = None
    best_distance = float('inf')
    
    for _ in range(num_runs):
        # 使用 K-means++ 初始化中心点
        indices = torch.randperm(data.size(0))[:K]
        # print(indices)
        centroids = data[indices]#kmeans_plusplus_init(data, K)##data[indices]##
        centroids = centroids.cuda()  # 将中心点移动到GPU
        # print(centroids)
        # 运行 K-means
        # print(centroids.shape)
        final_centroids, final_labels = kmeans(data, K, centroids, max_iters, tol)
        # print(final_labels)
        # 计算总距离
        distances = torch.cdist(data, final_centroids)
        total_distance = distances.min(dim=1)[0].sum().item()
        
        # 更新最优结果
        if total_distance < best_distance:
            best_distance = total_distance
            best_centroids = final_centroids
            best_labels = final_labels
    
    return best_centroids, best_labels

def parameters_to_gradvector(net):
    vec = []
    shapes = []
    for name, param in net.named_parameters():
        if param.grad !=None:
            vec.append(param.grad.view(-1))
        else:       
            vec.append(torch.zeros(param.size()).view(-1).cuda())
        shapes.append(param.shape)
    return torch.cat(vec),shapes

def parameters_to_gradvector_all(net):
    router_vec = []
    expert_vec=[[] for _ in range(net.expert_num)]
    backbone_vec=[]
    for name, param in net.named_parameters():
        if param.grad !=None:
            if 'router' in name:
                router_vec.append(param.grad.view(-1))
            elif 'experts' in name:
                match = re.search(r'experts\.(\d+)', name)
                expert_id = int(match.group(1))
                expert_vec[expert_id].append(param.grad.view(-1))
            else:
                backbone_vec.append(param.grad.view(-1))
        else:     
            if 'router' in name:
                router_vec.append(torch.zeros(param.size()).view(-1).cuda())
            elif 'experts' in name:
                match = re.search(r'experts\.(\d+)', name)
                expert_id = int(match.group(1))
                expert_vec[expert_id].append(torch.zeros(param.size()).view(-1).cuda())
            else:
                backbone_vec.append(torch.zeros(param.size()).view(-1).cuda())
    expert_vec_cat=[torch.cat(expert_vec[i]) for i in range(net.expert_num)]
    return torch.cat(backbone_vec), torch.stack(expert_vec_cat,dim=0), torch.cat(router_vec)

def parameters_to_gradvector_expert(net):
    expert_vec=[]
    for name, param in net.named_parameters():
        if param.grad !=None:
            if 'experts' in name:
                expert_vec.append(param.grad.view(-1))
        else:     
            if 'experts' in name:
                expert_vec.append(torch.zeros(param.size()).view(-1).cuda())
    return torch.cat(expert_vec)

def parameters_to_gradvector_router(net):
    expert_vec=[]
    for name, param in net.named_parameters():
        if param.grad !=None:
            if 'router' in name:
                expert_vec.append(param.grad.view(-1))
        else:     
            if 'router' in name:
                expert_vec.append(torch.zeros(param.size()).view(-1).cuda())
    return torch.cat(expert_vec)


def gradvector_to_parameters(vec, shapes):
    params = []
    start = 0
    for shape in shapes:
        size = torch.prod(torch.tensor(shape)).item()
        param = vec[start:start+size].view(shape)
        params.append(param)
        start += size
    return torch.tensor(params)

def get_avg_gradient(gradient_list):
    harmo_gradient=torch.zeros_like(gradient_list[0])
    for i in range(len(gradient_list)):
        harmo_gradient+=gradient_list[i]
    harmo_gradient/=len(gradient_list)
    return harmo_gradient

""" constructing envs """

def gen_env(env_name, seed=1, num_eval_episodes=0,total_env=None):
    # if 'cheetah_dir' in env_name:
    #     if '0' in env_name:
    #         env = HalfCheetahDirEnv([{'direction': 1}], include_goal = False)
    #     elif '1' in env_name:
    #         env = HalfCheetahDirEnv([{'direction': -1}], include_goal = False)
    #     max_ep_len = 200
    #     env_targets = [1500]
    #     scale = 1000.
    config_save_path='config'
    if 'cheetah-vel' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/cheetah_vel/config_cheetah_vel_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        env = HalfCheetahVelEnv(tasks, include_goal = False)
        env.seed(seed)
        max_ep_len = 200
        env_targets = [0]
        scale = 200.
    elif 'ant-dir' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/ant_dir/config_ant_dir_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        env = AntDirEnv(tasks, len(tasks), include_goal = False)
        env.seed(seed)
        max_ep_len = 200
        env_targets = [500]
        scale = 500.
    elif '-v2' in env_name: # metaworld ML1
        task = metaworld.MT1(env_name).train_tasks[0]
        env = metaworld.MT1(env_name).train_classes[env_name]()
        env.set_task(task)
        env.seed(seed)
        max_ep_len = 500
        env_targets = [4500]
        scale = 1000.
        if 'test' in total_env:
            task = [metaworld.MT1(env_name).train_tasks[i] for i in range(num_eval_episodes)]
            mt1 = [metaworld.MT1(env_name) for i in range(num_eval_episodes)]
            env_list = [mt1[i].train_classes[env_name]() for i in range(num_eval_episodes)]
            for i in range(len(env_list)):
                env_list[i].set_task(task[i])
                env_list[i].seed(seed)
            env = env_list
    else:
        env=make_dmcontrol_env(env_name,seed)
        max_ep_len = 500 
        env_targets= [1000]
        scale = 200.
        if 'test' in total_env:
            env = [make_dmcontrol_env(env_name,seed=i) for i in range(num_eval_episodes)]
    return env, max_ep_len, env_targets, scale

def gen_env_for_long_task(env_name, seed=1, num_eval_episodes=0,total_env=None):
    # if 'cheetah_dir' in env_name:
    #     if '0' in env_name:
    #         env = HalfCheetahDirEnv([{'direction': 1}], include_goal = False)
    #     elif '1' in env_name:
    #         env = HalfCheetahDirEnv([{'direction': -1}], include_goal = False)
    #     max_ep_len = 200
    #     env_targets = [1500]
    #     scale = 1000.
    config_save_path='config'
    if 'cheetah-vel' in env_name:
        max_ep_len = 200
        env_targets = [0]
        scale = 200.
    elif 'ant-dir' in env_name:
        max_ep_len = 200
        env_targets = [500]
        scale = 500.
    elif '-v2' in env_name: # metaworld ML1

        max_ep_len = 500
        env_targets = [4500]
        scale = 1000.
    else:
        max_ep_len = 500 
        env_targets= [1000]
        scale = 200.
    return max_ep_len, env_targets, scale

def get_env_list(env_name_list, device, seed=1, num_eval_episodes=10,total_env=None):
    info = {} # store all the attributes for each env
    env_list = []
    
    for env_name in env_name_list:
        info[env_name] = {}
        env, max_ep_len, env_targets, scale = gen_env(env_name=env_name, seed=seed, num_eval_episodes=num_eval_episodes,total_env=total_env)
        info[env_name]['max_ep_len'] = max_ep_len
        info[env_name]['env_targets'] = env_targets
        info[env_name]['scale'] = scale
        # if type(env) is list:
        #     info[env_name]['state_dim'] = env[0].observation_space.shape[0]
        #     info[env_name]['act_dim'] = env[0].action_space.shape[0] 
        # else:
        info[env_name]['state_dim'] = 39
        info[env_name]['act_dim'] = 8
        info[env_name]['device'] = device
        env_list.append(env)
    return info, env_list

def get_env_list_for_long_task(env_name_list, device, seed=1, num_eval_episodes=10,total_env=None):
    info = {} # store all the attributes for each env
    
    for env_name in env_name_list:
        info[env_name] = {}
        max_ep_len, env_targets, scale = gen_env_for_long_task(env_name=env_name, seed=seed, num_eval_episodes=num_eval_episodes,total_env=total_env)
        info[env_name]['max_ep_len'] = max_ep_len
        info[env_name]['env_targets'] = env_targets
        info[env_name]['scale'] = scale
        # if type(env) is list:
        #     info[env_name]['state_dim'] = env[0].observation_space.shape[0]
        #     info[env_name]['act_dim'] = env[0].action_space.shape[0] 
        # else:
        info[env_name]['state_dim'] = 39
        info[env_name]['act_dim'] = 8
        info[env_name]['device'] = device
    return info

def get_action_mask_dim(env_name):
    with open('config/SuperLong/action_mask.json','r') as file:
        action_mask_dict=json.load(file)
    if "ant-dir" in env_name:
        return action_mask_dict['ant-dir']
    elif "cheetah-vel" in env_name:
        return action_mask_dict['cheetah-vel']
    elif "-v2" in env_name:
        return action_mask_dict['meta-world']
    else:
        return action_mask_dict[env_name]

""" prompts """

def flatten_prompt(prompt, batch_size):
    p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
    p_s = p_s.reshape((batch_size, -1, p_s.shape[-1]))
    p_a = p_a.reshape((batch_size, -1, p_a.shape[-1]))
    p_r = p_r.reshape((batch_size, -1, p_r.shape[-1]))
    p_d = p_d.reshape((batch_size, -1))
    p_rtg = p_rtg[:,:-1,:]
    p_rtg = p_rtg.reshape((batch_size, -1, p_rtg.shape[-1]))
    p_timesteps = p_timesteps.reshape((batch_size, -1))
    p_mask = p_mask.reshape((batch_size, -1)) 
    return p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask


def get_prompt(prompt_trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_episodes, max_len = variant['prompt_episode'], variant['prompt_length']

    def fn(sample_size=1, index=None):
        # random sample prompts with fixed length (prompt-length) in num episodes (prompt-episode)
        batch_inds = np.random.choice(
            np.arange(len(prompt_trajectories)),
            size=int(num_episodes*sample_size),
            replace=True,
            # p=p_sample,  # reweights so we sample according to timesteps
        )
        assert len(prompt_trajectories) == len(sorted_inds)
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

        for i in range(int(num_episodes*sample_size)):
            if variant["stochastic_prompt"]:
                traj = prompt_trajectories[int(batch_inds[i])] # random select traj
            else:
                if i > len(sorted_inds):
                    i = 1
                traj = prompt_trajectories[int(sorted_inds[-i])] # select the best traj with highest rewards
                # traj = prompt_trajectories[i]
            if index is not None:
                traj = prompt_trajectories[int(sorted_inds[index])]
            si = max(0, traj['rewards'].shape[0] - max_len -1) # select the last traj with length max_len

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            # d.append(None)
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return s, a, r, d, rtg, timesteps, mask

    return fn

def get_prompt_batch(trajectories_list, prompt_trajectories_list, info, prompt_info, variant, train_env_name_list):
    per_env_batch_size = variant['batch_size']

    def fn(batch_size=per_env_batch_size, index=None):
        env_id = train_env_name_list.index(index)
        env_name = index
        if prompt_trajectories_list:
            get_prompt_fn = get_prompt(prompt_trajectories_list[env_id], prompt_info[env_name], variant)
        else:
            get_prompt_fn = get_prompt(trajectories_list[env_id], info[env_name], variant)
        get_batch_fn = get_batch(trajectories_list[env_id], info[env_name], variant) 
        prompt = flatten_prompt(get_prompt_fn(batch_size), batch_size)
        batch = get_batch_fn(batch_size=batch_size)

        p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt

        s, a, r, d, rtg, timesteps, mask = batch
        if variant['no_r']:
            r = torch.zeros_like(r)
        if variant['no_rtg']:
            rtg = torch.zeros_like(rtg)

        prompt = p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask
        batch = s, a, r, d, rtg, timesteps, mask, env_name
        return prompt, batch

    return fn


""" batches """

def get_batch(trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K = variant['batch_size'], variant['K']

    def fn(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            # d.append(None)
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # TODO: why mask only has several zeros

        return s, a, r, d, rtg, timesteps, mask

    return fn


def get_batch_finetune(trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K = variant['batch_size'], variant['prompt_length'] # use the same amount of data for funetuning

    def fn(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            si = max(0, traj['rewards'].shape[0] - max_len -1) # select the last traj with length max_len

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            # if 'terminals' in traj:
            #     d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            # else:
            #     d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            d.append(None)
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # TODO: why mask only has several zeros

        return s, a, r, d, rtg, timesteps, mask

    return fn

""" data processing """

def process_total_data_mean(trajectories, mode):

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    return state_mean, state_std


def process_dataset(trajectories, mode, env_name, pct_traj):
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    # print('=' * 50)
    # print(f'Starting new experiment: {env_name}')
    # print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    # print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    # print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    reward_info = [np.mean(returns), np.std(returns), np.max(returns), np.min(returns)]
    # print(f'num_trajectories: {num_trajectories}')
    # print('=' * 50)
    return trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info


def load_data_prompt(env_name_list, data_save_path):
    trajectories_list = []
    prompt_trajectories_list = []
    for env_name in env_name_list:
        path = os.path.join(data_save_path,env_name)
        cur_task_trajs = []
        length = len([file for file in os.listdir(path) if file.endswith('.npz')])
        length=min(length,2000)
        for i in range(length-2):
            cur_path = os.path.join(path, f"{i}.npz")
            with open(cur_path, 'rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
            cur_task_trajs.append(episode)
        trajectories_list.append(cur_task_trajs)

        cur_task_prompt_trajs = []
        for i in range(length-2, length):
            cur_path = os.path.join(path, f"{i}.npz")
            with open(cur_path, 'rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
            cur_task_prompt_trajs.append(episode)
        prompt_trajectories_list.append(cur_task_prompt_trajs)
    
    return trajectories_list, prompt_trajectories_list


def load_data_prompt_for_few_shot_generalize(env_name_list, data_save_path):
    trajectories_list = []
    prompt_trajectories_list = []
    for env_name in env_name_list:
        path = os.path.join(data_save_path,env_name)
        cur_task_trajs = []
        length = len([file for file in os.listdir(path) if file.endswith('.npz')])
        _length=min(length,10)
        prompt_length=min(length,2000)

        for i in range(_length):
            cur_path = os.path.join(path, f"{i}.npz")
            with open(cur_path, 'rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
            cur_task_trajs.append(episode)
        trajectories_list.append(cur_task_trajs)

        cur_task_prompt_trajs = []
        for i in range(prompt_length-2, prompt_length):
            cur_path = os.path.join(path, f"{i}.npz")
            print(cur_path)
            with open(cur_path, 'rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
            cur_task_prompt_trajs.append(episode)
        prompt_trajectories_list.append(cur_task_prompt_trajs)
    
    return trajectories_list, prompt_trajectories_list


def load_prompt_for_test(env_name_list, data_save_path):
    prompt_trajectories_list = []
    for env_name in env_name_list:
        path = os.path.join(data_save_path,env_name)
        length = len([file for file in os.listdir(path) if file.endswith('.npz')])
        length=min(length,2000)

        cur_task_prompt_trajs = []
        for i in range(length-2, length):
            cur_path = os.path.join(path, f"{i}.npz")
            print(cur_path)
            with open(cur_path, 'rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
            cur_task_prompt_trajs.append(episode)
        prompt_trajectories_list.append(cur_task_prompt_trajs)
    
    return prompt_trajectories_list


def process_info(env_name_list, trajectories_list, info, mode, pct_traj, variant):
    for i, env_name in enumerate(env_name_list):
        trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info = process_dataset(
            trajectories=trajectories_list[i], mode=mode, env_name=env_name_list[i], pct_traj=pct_traj)
        info[env_name]['num_trajectories'] = num_trajectories
        info[env_name]['sorted_inds'] = sorted_inds
        info[env_name]['p_sample'] = p_sample
        info[env_name]['state_mean'] = state_mean
        info[env_name]['state_std'] = state_std
        if variant['average_state_mean']:
            info[env_name]['state_mean'] = variant['total_state_mean']
            info[env_name]['state_std'] = variant['total_state_std']
    return info


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

""" evaluation """

def eval_episodes(target_rew, info, variant, env, env_name):
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_eval_episodes = variant['num_eval_episodes']
    mode = variant.get('mode', 'normal')
    # print(max_ep_len, state_mean, state_std, scale)
    # print(state_dim, act_dim, device)
    # print(num_eval_episodes)
    # print(mode)
    def fn(model, prompt=None):
        returns = []
        success = []
        length = []
        norm_score=None
        num_eval_episodes_=num_eval_episodes
        if 'ant-dir' in env_name or 'cheetah-vel' in env_name:
            num_eval_episodes_=1
        for i in range(num_eval_episodes_):
            with torch.no_grad():
                if type(env)==list:
                    cur_env=env[i]
                else:
                    cur_env=env
                ret, episode_length, suc = prompt_evaluate_episode_rtg(
                    env_name,
                    cur_env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    prompt=prompt,
                    no_r=variant['no_r'],
                    no_rtg=variant['no_rtg'],
                    no_state_normalize=variant['no_state_normalize']                
                    )
            returns.append(ret)
            length.append(episode_length)
            success.append(suc)
        if '-v2' in env_name:
            norm_score=np.mean(success)*100
        elif 'cheetah-vel' in env_name:
            norm_score=np.mean(returns)
            # if norm_score>100:
            #     norm_score=100
            # if norm_score<0:
            #     norm_score=0
            if norm_score<-100:
                norm_score=0
            elif norm_score>-30:
                norm_score=100
            else:
                norm_score=(norm_score+100)*100/70
        elif 'ant-dir' in env_name:
            norm_score=np.mean(returns)/5
            if norm_score>100:
                norm_score=100
            if norm_score<0:
                norm_score=0
        else:
            norm_score=np.mean(returns)/10
        return {
            f'{env_name}_target_{target_rew}_return_mean': np.mean(returns),
            # f'{env_name}_target_{target_rew}_return_std': np.std(returns),
            # f'{env_name}_target_{target_rew}_length_mean': np.mean(length),
            f'{env_name}_target_{target_rew}_normalized_score_mean': norm_score,
            }
    return fn

def _to_str(num):
    if num >= 1e6:
        return f'{(num/1e6):.2f} M'
    else:
        return f'{(num/1e3):.2f} k'
    
def param_to_module(param):
    module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
    return module_name  

def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters()}
    n_parameters = sum(counts.values())
    print(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    max_length = max([len(k) for k in sorted_keys])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(' '*8 + f'{key:10}: {_to_str(count)} | {modules[module]}')

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(' '*8 + f'... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters')
    return n_parameters

def report_training_parameters(model, topk=10):
    counts={}
    for k,p in model.named_parameters():
        if p.requires_grad==True:
            counts[k] = p.numel()
    n_parameters = sum(counts.values())
    print(f'[ utils/arrays ] Total Training parameters: {_to_str(n_parameters)}')

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    max_length = max([len(k) for k in sorted_keys])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(' '*8 + f'{key:10}: {_to_str(count)} | {modules[module]}')

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(' '*8 + f'... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters')
    return n_parameters

def plot_gradient_cluster(reduced_parameter_gradient,task_type, train_env_name_list,cal_time,iter):
    if '160' in task_type:
        a,b,c,d=50,80,120,160
    elif '80_dm_mw' in task_type:
        a,b,c,d=50,80,120,160
    elif '80' in task_type:
        a,b,c,d=25,40,60,80
    elif '50' in task_type:
        a,b,c,d=16,26,38,50
    elif '40' in task_type:
        a,b,c,d=12,20,30,40
    elif '30' in task_type:
        a,b,c,d=9,14,23,30
    elif '20' in task_type:
        a,b,c,d=6,10,15,20
    elif '10' in task_type:
        a,b,c,d=3,5,8,10
    elif '5' in task_type:
        a,b,c,d=2,3,4,5
    plt.clf()
    plt.scatter([x[0] for x in reduced_parameter_gradient[:a]], [x[1] for x in reduced_parameter_gradient[:a]])
    plt.scatter([x[0] for x in reduced_parameter_gradient[a:b]], [x[1] for x in reduced_parameter_gradient[a:b]])
    plt.scatter([x[0] for x in reduced_parameter_gradient[b:c]], [x[1] for x in reduced_parameter_gradient[b:c]])
    plt.scatter([x[0] for x in reduced_parameter_gradient[c:d]], [x[1] for x in reduced_parameter_gradient[c:d]])
    for i in range(0,len(train_env_name_list),10):
        plt.text(reduced_parameter_gradient[i][0], reduced_parameter_gradient[i][1], train_env_name_list[i], fontsize=9, ha='right')
    plt.title(f"training/Gradient-Conflicts_Iter_{iter+1}")
    plt.savefig(f"{task_type}_fig/seed_{cal_time}/Gradient-Conflicts_Iter_{iter+1}.png")