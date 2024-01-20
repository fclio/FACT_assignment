# Runs with:
# - D3RLPy version 1.1.1
# - Gym version 0.26.2
# - ale-py version 0.7.0

import numpy as np
import gzip
import torch
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gym
import ale_py
from d3rlpy.datasets import get_dataset
from d3rlpy.algos import DiscreteSAC
from gpt import GPTConfig, GPT

# def load_sq_data(self, version='Seaquest-ram-v0'):
#     """
#     Load the seaquest dataset from the d4rl-atari repo.
#     """
#     print("Loading Seaquest Dataset")

# dataset, env = get_d4rl('ALE/Seaquest-v5') # -v{0, 1, 2, 3, 4} for datasets with the other random seeds

# env = gym.make('ALE/Breakout-v5')
dataset, env = get_dataset('seaquest-mixed-v4')

print("Datasets loaded")

datasets_names = ["observation", "action", "reward", "terminal"]
datasets = {}
for dataset_name in datasets_names:
    with gzip.open("data/"+dataset_name+".gz", 'rb') as f:
        datasets[dataset_name] = np.load(f, allow_pickle=False)

sub_traj_len = 30
num_trajs = 60
sub_trajs = []
current_traj = []
num_sar = 0
total_trajs = 0
for obs, act, rew, ter in zip(datasets["observation"], datasets["action"], datasets["reward"], datasets["terminal"]):
    current_traj.append((obs, act, rew))
    num_sar += 1
    
    if num_sar == sub_traj_len:
        sub_trajs.append(current_traj)
        num_sar = 0
        current_traj = []
    elif ter:
        total_trajs += 1
        while num_sar < sub_traj_len:
            current_traj.append((0,0,0))
            num_sar += 1
        sub_trajs.append(current_traj)
        num_sar = 0
        current_traj = []
    if total_trajs == num_trajs:
        break
    
num_sub_trajs = len(sub_trajs)

observation_traj = torch.zeros((num_sub_trajs, sub_traj_len, 4, 84, 84), dtype=torch.int8)
action_traj = torch.zeros((num_sub_trajs, sub_traj_len, 1), dtype=torch.int8)
reward_traj = torch.zeros((num_sub_trajs, sub_traj_len, 1), dtype=torch.int8)
len_traj = torch.zeros((num_sub_trajs, 1), dtype=torch.int64)

for idx1, traj in enumerate(sub_trajs):
    len_traj[idx1] = len(traj)
    for idx2, (obs, act, rew) in enumerate(traj):
        observation_traj[idx1, idx2] = torch.tensor(obs)
        action_traj[idx1, idx2] = torch.tensor(act)
        reward_traj[idx1, idx2] = torch.tensor(rew)
        
# vocab_size = 18
# block_size = 90
# model_type = "reward_conditioned"
# timesteps = 2719

# mconf = GPTConfig(
#     vocab_size,
#     block_size,
#     n_layer=6,
#     n_head=8,
#     n_embd=128,
#     model_type=model_type,
#     max_timestep=timesteps,
# )
# model = GPT(mconf)

# model.load_pretrained("checkpoints/Seaquest_123.pth", cpu=True)

# batch_size = 30
# sub_traj_embs = np.empty((num_sub_trajs, 128))
# for idx in range(int(np.ceil(num_sub_trajs/batch_size))):
#     obs = observation_traj[batch_size*idx:batch_size*(idx+1)]
#     act = action_traj[batch_size*idx:batch_size*(idx+1)]
#     rew = reward_traj[batch_size*idx:batch_size*(idx+1)]
#     _, _, emb = model(obs, act, rtgs=rew, timesteps=torch.tensor([[[timesteps]]]))
#     emb = emb.mean(dim=1)
#     sub_traj_embs[batch_size*idx:batch_size*(idx+1)] = emb.detach().numpy()

# pca = PCA(n_components=2)
# pca_traj_embeddings = pca.fit_transform(sub_traj_embs)
# plotting_data = {'feature 1': pca_traj_embeddings[:, 0], 'feature 2': pca_traj_embeddings[:, 1]}
                 
# df = pd.DataFrame(plotting_data)

# plt.figure()
# sns.scatterplot(x='feature 1', y='feature 2', data=df)
# # plt.title('PCA: Trajectory Embeddings')
# # plt.legend()
# # plt.show()

# disc_sac= DiscreteSACConfig().create(device=None)
# disc_sac.inner_create_impl((4, 84, 84), 18)
# print(disc_sac._impl)
disc_sac = DiscreteSAC(
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    temp_learning_rate=3e-4,
    batch_size=256,
    n_steps=100000)




disc_sac.fit(dataset=dataset, n_epochs=10)
actions = disc_sac.predict(observation_traj[0])