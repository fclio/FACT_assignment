import numpy as np
import gzip
import torch
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gpt import GPTConfig, GPT
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans
# from d3rlpy.algos import DiscreteSAC, DiscreteSACConfig


# def load_sq_data(self, version='Seaquest-ram-v0'):
#     """
#     Load the seaquest dataset from the d4rl-atari repo.
#     """
#     print("Loading Seaquest Dataset")

#     env = gym.make(version) 
#     dataset = env.get_dataset()
#     return dataset

datasets_names = ["observation", "action", "reward", "terminal"]
datasets = {}
for dataset_name in datasets_names:
    with gzip.open("seaquest/data/"+dataset_name+".gz", 'rb') as f:
        datasets[dataset_name] = np.load(f, allow_pickle=False)

sub_traj_len = 30
num_trajs = 717
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
        
vocab_size = 18
block_size = 90
model_type = "reward_conditioned"
timesteps = 2719

mconf = GPTConfig(
    vocab_size,
    block_size,
    n_layer=6,
    n_head=8,
    n_embd=128,
    model_type=model_type,
    max_timestep=timesteps,
)
model = GPT(mconf)

model.load_pretrained("checkpoints/Seaquest_123.pth", cpu=True)

batch_size = 30
sub_traj_embs = np.empty((num_sub_trajs, 128))
for idx in range(int(np.ceil(num_sub_trajs/batch_size))):
    obs = observation_traj[batch_size*idx:batch_size*(idx+1)]
    act = action_traj[batch_size*idx:batch_size*(idx+1)]
    rew = reward_traj[batch_size*idx:batch_size*(idx+1)]
    _, _, emb = model(obs, act, rtgs=rew, timesteps=torch.tensor([[[timesteps]]]))
    emb = emb.mean(dim=1)
    sub_traj_embs[batch_size*idx:batch_size*(idx+1)] = emb.detach().numpy()



# Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
# start analysis.
amount_initial_centers = 2
initial_centers = kmeans_plusplus_initializer(sub_traj_embs, amount_initial_centers).initialize()
 
# Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
# number of clusters that can be allocated is 10.
xmeans_instance = xmeans(sub_traj_embs, initial_centers, 8)
xmeans_instance.process()
 
# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()


points_per_cluster = 50
trajs_in_cluster = [0]*len(clusters)
traj_cluster_labels = np.zeros(len(sub_traj_embs), dtype=int)
emb_ids = []
for cluster_id, cluster in enumerate(clusters):
    for traj_id in cluster:
        traj_cluster_labels[traj_id] = cluster_id
        if trajs_in_cluster[cluster_id] < points_per_cluster:
            trajs_in_cluster[cluster_id] += 1
            emb_ids.append(traj_id)
palette = sns.color_palette('husl', len(clusters) + 1)

pca_traj = PCA(n_components=2)
pca_traj_embeds = pca_traj.fit_transform(sub_traj_embs)

# plotting_data = {'feature 1': pca_traj_embeds[emb_ids, 0],
#                  'feature 2': pca_traj_embeds[emb_ids, 1],
#                  'cluster id': traj_cluster_labels[emb_ids]}
# df = pd.DataFrame(plotting_data)

# plt.figure(figsize=(4,3))
# data_ax = sns.scatterplot(x='feature 1',
#                           y='feature 2',
#                           hue='cluster id',
#                           palette=palette[:len(clusters)],
#                           data=df,
#                           legend=True)
# plt.legend(title = '$c_{j}$', loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=5)
# # plt.legend(title = '$c_{j}$', loc='center left', bbox_to_anchor=(1., 0.7), ncol=2)
# # for cid, _ in enumerate(cluster_data_embeds):
# #     data_ax.text(pca_data_embeds[:, 0][cid],
# #                  pca_data_embeds[:, 1][cid],
# #                  str(cid),
# #                  horizontalalignment='left',
# #                  size='medium',
# #                  color='black',
# #                  weight='semibold')
# plt.tight_layout()
# # plt.savefig('./traj_clustering_grid.pdf')
# plt.show()


# disc_sac_config = DiscreteSACConfig()
# disc_sac = DiscreteSAC(
#     config=disc_sac_config,
#     device="cpu"
# )

# disc_sac.fit(sub_traj_embs, n_steps=10000)
# actions = disc_sac.predict(observation_traj[0])