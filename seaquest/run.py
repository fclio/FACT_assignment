# Runs with:
# - D3RLPy version 1.1.1
# - Gym version 0.26.2
# - ale-py version 0.7.0

import numpy as np
# import gzip
import torch

# import d3rlpy


from sklearn.decomposition import PCA
from gpt import GPTConfig, GPT
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans
# from d3rlpy.algos import DiscreteSAC
from d3rlpy.datasets import MDPDataset
from d3rlpy.datasets import get_dataset


dataset, env = get_dataset('seaquest-mixed-v4')

    
# sub_traj_len = 30
# num_trajs = 717
# sub_trajs = []
# current_traj = []
# num_sar = 0
# total_trajs = 0
# for obs, act, rew, ter in zip(dataset.observations, dataset.actions, dataset.rewards, dataset.terminals):
#     current_traj.append((obs, act, rew))
#     num_sar += 1
    
#     if num_sar == sub_traj_len:
#         sub_trajs.append(current_traj)
#         num_sar = 0
#         current_traj = []
#     elif ter:
#         total_trajs += 1
#         while num_sar < sub_traj_len:
#             current_traj.append((0,0,0))
#             num_sar += 1
#         sub_trajs.append(current_traj)
#         num_sar = 0
#         current_traj = []
#     if total_trajs == num_trajs:
#         break
    
# num_sub_trajs = len(sub_trajs)


# observation_traj = torch.zeros((num_sub_trajs, sub_traj_len, 4, 84, 84), dtype=torch.int8)
# action_traj = torch.zeros((num_sub_trajs, sub_traj_len, 1), dtype=torch.int8)
# reward_traj = torch.zeros((num_sub_trajs, sub_traj_len, 1), dtype=torch.int8)
# len_traj = torch.zeros((num_sub_trajs, 1), dtype=torch.int64)

# for idx1, traj in enumerate(sub_trajs):
#     len_traj[idx1] = len(traj)
#     for idx2, (obs, act, rew) in enumerate(traj):
#         observation_traj[idx1, idx2] = torch.tensor(obs)
#         action_traj[idx1, idx2] = torch.tensor(act)
#         reward_traj[idx1, idx2] = torch.tensor(rew)
        
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

# # save the embeddings
# np.save("seaquest/data/sub_traj_embs.npy", sub_traj_embs)
sub_traj_embs = np.load("seaquest/data/sub_traj_embs.npy")

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

cluster_emb = [sub_traj_embs[traj_cluster_labels == i] for i in range(len(clusters))]

plot = False
if plot == True:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    palette = sns.color_palette('husl', len(clusters) + 1)
    pca_traj = PCA(n_components=2)
    pca_traj_embeds = pca_traj.fit_transform(sub_traj_embs)

    plotting_data = {'feature 1': pca_traj_embeds[emb_ids, 0],
                    'feature 2': pca_traj_embeds[emb_ids, 1],
                    'cluster id': traj_cluster_labels[emb_ids]}
    df = pd.DataFrame(plotting_data)

    plt.figure(figsize=(4,3))
    data_ax = sns.scatterplot(x='feature 1',
                            y='feature 2',
                            hue='cluster id',
                            palette=palette[:len(clusters)],
                            data=df,
                            legend=True)
    plt.legend(title = '$c_{j}$', loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=5)
    # plt.legend(title = '$c_{j}$', loc='center left', bbox_to_anchor=(1., 0.7), ncol=2)
    # for cid, _ in enumerate(cluster_data_embeds):
    #     data_ax.text(pca_data_embeds[:, 0][cid],
    #                  pca_data_embeds[:, 1][cid],
    #                  str(cid),
    #                  horizontalalignment='left',
    #                  size='medium',
    #                  color='black',
    #                  weight='semibold')
    plt.tight_layout()
    # plt.savefig('./traj_clustering_grid.pdf')
    plt.show()



# cluster embeddings
def get_data_embedding(traj_embeddings):
    return np.exp(np.array(traj_embeddings).sum(axis=0)/10.)/np.sum(np.exp(np.array(traj_embeddings).sum(axis=0)/10.))

data_embeddings = []
for idx, cluster in enumerate(cluster_emb):
    data_embeddings.append(get_data_embedding(cluster))

from d3rlpy.datasets import MDPDataset
from d3rlpy.algos import DiscreteSAC



observations = dataset.observations[:len(sub_traj_embs)*30]
actions = dataset.actions[:len(sub_traj_embs)*30]
rewards = dataset.rewards[:len(sub_traj_embs)*30]
terminals = dataset.terminals[:len(sub_traj_embs)*30]
dataset = MDPDataset(observations, actions, rewards, terminals)

traj_cluster_labels_np = np.repeat(np.array(traj_cluster_labels), 30)

cluster_datasets = []
for idx in range(len(clusters)):
    observations = dataset.observations[traj_cluster_labels_np == idx]
    actions = dataset.actions[traj_cluster_labels_np == idx]
    rewards = dataset.rewards[traj_cluster_labels_np == idx]
    terminals = dataset.terminals[traj_cluster_labels_np == idx]
    cluster_datasets.append(MDPDataset(observations, actions, rewards, terminals))

agents = []
data_embeddings = [] 
explanation_predictions = []
for idx, cluster_dataset in enumerate(cluster_datasets):
    disc_sac = DiscreteSAC(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        batch_size=256,
        n_steps=100000, 
        use_gpu=True)
    # disc_sac.fit(cluster_dataset, n_steps=10000)
    # disc_sac.save_model("seaquest/data/agent_c{}.pt".format(idx))
    data_embeddings.append(get_data_embedding(cluster_dataset))
    disc_sac = DiscreteSAC.load_model("seaquest/data/agent_c{}.pt".format(idx))
    agents.append(disc_sac)
    prediction = []
    for observation in dataset.observations:
        prediction.append(disc_sac.predict(observation))
    explanation_predictions.append(prediction)



from scipy.stats import wasserstein_distance

action_dict = {}

attributions = []

original_policy = DiscreteSAC(
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    temp_learning_rate=3e-4,
    batch_size=256,
    n_steps=100000, 
    use_gpu=True)
origin_policy = original_policy.load_model("seaquest/data/agent_c0.pt")

# action dict seaquest
action_dict = {
    0: 'NOOP',
    1: 'FIRE',
    2: 'UP',
    3: 'RIGHT',
    4: 'LEFT',
    5: 'DOWN',
    6: 'UPRIGHT',
    7: 'UPLEFT',
    8: 'DOWNRIGHT',
    9: 'DOWNLEFT',
    10: 'UPFIRE',
    11: 'RIGHTFIRE',
    12: 'LEFTFIRE',
    13: 'DOWNFIRE',
    14: 'UPRIGHTFIRE',
    15: 'UPLEFTFIRE',
    16: 'DOWNRIGHTFIRE',
    17: 'DOWNLEFTFIRE'
}
original_prediction = original_policy.predict(dataset.observations)
action_distance = np.ones((18, 18))
    
for idx, (observation, action, reward, terminal)  in enumerate(zip(dataset.observations, dataset.actions, dataset.rewards, dataset.terminals)):
    if idx > 100:
        break
    if terminal:
        continue
    
    original_action = original_prediction[idx]
    agent_predictions = []
    for new_prediction in explanation_predictions:
        agent_predictions.append(new_prediction[idx])
    

    cluster_distance = []
    alternative_actions = []
    for agent_pred in agent_predictions[agent_predictions != original_action]:
        cluster_distance.append(wasserstein_distance(original_action, agent_pred))
    
        alternative_actions.append(agent_pred)
    
    responsible_cluster_id = np.argsort(cluster_distance)[0]
    responsible_action = agent_predictions[responsible_cluster_id]
        
    print('-'*10)
    print(f'State - {idx}')
    print(f'Distance - {cluster_distance[responsible_cluster_id]}')
    print(f'Original Actions -{action_dict[original_action]}')
    print(f'New Action - {action_dict[responsible_action]}')

    print(f'Responsible data combination - data id {responsible_cluster_id}')
    print(f'Responsible trajectory id {clusters[responsible_cluster_id - 1]}')
    if len(clusters[responsible_cluster_id - 1]):
        cid_list = list(range(len(clusters)))
        cid_list.pop(responsible_cluster_id - 1)
        alternate_cid = np.random.choice(cid_list)
        attributions.append({
            'state' : idx,
            'orig_act': action_dict[original_action],
            'new_act': action_dict[responsible_action],
            # 'attributed_trajs':clusters[responsible_cluster_id - 1],
            # 'random_baseline_trajs': list(np.random.randint(0, len(sub_traj_embs), 5)),
            # 'alternate_cluster_trajs': clusters[alternate_cid - 1],
            'responsible_cluster': responsible_cluster_id
        })
#         for traj in clusters[responsible_data_combination - 1]:
#             env.plot_traj(offline_data[traj])
    print('-'*10)
        
