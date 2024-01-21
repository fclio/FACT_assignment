# Runs with:
# - d3rlpy == 1.1.1
# - d4rl-atari==0.1
# - d4rl == 1.1
# - numpy == 1.23.1 / 1.21.5
# - python=3.8.18=h4de0772_1_cpython / 3.7.9
# - gym == 0.26.2
# - ale-py == 0.8.1
# - atari-py==0.2.6
# export LD_LIBRARY_PATH=/home/bart/miniconda3/envs/decision-transformer-atari/lib

import numpy as np
# import gzip
import torch
import time 

from sklearn.decomposition import PCA
from gpt import GPTConfig, GPT
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans
from d3rlpy.datasets import MDPDataset
from d3rlpy.datasets import get_dataset
from d3rlpy.datasets import MDPDataset
from d3rlpy.algos import DiscreteSAC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

ACTION_DICT = {
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

def create_trajectories_from_dataset(dataset, sub_traj_len=30):
    """
    Split up the dataset into sub trajectories of length sub_traj_len.
    If a terminal is encountered before the sub_traj_len is reached, 
    the sub trajectory is padded with zeros.
    """ 
    sub_traj_len = 30
    num_trajs = 717
    sub_trajs = []
    current_traj = []
    num_sar = 0
    total_trajs = 0
    for obs, act, rew, ter in zip(dataset.observations, dataset.actions, dataset.rewards, dataset.terminals):
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
            
    return observation_traj, action_traj, reward_traj, num_sub_trajs

def get_decision_transformer(vocab_size=18, block_size=90, model_type="reward_conditioned", timesteps=2719):
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
    
    return model

def encode_trajectories(observation_traj, action_traj, reward_traj, num_sub_trajs, batch_size=30):
    vocab_size = 18
    block_size = 90
    model_type = "reward_conditioned"  
    timesteps = 2719
    
    model = get_decision_transformer(vocab_size, block_size, model_type, timesteps)
    
    sub_traj_embs = np.empty((num_sub_trajs, 128))
    for idx in range(int(np.ceil(num_sub_trajs/batch_size))):
        obs = observation_traj[batch_size*idx:batch_size*(idx+1)]
        act = action_traj[batch_size*idx:batch_size*(idx+1)]
        rew = reward_traj[batch_size*idx:batch_size*(idx+1)]
        _, _, emb = model(obs, act, rtgs=rew, timesteps=torch.tensor([[[timesteps]]]))
        emb = emb.mean(dim=1)
        sub_traj_embs[batch_size*idx:batch_size*(idx+1)] = emb.detach().numpy()
    return sub_traj_embs

def plot_clusters(cluster_data_embeds, sub_traj_embs, traj_cluster_labels, clusters, emb_ids):
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
    plt.legend(title = '$c_{j}$', loc='center left', bbox_to_anchor=(1., 0.7), ncol=2)
    # for cid, _ in enumerate(cluster_data_embeds):
    #     data_ax.text(pca_traj_embeds[:, 0][cid],
    #                  pca_traj_embeds[:, 1][cid],
    #                  str(cid),
    #                  horizontalalignment='left',
    #                  size='medium',
    #                  color='black',
    #                  weight='semibold')
    plt.tight_layout()
    plt.savefig('./traj_clustering_grid.pdf')
    plt.show()

def cluster_trajectories(sub_traj_embs, plot=False, num_clusters=8):
    # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # start analysis.
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(sub_traj_embs, amount_initial_centers).initialize()
    
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    # number of clusters that can be allocated is 10.
    xmeans_instance = xmeans(sub_traj_embs, initial_centers, num_clusters)
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

    cluster_traj_embeddings = [sub_traj_embs[traj_cluster_labels == i] for i in range(len(clusters))]
    
    if plot:
        plot_clusters(sub_traj_embs, traj_cluster_labels, clusters, emb_ids)
            
    return clusters, cluster_traj_embeddings, traj_cluster_labels

# cluster embeddings
def get_data_embedding(traj_embeddings):
    return np.exp(np.array(traj_embeddings).sum(axis=0)/10.)/np.sum(np.exp(np.array(traj_embeddings).sum(axis=0)/10.))

def compute_dataset_embeddings(cluster_traj_embeddings):
    cluster_embeddings = []
    for cluster in cluster_traj_embeddings:
        cluster_embeddings.append(get_data_embedding(cluster))  

    compl_dataset_embeddings = []
    for idx, cluster in enumerate(cluster_embeddings):
        compl_dataset = [item for i, item in enumerate(cluster_embeddings) if i != idx]
        compl_dataset_embeddings.append(get_data_embedding(compl_dataset))

    original_data_embedding = get_data_embedding(cluster_embeddings)
    
    return original_data_embedding, compl_dataset_embeddings

def create_complementary_dataset(dataset, sub_traj_embs, traj_cluster_labels, clusters):
    observations_np = dataset.observations[:len(sub_traj_embs)*30]
    actions_np = dataset.actions[:len(sub_traj_embs)*30]
    rewards_np = dataset.rewards[:len(sub_traj_embs)*30]
    terminals_np = dataset.terminals[:len(sub_traj_embs)*30]

    dataset = MDPDataset(observations_np, actions_np, rewards_np, terminals_np)

    cluster_mask = np.repeat(np.array(traj_cluster_labels), 30)

    cluster_datasets = []
    for idx in range(len(clusters)):
        observations = dataset.observations[cluster_mask == idx]
        actions = dataset.actions[cluster_mask == idx]
        rewards = dataset.rewards[cluster_mask == idx]
        terminals = dataset.terminals[cluster_mask == idx]
        cluster_datasets.append(MDPDataset(observations, actions, rewards, terminals))
    
    return cluster_datasets

def compute_explanation_policies(dataset, cluster_datasets, env=None, load_model=False):
    agents = []
    for idx, cluster_dataset in enumerate(cluster_datasets):
        discrete_sac = DiscreteSAC(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            batch_size=256,
            n_steps=100000, 
            use_gpu=False)
        
        if load_model and env is not None:
            discrete_sac.build_with_env(env)
            discrete_sac.load_model(fname="checkpoints/agent_c{}.pt".format(idx))
        else:
            discrete_sac.fit(cluster_dataset, n_steps=10000)
            discrete_sac.save_model("seaquest/data/agent_c{}.pt".format(idx))
            
        agents.append(discrete_sac)
        
        # Make Predictions
        explanation_predictions = []
        predictions = []
        for observation in dataset.observations[:1000]:
            predictions.append(discrete_sac.predict([observation])[0])
        explanation_predictions.append(predictions)
        
    return agents, explanation_predictions 

def compute_original_policy(dataset, env=None, load_model=False):
    original_policy = DiscreteSAC(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        batch_size=256,
        n_steps=100000, 
        use_gpu=False)
    if load_model and env is not None:
        original_policy.build_with_env(env) 
        original_policy.load_model(fname="checkpoints/agent.pt")
    else:
        original_policy.fit(dataset, n_steps=10000)
        original_policy.save_model("seaquest/data/agent.pt")
        
    # Make predictions
    original_predictions = []
    for observation in dataset.observations[:1000]:
        original_predictions.append([original_policy.predict([observation])][0])
    
    return original_policy, original_predictions

def generate_attributions(dataset, original_predictions, explanation_predictions, original_data_embedding, compl_dataset_embeddings, clusters):
    attributions = []
    
    for idx, (observation, action, reward, terminal)  in enumerate(zip(dataset.observations, dataset.actions, dataset.rewards, dataset.terminals)):
        # To keep it short, can extend 
        if idx > 100:  
            break
        if terminal: 
            continue
        
        original_action = original_predictions[idx][0]
        agent_predictions = []
        for predictions in explanation_predictions:
            agent_predictions.append(predictions[idx])
        
        cluster_distance = []
        alternative_actions = []
        for cluster_idx in np.where(np.array(agent_predictions) != original_action)[0]:
            cluster_distance.append(wasserstein_distance(original_data_embedding, compl_dataset_embeddings[cluster_idx]))
        
            alternative_actions.append(cluster_idx)
        
        responsible_cluster_id = np.argsort(cluster_distance)[0]
        responsible_action = agent_predictions[responsible_cluster_id]
            
        print('-'*10)
        print(f'State - {idx}')
        print(f'Distance - {cluster_distance[responsible_cluster_id]}')
        print(f'Original Actions -{ACTION_DICT[original_action]}')
        print(f'New Action - {ACTION_DICT[responsible_action]}')

        print(f'Responsible data combination - data id {responsible_cluster_id}')
        print(f'Responsible trajectory id {clusters[responsible_cluster_id - 1]}')
        if len(clusters[responsible_cluster_id - 1]):
            cid_list = list(range(len(clusters)))
            cid_list.pop(responsible_cluster_id - 1)
            # alternate_cid = np.random.choice(cid_list)
            attributions.append({
                'state' : idx,
                'orig_act': ACTION_DICT[original_action],
                'new_act': ACTION_DICT[responsible_action],
                # 'attributed_trajs':clusters[responsible_cluster_id - 1],
                # 'random_baseline_trajs': list(np.random.randint(0, len(sub_traj_embs), 5)),
                # 'alternate_cluster_trajs': clusters[alternate_cid - 1],
                'responsible_cluster': responsible_cluster_id
            })
    #         for traj in clusters[responsible_data_combination - 1]:
    #             env.plot_traj(offline_data[traj])
        print('-'*10)
    return attributions
    

def run_trajectory_attribution(load_emb = False, load_model=False, save_attributions=True):
    # Load dataset and environment
    start = time.time()
    dataset, env = get_dataset('seaquest-mixed-v4')
    print(f"Dataset loaded in {time.time() - start} seconds")
    
    if not load_emb:
        # Preprocess dataset into sub trajectories
        observation_traj, action_traj, reward_traj, num_sub_trajs = create_trajectories_from_dataset(dataset, sub_traj_len=30)
        
        # Encode sub trajectories using decision transformer
        sub_traj_embs = encode_trajectories(observation_traj, action_traj, reward_traj, num_sub_trajs, batch_size=30)
    else:
        # np.save("seaquest/data/sub_traj_embs.npy", sub_traj_embs)
        sub_traj_embs = np.load("data/sub_traj_embs.npy")
    
    # Cluster trajectories
    clusters, cluster_traj_embeddings, traj_cluster_labels = cluster_trajectories(sub_traj_embs, plot=False, num_clusters=8)
    
    # Compute original dataset embedding and complementary dataset embeddings
    original_data_embedding, compl_dataset_embeddings = compute_dataset_embeddings(cluster_traj_embeddings)
    
    # Create complementary datasets
    cluster_datasets = create_complementary_dataset(dataset, sub_traj_embs, traj_cluster_labels, clusters)
   
   # Fit explanation policies
    explanation_policies, explanation_predictions = compute_explanation_policies(cluster_datasets, env=env, load_model=False)
   
   # Fit original policy
    original_policy, original_predictions = compute_original_policy(dataset, env=env, load_model=False)
   
    # Generate attributions
    attributions = generate_attributions(dataset, original_predictions, explanation_predictions, original_data_embedding, compl_dataset_embeddings, clusters)
    if save_attributions:
        np.save("data/attributions.npy", attributions)
    
if __name__ == "__main__":
    
    run_trajectory_attribution(load_emb = True, load_model=True, save_attributions=True)

    # start = time.time()
    # dataset, env = get_dataset('seaquest-mixed-v4')
    # print(f"Dataset loaded in {time.time() - start} seconds")
        
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
    # sub_traj_embs = np.load("data/sub_traj_embs.npy")

    # # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # # start analysis.
    # amount_initial_centers = 2
    # initial_centers = kmeans_plusplus_initializer(sub_traj_embs, amount_initial_centers).initialize()
    
    # # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    # # number of clusters that can be allocated is 10.
    # xmeans_instance = xmeans(sub_traj_embs, initial_centers, 8)
    # xmeans_instance.process()
    
    # # Extract clustering results: clusters and their centers
    # clusters = xmeans_instance.get_clusters()
    # centers = xmeans_instance.get_centers()

    # points_per_cluster = 50
    # trajs_in_cluster = [0]*len(clusters)
    # traj_cluster_labels = np.zeros(len(sub_traj_embs), dtype=int)
    # emb_ids = []
    # for cluster_id, cluster in enumerate(clusters):
    #     for traj_id in cluster:
    #         traj_cluster_labels[traj_id] = cluster_id
    #         if trajs_in_cluster[cluster_id] < points_per_cluster:
    #             trajs_in_cluster[cluster_id] += 1
    #             emb_ids.append(traj_id)

    # cluster_traj_embeddings = [sub_traj_embs[traj_cluster_labels == i] for i in range(len(clusters))]


    # plot = False
    # if plot == True:
    #     import pandas as pd
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     palette = sns.color_palette('husl', len(clusters) + 1)
    #     pca_traj = PCA(n_components=2)
    #     pca_traj_embeds = pca_traj.fit_transform(sub_traj_embs)

    #     plotting_data = {'feature 1': pca_traj_embeds[emb_ids, 0],
    #                     'feature 2': pca_traj_embeds[emb_ids, 1],
    #                     'cluster id': traj_cluster_labels[emb_ids]}
    #     df = pd.DataFrame(plotting_data)

    #     plt.figure(figsize=(4,3))
    #     data_ax = sns.scatterplot(x='feature 1',
    #                             y='feature 2',
    #                             hue='cluster id',
    #                             palette=palette[:len(clusters)],
    #                             data=df,
    #                             legend=True)
    #     plt.legend(title = '$c_{j}$', loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=5)
    #     # plt.legend(title = '$c_{j}$', loc='center left', bbox_to_anchor=(1., 0.7), ncol=2)
    #     # for cid, _ in enumerate(cluster_data_embeds):
    #     #     data_ax.text(pca_data_embeds[:, 0][cid],
    #     #                  pca_data_embeds[:, 1][cid],
    #     #                  str(cid),
    #     #                  horizontalalignment='left',
    #     #                  size='medium',
    #     #                  color='black',
    #     #                  weight='semibold')
    #     plt.tight_layout()
    #     # plt.savefig('./traj_clustering_grid.pdf')
    #     plt.show()



    # # cluster embeddings
    # def get_data_embedding(traj_embeddings):
    #     return np.exp(np.array(traj_embeddings).sum(axis=0)/10.)/np.sum(np.exp(np.array(traj_embeddings).sum(axis=0)/10.))


    # cluster_embeddings = []
    # for cluster_idx, cluster in enumerate(cluster_traj_embeddings):
    #     cluster_embeddings.append(get_data_embedding(cluster))  

    # compl_dataset_embeddings = []
    # for idx, cluster in enumerate(cluster_embeddings):
    #     compl_dataset = [item for i, item in enumerate(cluster_embeddings) if i != idx]
    #     compl_dataset_embeddings.append(get_data_embedding(compl_dataset))

    # original_data_embedding = get_data_embedding(cluster_embeddings)





    # observations_np = dataset.observations[:len(sub_traj_embs)*30]
    # actions_np = dataset.actions[:len(sub_traj_embs)*30]
    # rewards_np = dataset.rewards[:len(sub_traj_embs)*30]
    # terminals_np = dataset.terminals[:len(sub_traj_embs)*30]

    # # np.save("data/observations.npy", observations_np)
    # # np.save("data/actions.npy", actions_np)
    # # np.save("data/rewards.npy", rewards_np)
    # # np.save("data/terminals.npy", terminals_np)

    # # observations_np = np.load("data/observations.npy")
    # # actions_np = np.load("data/actions.npy")
    # # rewards_np = np.load("data/rewards.npy")
    # # terminals_np = np.load("data/terminals.npy")
    # debug = False
    # if debug == False:
    #     dataset = MDPDataset(observations_np, actions_np, rewards_np, terminals_np)

    #     traj_cluster_labels_np = np.repeat(np.array(traj_cluster_labels), 30)

    #     cluster_datasets = []
    #     for idx in range(len(clusters)):
    #         observations = dataset.observations[traj_cluster_labels_np == idx]
    #         actions = dataset.actions[traj_cluster_labels_np == idx]
    #         rewards = dataset.rewards[traj_cluster_labels_np == idx]
    #         terminals = dataset.terminals[traj_cluster_labels_np == idx]
    #         cluster_datasets.append(MDPDataset(observations, actions, rewards, terminals))

    #     agents = []
    #     # data_embeddings = [] 
    #     explanation_predictions = []
    #     for idx, cluster_dataset in enumerate(cluster_datasets):
    #         disc_sac = DiscreteSAC(
    #             actor_learning_rate=3e-4,
    #             critic_learning_rate=3e-4,
    #             temp_learning_rate=3e-4,
    #             batch_size=256,
    #             n_steps=100000, 
    #             use_gpu=False)
    #         # disc_sac.fit(cluster_dataset, n_steps=10000)
    #         # disc_sac.save_model("seaquest/data/agent_c{}.pt".format(idx))
    #         disc_sac.build_with_env(env)
    #         disc_sac.load_model(fname="checkpoints/agent_c{}.pt".format(idx))
    #         agents.append(disc_sac)
    #         # data_embeddings.append(get_data_embedding(cluster_embeddings[idx]))

    #         prediction = []
    #         for observation in dataset.observations[:1000]:
    #             prediction.append(disc_sac.predict([observation])[0])
    #         explanation_predictions.append(prediction)



    #     from scipy.stats import wasserstein_distance

    #     attributions = []

    #     original_policy = DiscreteSAC(
    #         actor_learning_rate=3e-4,
    #         critic_learning_rate=3e-4,
    #         temp_learning_rate=3e-4,
    #         batch_size=256,
    #         n_steps=100000, 
    #         use_gpu=False)
    #     original_policy.build_with_env(env) 
    #     original_policy.load_model(fname="checkpoints/agent.pt")
    #     original_predictions = []
    #     for observation in dataset.observations[:1000]:
    #         original_predictions.append([original_policy.predict([observation])][0])
    #     # original_prediction = original_policy.predict(dataset.observations)
    #     # action dict seaquest
    #     action_dict = {
    #         0: 'NOOP',
    #         1: 'FIRE',
    #         2: 'UP',
    #         3: 'RIGHT',
    #         4: 'LEFT',
    #         5: 'DOWN',
    #         6: 'UPRIGHT',
    #         7: 'UPLEFT',
    #         8: 'DOWNRIGHT',
    #         9: 'DOWNLEFT',
    #         10: 'UPFIRE',
    #         11: 'RIGHTFIRE',
    #         12: 'LEFTFIRE',
    #         13: 'DOWNFIRE',
    #         14: 'UPRIGHTFIRE',
    #         15: 'UPLEFTFIRE',
    #         16: 'DOWNRIGHTFIRE',
    #         17: 'DOWNLEFTFIRE'
    #     }
    #     # action_distance = np.ones((18, 18))
            
    #     for idx, (observation, action, reward, terminal)  in enumerate(zip(dataset.observations, dataset.actions, dataset.rewards, dataset.terminals)):
    #         if idx > 100:
    #             break
    #         if terminal:
    #             continue
            
    #         original_action = original_predictions[idx][0]
    #         agent_predictions = []
    #         print("explanation_predictions", len(explanation_predictions))
    #         for predictions in explanation_predictions:
    #             print(predictions[idx])
    #             agent_predictions.append(predictions[idx])
    #         print("agent_predictions", agent_predictions)
    #         print("original_action", original_action)
            
    #         cluster_distance = []
    #         alternative_actions = []
    #         for cluster_idx in np.where(np.array(agent_predictions) != original_action)[0]:
    #             cluster_distance.append(wasserstein_distance(original_data_embedding, compl_dataset_embeddings[cluster_idx]))
            
    #             alternative_actions.append(cluster_idx)
            
    #         responsible_cluster_id = np.argsort(cluster_distance)[0]
    #         responsible_action = agent_predictions[responsible_cluster_id]
                
    #         print('-'*10)
    #         print(f'State - {idx}')
    #         print(f'Distance - {cluster_distance[responsible_cluster_id]}')
    #         print(f'Original Actions -{action_dict[original_action]}')
    #         print(f'New Action - {action_dict[responsible_action]}')

    #         print(f'Responsible data combination - data id {responsible_cluster_id}')
    #         print(f'Responsible trajectory id {clusters[responsible_cluster_id - 1]}')
    #         if len(clusters[responsible_cluster_id - 1]):
    #             cid_list = list(range(len(clusters)))
    #             cid_list.pop(responsible_cluster_id - 1)
    #             alternate_cid = np.random.choice(cid_list)
    #             attributions.append({
    #                 'state' : idx,
    #                 'orig_act': action_dict[original_action],
    #                 'new_act': action_dict[responsible_action],
    #                 # 'attributed_trajs':clusters[responsible_cluster_id - 1],
    #                 # 'random_baseline_trajs': list(np.random.randint(0, len(sub_traj_embs), 5)),
    #                 # 'alternate_cluster_trajs': clusters[alternate_cid - 1],
    #                 'responsible_cluster': responsible_cluster_id
    #             })
    #     #         for traj in clusters[responsible_data_combination - 1]:
    #     #             env.plot_traj(offline_data[traj])
    #         print('-'*10)
    #     np.save("data/attributions.npy", attributions)
        

            
