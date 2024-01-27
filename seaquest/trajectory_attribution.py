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
import torch
import time 
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans
from d3rlpy.datasets import MDPDataset, get_dataset
from d3rlpy.algos import DiscreteSAC
from scipy.stats import wasserstein_distance

from metrics import get_metrics
from utils import get_data_embedding, get_decision_transformer, plot_clusters
from constants import ACTION_DICT


def create_trajectories_from_dataset(dataset, sub_traj_len=30):
    """
    Split up the dataset into sub trajectories of length sub_traj_len.
    If a terminal is encountered before the sub_traj_len is reached, 
    the sub trajectory is padded with zeros.
    
    Args:
    - dataset: MDPDataset
    - sub_traj_len: int, length of sub trajectories
    
    Returns:
    - observation_traj: torch.Tensor, shape (num_sub_trajs, sub_traj_len, 4, 84, 84)
    - action_traj: torch.Tensor, shape (num_sub_trajs, sub_traj_len, 1)
    - reward_traj: torch.Tensor, shape (num_sub_trajs, sub_traj_len, 1)
    - num_sub_trajs: int, number of sub trajectories
    """ 
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


def encode_trajectories(observation_traj, action_traj, reward_traj, num_sub_trajs, batch_size=30):
    """
    Encodes the sub trajectories using the decision transformer with batching.
    
    Args:
    - observation_traj: torch.Tensor, shape (num_sub_trajs, sub_traj_len, 4, 84, 84)
    - action_traj: torch.Tensor, shape (num_sub_trajs, sub_traj_len, 1)
    - reward_traj: torch.Tensor, shape (num_sub_trajs, sub_traj_len, 1)
    - num_sub_trajs: int, number of sub trajectories
    - batch_size: int, batch size for encoding
    
    Returns:
    - sub_traj_embs: np.array, shape (num_sub_trajs, 128)
    """
    vocab_size = 18
    block_size = 90
    model_type = "reward_conditioned"  
    timesteps = 2719
    
    model = get_decision_transformer(vocab_size, block_size, model_type, timesteps)
    
    sub_traj_embs = np.empty((num_sub_trajs, 128))
    for idx in range(int(np.ceil(num_sub_trajs/batch_size))):
        # Create (state, action, reward) tuples for the batch
        obs = observation_traj[batch_size*idx:batch_size*(idx+1)]
        act = action_traj[batch_size*idx:batch_size*(idx+1)]
        rew = reward_traj[batch_size*idx:batch_size*(idx+1)]
        
        # We return the embeddings from the forward call
        # before it is passed through the decoder head
        _, _, emb = model(obs, act, rtgs=rew, timesteps=torch.tensor([[[timesteps]]]))
        emb = emb.mean(dim=1)
        sub_traj_embs[batch_size*idx:batch_size*(idx+1)] = emb.detach().numpy()
        
    return sub_traj_embs


def cluster_trajectories(sub_traj_embs, plot=False, num_clusters=8):
    """
    Cluster the trajectory embeddings using X-Means.
    
    Args:
    - sub_traj_embs: np.array, shape (num_sub_trajs, 128)
    - plot: bool, whether to plot the clusters
    - num_clusters: int, number of clusters to cluster the embeddings into
    """
    # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # start analysis.
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(sub_traj_embs, amount_initial_centers).initialize()
    
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    # number of clusters that can be allocated is num_clusters.
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


def compute_dataset_embeddings(cluster_traj_embeddings):
    """
    Compute original and complementary dataset embeddings from cluster 
    embeddings. 
    
    Args:
    - cluster_traj_embeddings: list, list of cluster embeddings
    
    Returns:
    - original_data_embedding: np.array, shape (128,)
    - compl_dataset_embeddings: list, list of complementary dataset embeddings
    """
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
    """
    Create complementary datasets for each cluster.
    
    Args:
    - dataset: MDPDataset
    - sub_traj_embs: np.array, shape (num_sub_trajs, 128)
    - traj_cluster_labels: np.array, shape (num_sub_trajs,)
    - clusters: list, list of clusters
    
    Returns:
    - cluster_datasets: list, list of complementary datasets
    """
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


def compute_explanation_policies(dataset, cluster_datasets, env=None, load_model=False, n_fit_steps=10000):
    """
    Compute explanation policies and predictions on first 1000 observations
    for each complementary dataset.
    
    Args:
    - dataset: MDPDataset
    - cluster_datasets: list, list of complementary datasets
    - env: gym.Env, environment
    - load_model: bool, whether to load the pretrained model
    
    Returns:
    - agents: list, list of explanation policies
    - explanation_predictions: list, list of predictions of the explanation policies
    """
    agents = []
    explanation_predictions = []
    sv_explanation_predictions = []
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
            discrete_sac.fit(cluster_dataset, n_steps=n_fit_steps)
            discrete_sac.save_model("checkpoints/agent_c{}.pt".format(idx))
            
        agents.append(discrete_sac)
        
        # Make Predictions        
        predictions = []
        sv_predictions = []
        actions = list(range(18))
        for observation in dataset.observations[:1000]:
            state_action_values = []
            for action in actions:
                state_action_values.append(discrete_sac.predict_value([observation], [action])[0])
            sv_predictions.append(state_action_values)
            predictions.append(np.argmax(state_action_values))
            # predictions.append(discrete_sac.predict([observation])[0])
        explanation_predictions.append(predictions)
        sv_explanation_predictions.append(sv_predictions)
        
    return agents, explanation_predictions, sv_explanation_predictions


def compute_original_policy(dataset, env=None, load_model=False, n_fit_steps=10000):
    """
    Compute original policy and predictions on first 1000 observations.
    
    Args:
    - dataset: MDPDataset
    - env: gym.Env, environment
    - load_model: bool, whether to load the pretrained model
    
    Returns:
    - original_policy: DiscreteSAC, original policy
    - original_predictions: list, list of predictions of the original policy
    """
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
        original_policy.fit(dataset, n_steps=n_fit_steps)
        original_policy.save_model("checkpoints/agent.pt")
        
    # Make predictions
    original_predictions = []
    sv_original_predictions = []
    actions = list(range(18))
    for observation in dataset.observations[:1000]:
        state_action_values = []
        for action in actions:
        # original_predictions.append([original_policy.predict([observation])][0])
            state_action_values.append(original_policy.predict_value([observation], [action])[0])
        sv_original_predictions.append(state_action_values)
        original_predictions.append(np.argmax(state_action_values))
    
    return original_policy, original_predictions, sv_original_predictions


def generate_attributions(dataset, original_predictions, explanation_predictions, original_data_embedding, compl_dataset_embeddings, clusters):
    """
    Cluster attribution algorithm from the paper. Attributes responsible cluster for first
    100 observations.
    
    Args:
    - dataset: MDPDataset
    - original_predictions: list, list of predictions of the original policy
    - explanation_predictions: list, list of predictions of the explanation policies
    - original_data_embedding: np.array, shape (128,)
    - compl_dataset_embeddings: list, list of complementary dataset embeddings
    - clusters: list, list of clusters
    
    Returns:
    - attributions: list, list of attributions
    """
    attributions = []
    
    for idx, (observation, action, reward, terminal)  in enumerate(zip(dataset.observations, dataset.actions, dataset.rewards, dataset.terminals)):
        # To keep it short, can extend 
        if idx >= 1000:  
            break
        if terminal: 
            continue
              
        original_action = original_predictions[idx]
        agent_predictions = []
        for predictions in explanation_predictions:
            agent_predictions.append(predictions[idx])
        
        cluster_distance = []
        # alternative_actions = []
        for cluster_idx in range(len(explanation_predictions)):
            if agent_predictions[cluster_idx] != original_action:
                cluster_distance.append(wasserstein_distance(original_data_embedding, compl_dataset_embeddings[cluster_idx]))
            
                # alternative_actions.append(cluster_idx)
            # To keep cluster-order intact
            else: 
                cluster_distance.append(1e9)
                    
        responsible_cluster_id = np.argsort(cluster_distance)[0]
        responsible_action = agent_predictions[responsible_cluster_id]
        
        print('-'*50)	
        print(f'Observation - {idx}')
        print(f'Original Action - {original_action}')
        print(f'Explanation Actions - {[action for action in agent_predictions]}')
        print(f'Responsible Cluster - {responsible_cluster_id}')
        print(f"Cluster distances - {cluster_distance}")
        print(f"argsort {np.argsort(cluster_distance)}")
        
        if responsible_action == original_action:
            print('-'*10)
            print("SAME ACTION")
            print("Original action", original_predictions[idx])
            for i, pred in enumerate(explanation_predictions):
                print(f"Explanation actions {i}", pred[idx])
            print("Responsible action", responsible_action)
            print('-'*10)
        
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
                'attributed_trajs':clusters[responsible_cluster_id - 1],
                # 'random_baseline_trajs': list(np.random.randint(0, len(sub_traj_embs), 5)),
                # 'alternate_cluster_trajs': clusters[alternate_cid - 1],
                'responsible_cluster': responsible_cluster_id
            })
    #         for traj in clusters[responsible_data_combination - 1]:
    #             env.plot_traj(offline_data[traj])
        print('-'*10)
    return attributions


def run_trajectory_attribution(load_emb = False, load_model=False, plot_clusters=False, save_attributions=True, n_fit_steps=10000):
    """
    Runs the full code to generate cluster attributions for the Seaquest environment.
    
    Args:
    - load_emb: bool, whether to load the trajectory embeddings
    - load_model: bool, whether to load the pretrained models
    - plot_clusters: bool, whether to plot the clusters
    - save_attributions: bool, whether to save the attributions
    
    Returns:
    - attributions: list, list of attributions
    - metrics: dict, dictionary of metrics
    """
    # Load dataset and environment
    start = time.time()
    dataset, env = get_dataset('seaquest-mixed-v4')
    print(f"Dataset loaded in {time.time() - start} seconds")
    
    if not load_emb:
        start = time.time()
        # Preprocess dataset into sub trajectories
        observation_traj, action_traj, reward_traj, num_sub_trajs = create_trajectories_from_dataset(dataset, sub_traj_len=30)
        print(f"Dataset preprocessed in {time.time() - start} seconds")
        np.save("data/observation_traj.npy", observation_traj)
        np.save("data/action_traj.npy", action_traj)
        np.save("data/reward_traj.npy", reward_traj)
        
        start = time.time()
        # Encode sub trajectories using decision transformer
        sub_traj_embs = encode_trajectories(observation_traj, action_traj, reward_traj, num_sub_trajs, batch_size=30)
        print(f"Sub trajectories encoded in {time.time() - start} seconds")
        np.save("data/sub_traj_embs.npy", sub_traj_embs)
    else:
        # np.save("seaquest/data/sub_traj_embs.npy", sub_traj_embs)
        start = time.time()
        sub_traj_embs = np.load("data/sub_traj_embs.npy")
        print(f"Sub trajectories loaded in {time.time() - start} seconds")
    
    start = time.time()
    # Cluster trajectories
    clusters, cluster_traj_embeddings, traj_cluster_labels = cluster_trajectories(sub_traj_embs, plot=plot_clusters, num_clusters=8)
    print(f"Trajectories clustered in {time.time() - start} seconds")

    start = time.time()
    # Compute original dataset embedding and complementary dataset embeddings
    original_data_embedding, compl_dataset_embeddings = compute_dataset_embeddings(cluster_traj_embeddings)
    print(f"Dataset embeddings computed in {time.time() - start} seconds")
    
    start = time.time()
    # Create complementary datasets
    cluster_datasets = create_complementary_dataset(dataset, sub_traj_embs, traj_cluster_labels, clusters)
    print(f"Complementary datasets created in {time.time() - start} seconds")
   
    start = time.time()
    # Fit explanation policies
    explanation_policies, explanation_predictions, sv_explanation_predictions = compute_explanation_policies(dataset, cluster_datasets, env=env, load_model=load_model, n_fit_steps=n_fit_steps)
    print(f"Explanation policies fitted in {time.time() - start} seconds")

    start = time.time()
    # Fit original policy
    original_policy, original_predictions, sv_original_predictions = compute_original_policy(dataset, env=env, load_model=load_model, n_fit_steps=n_fit_steps)
    print(f"Original policy fitted in {time.time() - start} seconds")
   
    start = time.time()
    # Generate attributions
    attributions = generate_attributions(dataset, original_predictions, explanation_predictions, original_data_embedding, compl_dataset_embeddings, clusters)
    print(f"Attributions generated in {time.time() - start} seconds")
    if save_attributions:
        np.save("data/attributions.npy", attributions)
        
    np.save("data/sv_explanation_predictions.npy", sv_explanation_predictions)
    np.save("data/sv_original_predictions.npy", sv_original_predictions)
    # sv_explanation_predictions = np.load("data/sv_explanation_predictions.npy")
    # sv_original_predictions = np.load("data/sv_original_predictions.npy")
    # attributions = np.load("data/attributions.npy", allow_pickle=True)
       
    metrics = get_metrics(sv_explanation_predictions, sv_original_predictions, attributions, original_data_embedding, compl_dataset_embeddings)
    
    for cid, cluster in enumerate(clusters):
        print("Len of cluster ", len(cluster))
    
    return attributions, metrics
    
if __name__ == "__main__":
    
    attributions, metrics = run_trajectory_attribution(load_emb=True, load_model=False, plot_clusters=False, save_attributions=True, n_fit_steps=10000)
    
    # TODO:
    # - Fit discrete SAC agents for more epochs
    # - add reward scaler?
    # - More visualizations