import numpy as np 
from skimage import metrics
import gzip

from gpt import GPT, GPTConfig
from visualization import plot_original_state, create_gif


def get_decision_transformer(vocab_size=18, block_size=90, model_type="reward_conditioned", timesteps=2719):
    """
    Loads the pretrained decision transformer model.
    
    Args:
    - vocab_size: int, size of the action space
    - block_size: int, length of the sub trajectories
    - model_type: str, type of decision transformer
    - timesteps: int, maximum number of timesteps
    
    Returns:
    - model: GPT, pretrained decision transformer model
    """
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
    
    
def get_data_embedding(traj_embeddings):
    """
    Create a data embedding by averaging the trajectory embeddings.
    Basically normalized softmax.
    """
    return np.exp(np.array(traj_embeddings).sum(axis=0)/1000.)/np.sum(np.exp(np.array(traj_embeddings).sum(axis=0)/1000.))


def calculate_similarities(observation_trajs, observation, cluster, k, metric="ssim"):
    """
    Calculate similarity between given observation and trajectories in the cluster using the given metric.
    Sort the trajectories by similarity.
    
    Args:   
    - observation_trajs: list, list of trajectories
    - observation: np.array, shape (1, 84, 84)
    - cluster: list, list of trajectory ids
    - k: int, number of trajectories to return
    - metric: str, metric to use for similarity calculation
    
    Returns:
    - traj_similarity: list, top k list of similarities
    """
    traj_similarity = []
    
    # For all trajectories in the cluster
    for traj_id in cluster:
        similarity = []
        
        # check all frames of the trajectory
        for i in range(30):
            # Calculate the structural similarity between the observation and the frame
            if metric == "ssim":
                similarity.append(metrics.structural_similarity(observation_trajs[traj_id][i][0], observation, full=True)[0])
            elif metric == "mse":
                similarity.append(metrics.mean_squared_error(observation_trajs[traj_id][i][0], observation))
        traj_similarity.append([np.max(similarity), traj_id, np.argmax(similarity)])
 
    # Sort the array with the highest similarity to observation on top
    sort_indices = np.argsort(np.array(traj_similarity)[:, 0])
    traj_similarity = np.array(traj_similarity)[sort_indices][::-1]
    return traj_similarity[0:k]


def pick_k_trajectories(attributions, observation_traj, observation, obs_id, k=3, metric="ssim"):
    """
    Pick k individual trajectories from a responsible cluster and save to images/ folder..
    
    Args:
    - attributions: list, list of attributions
    - observation_traj: list, list of trajectories
    - observation: np.array, shape (1, 84, 84)
    - obs_id: int, observation id
    - k: int, number of trajectories to return
    - metric: str, metric to use for similarity calculation
    """
    trajectories_in_cluster = attributions[obs_id]["attributed_trajs"]
    
    # Pick false explanation
    for i in range(2):
        rng = np.random.default_rng()
        false_explanation = rng.integers(0, 1000)
        while false_explanation in trajectories_in_cluster:
            false_explanation = rng.integers(0, 1000)

        # Create gif of the false explanation trajectory and save to images/False_explanation_{obs_id}_{traj_id}.gif
        create_gif(observation_traj, obs_id, [false_explanation], resp=False) 
    
    # Calculate similarities
    traj_similarity = calculate_similarities(observation_traj, observation, trajectories_in_cluster, k, metric)
    
    # Plot original state and save image to images/original_state_{obs_id}.png
    plot_original_state(observation, obs_id, attributions[obs_id]["orig_act"])
    # Create gif of the explanation trajectories and save to images/explanation_{obs_id}_{traj_id}.gif
    create_gif(observation_traj, obs_id, traj_similarity[:,1], resp=True)
    
    
if __name__ == "__main__":
    attributions = np.load("data/attributions.npy", allow_pickle=True)
    observation_trajs = np.load("data/observation_traj.npy", allow_pickle=True)
    datasets_names = ["observation"]
    datasets = {}
    for dataset_name in datasets_names:
        with gzip.open("data/"+dataset_name+".gz", 'rb') as f:
            datasets[dataset_name] = np.load(f, allow_pickle=False)
    
    
    pick_k_trajectories(attributions, observation_trajs, datasets["observation"][50], 50, k=3, metric="mse")
