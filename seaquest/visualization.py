import numpy as np 
import matplotlib.pyplot as plt
import gzip
import imageio
from run import ACTION_DICT


def plot_explanation(orig_obs_id, expl_traj_id, orig_action, observation_traj, action_traj):
    """
    Plot original observation large on the left and 2 frames from explanation trajectories on the right
    """
    fig = plt.figure(constrained_layout=True)
    axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                            gridspec_kw={'width_ratios':[2, 1]})
    axs['Left'].imshow(datasets["observation"][orig_obs_id], cmap="gray")
    axs['Left'].set_title(f"Original Action: {orig_action}")

    axs['TopRight'].imshow(observation_traj[expl_traj_id][0][0], cmap="gray")
    axs['TopRight'].set_title(f"New action: {ACTION_DICT[action_traj[expl_traj_id][0][0]]}")

    axs['BottomRight'].imshow(observation_traj[expl_traj_id][1][0], cmap="gray")
    axs['BottomRight'].set_title(f"New action: {ACTION_DICT[action_traj[expl_traj_id][1][0]]}")
    fig.suptitle(f"Explanation (right) of action original action (left)")
    plt.savefig(f"images/explanation_{orig_obs_id}_{expl_traj_id}.png")
    

def plot_trajectories(orig_obs_id, expl_traj_id, orig_action, observation_traj, action_traj):
    """
    Plot the first six steps of three explanation trajectories
    """
    fig, axs = plt.subplots(3, 6, constrained_layout=True)

    for idx, traj_id in enumerate(expl_traj_id):
        # Plot the first three steps of the explanation trajectory
        for i in range(6):
            axs[idx, i].imshow(observation_traj[traj_id][i][0], cmap="gray")
            axs[idx, i].set_title(f"{ACTION_DICT[action_traj[traj_id][i][0]]}")
            axs[idx, i].axis("off")

    fig.suptitle(f"Three explanation trajectories")
    plt.savefig(f"images/explanation_{orig_obs_id}_{expl_traj_id}.png")


def create_gif(observation_traj, orig_obs_id, expl_traj_id, resp=True):
    """
    Create a gif of the explanation trajectory consisting of 30 frames
    """
    for traj_id in expl_traj_id:
        images = []
        for i in range(30):
            images.append(observation_traj[traj_id][i][0])
        if resp:
            imageio.mimsave(f'images/explanation_{orig_obs_id}_{traj_id}.gif', images)
        else:
            imageio.mimsave(f'images/False_explanation_{orig_obs_id}_{traj_id}.gif', images)

    
def plot_original_state(observation, obs_id, orig_action):
    fig = plt.figure()
    plt.imshow(observation, cmap="gray")
    plt.title(f"Original Action: {orig_action}")
    plt.axis("off")
    plt.savefig(f"images/original_state_{obs_id}.png")
    
if __name__ == "__main__":
    datasets_names = ["observation"]
    datasets = {}
    for dataset_name in datasets_names:
        with gzip.open("data/"+dataset_name+".gz", 'rb') as f:
            datasets[dataset_name] = np.load(f, allow_pickle=False)
            
    observation_traj = np.load("data/observation_traj.npy", allow_pickle=True)
    reward_traj = np.load("data/reward_traj.npy", allow_pickle=True)
    action_traj = np.load("data/action_traj.npy", allow_pickle=True)

    traj_id = [23] #[13, 16, 18]
    obs_id = 998
    action = "UP"
    
    # plot_original_state(datasets["observation"][obs_id], obs_id, action)
    # create_gif(observation_traj, obs_id, traj_id)
    create_gif(observation_traj, obs_id, traj_id, resp=False)
    
    # plot_explanation(obs_id, traj_id, action, observation_traj, action_traj)
    # plot_trajectories(998, [13,16,18], action, observation_traj, action_traj)
    # plot_explanation(997, 0, action, observation_traj, action_traj)
    # plot_explanation(997, 11, action, observation_traj, action_traj)
    # plot_explanation(997, 23, action, observation_traj, action_traj)
    
    # state 998 action UP, explained by trajectories 13, 16, 18 from cluster 0
    # state 997 action UP, explained by trajectories 0. 11. 23 from cluster 3