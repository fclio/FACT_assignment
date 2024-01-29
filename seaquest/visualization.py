import numpy as np 
import matplotlib.pyplot as plt
import gzip
import imageio
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from constants import ACTION_DICT


def plot_clusters(sub_traj_embs, traj_cluster_labels, clusters, emb_ids):
    """
    Brings down dimensions of the embeddings to 2D and plots the clusters.
    
    Args: 
    - sub_traj_embs: np.array, shape (num_sub_trajs, 128)
    - traj_cluster_labels: np.array, shape (num_sub_trajs,)
    - clusters: list, list of clusters
    - emb_ids: list, list of indices of embeddings to plot
    """
    palette = sns.color_palette('husl', len(clusters) + 1)
    pca_traj = PCA(n_components=2)
    pca_traj_embeds = pca_traj.fit_transform(sub_traj_embs)

    plotting_data = {'feature 1': pca_traj_embeds[emb_ids, 0],
                    'feature 2': pca_traj_embeds[emb_ids, 1],
                    'cluster id': traj_cluster_labels[emb_ids]}
    df = pd.DataFrame(plotting_data)

    plt.figure(figsize=(5,4))
    data_ax = sns.scatterplot(x='feature 1',
                            y='feature 2',
                            hue='cluster id',
                            palette=palette[:len(clusters)],
                            data=df,
                            legend=True)
    plt.legend(title = '$c_{j}$', loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=5)
    # plt.legend(title = '$c_{j}$', loc='center left', bbox_to_anchor=(1., 0.7), ncol=2)
    # for cid, _ in enumerate(cluster_data_embeds):
    #     data_ax.text(pca_traj_embeds[:, 0][cid],
    #                  pca_traj_embeds[:, 1][cid],
    #                  str(cid),
    #                  horizontalalignment='left',
    #                  size='medium',
    #                  color='black',
    #                  weight='semibold')
    plt.tight_layout()
    plt.title("Trajectory Clustering Seaquest")
    plt.savefig('images/traj_clustering_grid.pdf')
    plt.show()
    

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
    Create a gif of the explanation trajectories in expl_traj_id consisting of 30 frames
    """
    for traj_id in expl_traj_id:
        traj_id = int(traj_id)
        images = []
        for i in range(30):
            images.append(observation_traj[traj_id][i][0])
        if resp:
            imageio.mimsave(f'images/explanation_{orig_obs_id}_{traj_id}.gif', images, loop=10000)
        else:
            imageio.mimsave(f'images/False_explanation_{orig_obs_id}_{traj_id}.gif', images, loop=10000)

    
def plot_original_state(observation, obs_id, orig_action):
    fig = plt.figure()
    plt.imshow(observation, cmap="gray")
    plt.title(f"Original Action: {orig_action}")
    plt.axis("off")
    plt.savefig(f"images/original_state_{obs_id}.png")
    
    
# if __name__ == "__main__":
#     datasets_names = ["observation"]
#     datasets = {}
#     for dataset_name in datasets_names:
#         with gzip.open("data/"+dataset_name+".gz", 'rb') as f:
#             datasets[dataset_name] = np.load(f, allow_pickle=False)
            
#     observation_traj = np.load("data/observation_traj.npy", allow_pickle=True)
#     reward_traj = np.load("data/reward_traj.npy", allow_pickle=True)
#     action_traj = np.load("data/action_traj.npy", allow_pickle=True)

#     traj_id = [470, 473, 474] #[13, 16, 18]
#     false_ids = [7931]
#     obs_id = 2
#     action = "DOWN"
    
#     plot_original_state(datasets["observation"][obs_id], obs_id, action)
#     create_gif(observation_traj, obs_id, traj_id)
#     create_gif(observation_traj, obs_id, false_ids, resp=False)

    
    # plot_explanation(obs_id, traj_id, action, observation_traj, action_traj)
    # plot_trajectories(998, [13,16,18], action, observation_traj, action_traj)
    # plot_explanation(997, 0, action, observation_traj, action_traj)
    # plot_explanation(997, 11, action, observation_traj, action_traj)
    # plot_explanation(997, 23, action, observation_traj, action_traj)
    
    # state 998 action UP, explained by trajectories 13, 16, 18 from cluster 0
    # state 997 action UP, explained by trajectories 0. 11. 23 from cluster 3
    # state 0 action DOWN, explained by trajectories 36, 37, 48 from cluster 3
    # state 2 action DOWN, explained by trajectories 470, 473, 474 from cluster 3 