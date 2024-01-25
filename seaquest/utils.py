import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import metrics

from gpt import GPT, GPTConfig


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


def plot_clusters(cluster_data_embeds, sub_traj_embs, traj_cluster_labels, clusters, emb_ids):
    """
    Brings down dimensions of the embeddings to 2D and plots the clusters.
    
    Args: 
    - cluster_data_embeds: np.array, shape (num_clusters, 128)
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
    
    
def get_data_embedding(traj_embeddings):
    """
    Create a data embedding by averaging the trajectory embeddings.
    Basically normalized softmax.
    """
    return np.exp(np.array(traj_embeddings).sum(axis=0)/10.)/np.sum(np.exp(np.array(traj_embeddings).sum(axis=0)/10.))


def calculate_similarities(observation_trajs, observation, cluster):
    traj_similarity = []
    
    # For all trajectories in the cluster
    for traj_id in cluster:
        similarity = []
        
        # check all frames of the trajectory
        for i in range(30):
            # Calculate the structural similarity between the observation and the frame
            similarity.append(metrics.structural_similarity(observation_trajs[traj_id][i][0], observation[0], full=True)[0])
        traj_similarity.append([np.max(similarity), traj_id, np.argmax(similarity)])
 
    # Sort the array with the highest similarity to observation on top
    sort_indices = np.argsort(np.array(traj_similarity)[:, 0])
    traj_similarity = np.array(traj_similarity)[sort_indices][::-1]
    return traj_similarity
    