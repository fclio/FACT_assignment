"""
TODO HEADER
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import d3rlpy

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import (
    make_prefix,
    update_context,
)
from trajectory.search.sampling import forward

from sklearn.decomposition import PCA
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from scipy.stats import wasserstein_distance
from moviepy.editor import VideoFileClip

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-v2'
    config: str = 'config.offline'

# utils

def cluster_trajectories(trajectories, n_clusters=10):
    """TODO"""

    # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # start analysis.
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(trajectories, amount_initial_centers).initialize()
    
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    # number of clusters that can be allocated is 10.
    xmeans_instance = xmeans(trajectories, initial_centers, n_clusters)
    xmeans_instance.process()
    
    # Extract clustering results: clusters
    idxs_per_cluster = xmeans_instance.get_clusters()

    clusters = []
    for i in range(len(trajectories)):
        for j in range(len(idxs_per_cluster)):
            if i in idxs_per_cluster[j]: clusters.append(j)

    return idxs_per_cluster, np.array(clusters)
 
def softmax(x, temp):
    """TODO"""
    max_x = np.max(x)
    return np.exp(np.divide(x-max_x,temp)) / np.sum(np.exp(np.divide(x-max_x,temp)))

def generate_data_embedding(trajectory_embeddings, temperature=10000):
    """TODO"""

    embedding = np.sum(trajectory_embeddings, axis=0)
    embedding = softmax(embedding, temperature)
    

    return embedding

def embed_trajectory(gpt, discretizer, observations, actions, rewards, preprocess_fn):
    """TODO"""

    context = []

    output = []

    for i in range(len(observations)):
        observation = observations[i]
        action = actions[i]
        reward = rewards[i]

        observation = preprocess_fn(observation)

        prefix = make_prefix(discretizer, context, observation, True)

        out = forward(gpt, prefix)

        if len(context) >= 9:
            context.pop(0)
            if len(output) == 0:
                output = out.detach().numpy()[0]
            else:
                output = np.concatenate((output, out.detach().numpy()[0][217:]), axis=0)

        context = update_context(context, discretizer, observation, action, reward, len(observations))

    emb = np.mean(output, axis=0)
    return emb

def create_complementary_dataset(dataset, idxs, trajectory_length=10, inverse=False):
    """TODO"""

    observations = []
    actions = []
    rewards = []
    terminals = []

    n_trajs = int(1000000/trajectory_length)
    for i in range(n_trajs):
        condition = i not in idxs
        if inverse: condition = not condition

        if condition:
            observations += list(dataset.observations[trajectory_length*i:trajectory_length*(i+1)])
            actions += list(dataset.actions[trajectory_length*i:trajectory_length*(i+1)])
            rewards += list(dataset.rewards[trajectory_length*i:trajectory_length*(i+1)])

    terminals = np.tile([0]*(trajectory_length-1)+[1], int(len(observations)/trajectory_length))

    new_dataset = d3rlpy.dataset.MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
    )
    return new_dataset
    

def clusters_to_idxs(clusters):
    idxs_per_cluster = []
    for i in np.sort(np.unique(clusters)):
        idxs_per_cluster.append(list(np.argwhere(clusters == i).flatten()))
    
    return idxs_per_cluster


def main():

    args = Parser().parse_args('plan')


    ### DATASET ###

    dataset_d3, env = d3rlpy.datasets.get_dataset("halfcheetah-medium-v2")

    ### IMPORTANT DEFINITIONS XRL SCRIPT ###

    load_embeddings = False
    load_clusters = True
    load_agents = True
    generate_human_study = False
    
    seed = 4 
    trajectory_length = 25 # 10 = max
    n_clusters = 10
    k = 3
    temperature = 10000
    logging_folder = f"results/v2_models_100k_{seed}"
    training_steps = 100000

    d3rlpy.seed(seed)

    if load_embeddings:
        embeddings = np.load(f"{logging_folder}/embeddings.npy")
    else:
   
        ### TRAJECTORY TRANSFORMER ###
    
        dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
                'data_config.pkl')
        gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
                epoch=args.gpt_epoch, device=args.device)
        env = datasets.load_environment(args.dataset)
        discretizer = dataset.discretizer
        preprocess_fn = datasets.get_preprocess_fn(env.name)
    
        ### TRAJECTORY EMBEDDINGS ###
    
        embeddings = []
        n_trajs = int(1000000/trajectory_length)
        for i in range(n_trajs):
            observations = dataset_d3.observations[trajectory_length*i:trajectory_length*(i+1)]
            actions = dataset_d3.actions[trajectory_length*i:trajectory_length*(i+1)]
            rewards = dataset_d3.rewards[trajectory_length*i:trajectory_length*(i+1)]
            terminals = dataset_d3.terminals[trajectory_length*i:trajectory_length*(i+1)]
            emb = embed_trajectory(gpt, discretizer, observations, actions, rewards, preprocess_fn)
            embeddings.append(emb)
        embeddings = np.array(embeddings)
        np.save(f"{logging_folder}/embeddings.npy", embeddings)

    print("embeddings ready")

    ### TRAJECTORY CLUSTERS ###

    if load_clusters:
        clusters = np.load(f"{logging_folder}/clusters.npy")
        idxs_per_cluster = clusters_to_idxs(clusters)
    else:
        idxs_per_cluster, clusters = cluster_trajectories(embeddings, n_clusters)
        np.save(f"{logging_folder}/clusters.npy", clusters)

    print("clusters ready")

    ### PCA (solely for visualization) ###
 
    pca_idxs = np.random.choice(len(embeddings), 500, replace=False)

    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings[pca_idxs])
    pca_clusters = clusters[pca_idxs]

    print("pca ready")

    ### COMPLEMENTARY DATASETS & CLUSTER EMBEDDINGS (also plotting PCA) ###
    d_orig = generate_data_embedding(embeddings, temperature=temperature)

    unique_clusters = np.unique(clusters)
    d_j = []
    complementary_datasets = []
    cluster_datasets = []
    fig, ax = plt.subplots(figsize=(5,4))
    for j in np.sort(unique_clusters):
        d_j.append(generate_data_embedding(embeddings[clusters != j], temperature=temperature))
        ax.scatter(pca_embeddings[pca_clusters == j][:,0], pca_embeddings[pca_clusters == j][:,1], label=j)
        complementary_datasets.append(create_complementary_dataset(dataset_d3, idxs_per_cluster[j], trajectory_length))
        cluster_datasets.append(create_complementary_dataset(dataset_d3, idxs_per_cluster[j], trajectory_length, inverse=True))
    
    original_dataset = create_complementary_dataset(dataset_d3, [], trajectory_length)

    ax.legend(title="$c_j$", bbox_to_anchor=(0.5, 1.2), loc="lower center", ncol=5)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title("Trajectory Clustering HalfCheetah")
    plt.tight_layout()

    plt.savefig(f"{logging_folder}/pca.pdf")

    print("complementary datasets ready")

    ### AGENT TRAINING (original & complementary) ###

    agent_orig = d3rlpy.algos.SAC(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        batch_size=512)

    if load_agents:
        agent_orig.build_with_dataset(original_dataset)
        agent_orig.load_model(f"{logging_folder}/agent_orig.pt")
    else:
        agent_orig.fit(original_dataset, n_steps=training_steps)
        agent_orig.save_model(f"{logging_folder}/agent_orig.pt")

    agents_compl = []

    for i in range(len(complementary_datasets)):
        agent = d3rlpy.algos.SAC(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            batch_size=512)
        if load_agents:
            agent.build_with_dataset(complementary_datasets[i])
            agent.load_model(f"{logging_folder}/agent_compl_{i}.pt")
        else:
            agent.fit(complementary_datasets[i], n_steps=training_steps)
            agent.save_model(f"{logging_folder}/agent_compl_{i}.pt")
        agents_compl.append(agent)

    print("agents ready")

    ### OBSERVATION EXPLANATION (cluster assignment) ###

    original_state = np.random.get_state()
    np.random.seed(0)
    idxs_to_explain = np.random.choice(range(len(dataset_d3.observations)), 1000, replace=False)
    np.random.set_state(original_state)

    observations_to_explain = [dataset_d3.observations[i] for i in idxs_to_explain] 

    ISVE = []
    ISVE_orig = 0.
    LMAAVD = []
    ACM = []
    NWD = []
    CAF = [0] * len(d_j)

    if generate_human_study:
        ctr = 0
        unrelated_idxs = [690, 1520, 3030, 6050, 7080, 8030]
        if not os.path.isdir(f"{logging_folder}/mp4s"): os.mkdir(f"{logging_folder}/mp4s")
        if not os.path.isdir(f"{logging_folder}/gifs"): os.mkdir(f"{logging_folder}/gifs")

    for observation_to_explain in observations_to_explain:
        action_orig = agent_orig.predict([observation_to_explain])

        actions_compl = []
        for agent in agents_compl:
            actions_compl.append(agent.predict([observation_to_explain]))

        action_dists = []
        for action in actions_compl:
            action_dists.append(np.linalg.norm(action_orig-action))

        topk = np.argpartition(action_dists, -k)[-k:]

        d_w = {}
        for idx in topk:
            d_w[idx] = wasserstein_distance(d_j[idx], d_orig)

        cluster_assignment = min(d_w, key=d_w.get)

        ### OBSERVATION EXPLANATION (representing cluster with 1 trajectory) ###

        distances_to_obs = [np.linalg.norm(observation_to_explain-obs) for obs in cluster_datasets[cluster_assignment].observations]
        trajectory_to_assign = np.floor(np.argmin(distances_to_obs) / trajectory_length)
        assigned_trajectory = np.arange(trajectory_to_assign * trajectory_length, (trajectory_to_assign+1) * trajectory_length)

        ### OBSERVATION EXPLANATION (metrics) ###

        # Initial State Value Estimate
 
        V_s = 0.
        for _ in range(10):
            sampled_action = agent_orig.sample_action([observation_to_explain])
            Q_sa = agent_orig.predict_value([observation_to_explain], [sampled_action[0]])[0]
            V_s += Q_sa
        ISVE_orig += V_s/10
 
        new_ISVE = []
        for agent in agents_compl:
            V_s = 0.
            for _ in range(10):
                sampled_action = agent.sample_action([observation_to_explain])
                Q_sa = agent.predict_value([observation_to_explain], [sampled_action[0]])[0]
                V_s += Q_sa
            new_ISVE.append(V_s/10)
        ISVE.append(new_ISVE)
 
        # Local Mean Absolute Action-Value Difference
        Q_orig = agent_orig.predict_value([observation_to_explain], [action_orig[0]])
        Q_j = [agent_orig.predict_value([observation_to_explain], [ac[0]]) for ac in actions_compl]
        LMAAVD.append(np.abs(np.array(Q_j) - Q_orig).flatten())
 
        # Action Contrast Measure
        ACM.append(action_dists)
 
        # Normalized Wasserstein distance (between cluster embeddings)
        wasser = np.array([wasserstein_distance(d, d_orig) for d in d_j])
        NWD.append(list((wasser-np.min(wasser))/(np.max(wasser)-np.min(wasser))))
 
        # Cluster attribution frequency
        CAF[cluster_assignment] += 1
 
        if generate_human_study:
            ### RENDERING ###
            if not os.path.isdir(f"{logging_folder}/gifs/question_{ctr}"): os.mkdir(f"{logging_folder}/gifs/question_{ctr}")

            rollout = dataset_d3.observations[idxs_to_explain[ctr]-25:idxs_to_explain[ctr]]
            print(idxs_to_explain[ctr])
            print(rollout)
            renderer = utils.make_renderer(args)
            rollout_savepath = f"{logging_folder}/mp4s/question_{ctr}/traj_to_explain.mp4"
            renderer.render_rollout(rollout_savepath, rollout, fps=10)
            videoClip = VideoFileClip(f"{logging_folder}/mp4s/question_{ctr}/traj_to_explain.mp4")
            videoClip.write_gif(f"{logging_folder}/gifs/question_{ctr}/traj_to_explain.gif")


            rollout = cluster_datasets[cluster_assignment].observations[assigned_trajectory.astype(int)]
            print(rollout)
            renderer = utils.make_renderer(args)
            rollout_savepath = f"{logging_folder}/mp4s/question_{ctr}/traj_assigned_cluster_attr.mp4"
            renderer.render_rollout(rollout_savepath, rollout, fps=10)
            videoClip = VideoFileClip(f"{logging_folder}/mp4s/question_{ctr}/traj_assigned_cluster_attr.mp4")
            videoClip.write_gif(f"{logging_folder}/gifs/question_{ctr}/traj_assigned_cluster_attr.gif")

            random_trajs = np.random.randint(len(cluster_datasets[cluster_assignment])//25, size=3)

            rollout = cluster_datasets[cluster_assignment].observations[random_trajs[0]*25:(random_trajs[0]+1)*25]
            print(rollout)
            renderer = utils.make_renderer(args)
            rollout_savepath = f"{logging_folder}/mp4s/question_{ctr}/traj_assigned_cluster_1.mp4"
            renderer.render_rollout(rollout_savepath, rollout, fps=10)
            videoClip = VideoFileClip(f"{logging_folder}/mp4s/question_{ctr}/traj_assigned_cluster_1.mp4")
            videoClip.write_gif(f"{logging_folder}/gifs/question_{ctr}/traj_assigned_cluster_1.gif")

            rollout = cluster_datasets[cluster_assignment].observations[random_trajs[1]*25:(random_trajs[1]+1)*25]
            print(rollout)
            renderer = utils.make_renderer(args)
            rollout_savepath = f"{logging_folder}/mp4s/question_{ctr}/traj_assigned_cluster_2.mp4"
            renderer.render_rollout(rollout_savepath, rollout, fps=10)
            videoClip = VideoFileClip(f"{logging_folder}/mp4s/question_{ctr}/traj_assigned_cluster_2.mp4")
            videoClip.write_gif(f"{logging_folder}/gifs/question_{ctr}/traj_assigned_cluster_2.gif")


            rollout = cluster_datasets[cluster_assignment].observations[random_trajs[2]*25:(random_trajs[2]+1)*25]
            print(rollout)
            renderer = utils.make_renderer(args)
            rollout_savepath = f"{logging_folder}/mp4s/question_{ctr}/traj_assigned_cluster_3.mp4"
            renderer.render_rollout(rollout_savepath, rollout, fps=10)
            videoClip = VideoFileClip(f"{logging_folder}/mp4s/question_{ctr}/traj_assigned_cluster_3.mp4")
            videoClip.write_gif(f"{logging_folder}/gifs/question_{ctr}/traj_assigned_cluster_3.gif")


            different_cluster = 0 if cluster_assignment != 0 else 1
            random_trajs = np.random.randint(len(cluster_datasets[different_cluster])//25, size=3)
            rollout = cluster_datasets[different_cluster].observations[random_trajs[2]*25:(random_trajs[2]+1)*25]
            print(rollout)
            renderer = utils.make_renderer(args)
            rollout_savepath = f"{logging_folder}/mp4s/question_{ctr}/traj_different_cluster.mp4"
            renderer.render_rollout(rollout_savepath, rollout, fps=10)
            videoClip = VideoFileClip(f"{logging_folder}/mp4s/question_{ctr}/traj_different_cluster.mp4")
            videoClip.write_gif(f"{logging_folder}/gifs/question_{ctr}/traj_different_cluster.gif")

            rollout = dataset_d3.observations[unrelated_idxs[ctr]-25:unrelated_idxs[ctr]]
            print(rollout)
            renderer = utils.make_renderer(args)
            rollout_savepath = f"{logging_folder}/mp4s/question_{ctr}/traj_unrelated.mp4"
            renderer.render_rollout(rollout_savepath, rollout, fps=10)
            videoClip = VideoFileClip(f"{logging_folder}/mp4s/question_{ctr}/traj_unrelated.mp4")
            videoClip.write_gif(f"{logging_folder}/gifs/question_{ctr}/traj_unrelated.gif")

            ctr += 1

    ### RESULTS ###
    ISVE_orig /= len(observations_to_explain)
    ISVE = np.mean(ISVE, axis=0)
    LMAAVD = np.mean(LMAAVD, axis=0)
    ACM = np.mean(ACM, axis=0)
    NWD = np.mean(NWD, axis=0)
    CAF = np.array(CAF) / np.sum(CAF)

    print("ISVE orig:", ISVE_orig)
    print("ISVE:",ISVE)
    print("LMAAVD:",LMAAVD)
    print("ACM:",ACM)
    print("NWD:",NWD)
    print("CAF:",CAF)


if __name__ == "__main__":
    main()