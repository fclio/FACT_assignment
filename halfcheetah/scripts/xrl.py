import json
import pdb
from os.path import join

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import (
    make_prefix,
    update_context,
)
from trajectory.search.sampling import forward

import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import numpy as np
import d3rlpy
import math as mt
from sklearn.cluster import KMeans
from sklearn import datasets as skdatasets
from sklearn.decomposition import PCA

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from scipy.stats import wasserstein_distance

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.offline'

# utils
    
class XMeans:
    def loglikelihood(self, r, rn, var, m, k):
        l1 = - rn / 2.0 * mt.log(2 * mt.pi)
        l2 = - rn * m / 2.0 * mt.log(var)
        l3 = - (rn - k) / 2.0
        l4 = rn * mt.log(rn)
        l5 = - rn * mt.log(r)

        return l1 + l2 + l3 + l4 + l5

    def __init__(self, X, kmax = 20):
        self.X = X
        self.num = np.size(self.X, axis=0)
        self.dim = np.size(X, axis=1)
        self.KMax = kmax

    def fit(self):
        k = 1
        X = self.X
        M = self.dim
        num = self.num

        while(1):
            ok = k

            #Improve Params
            kmeans = KMeans(n_clusters=k).fit(X)
            labels = kmeans.labels_
            m = kmeans.cluster_centers_

            #Improve Structure
            #Calculate BIC
            p = M + 1

            obic = np.zeros(k)

            for i in range(k):
                rn = np.size(np.where(labels == i))
                var = np.sum((X[labels == i] - m[i])**2)/float(rn - 1)
                obic[i] = self.loglikelihood(rn, rn, var, M, 1) - p/2.0*mt.log(rn)

            #Split each cluster into two subclusters and calculate BIC of each splitted cluster
            sk = 2 #The number of subclusters
            nbic = np.zeros(k)
            addk = 0

            for i in range(k):
                ci = X[labels == i]
                r = np.size(np.where(labels == i))

                kmeans = KMeans(n_clusters=sk).fit(ci)
                ci_labels = kmeans.labels_
                sm = kmeans.cluster_centers_

                for l in range(sk):
                    rn = np.size(np.where(ci_labels == l))
                    var = np.sum((ci[ci_labels == l] - sm[l])**2)/float(rn - sk)
                    nbic[i] += self.loglikelihood(r, rn, var, M, sk)

                p = sk * (M + 1)
                nbic[i] -= p/2.0*mt.log(r)

                if obic[i] < nbic[i]:
                    addk += 1

            k += addk

            if ok == k or k >= self.KMax:
                break


        #Calculate labels and centroids
        kmeans = KMeans(n_clusters=k).fit(X)
        self.labels = kmeans.labels_
        self.k = k
        self.m = kmeans.cluster_centers_


def cluster_trajectories(trajectories):
    xmeans_instance = XMeans(trajectories, kmax=10)
    xmeans_instance.fit()

    clusters = xmeans_instance.labels
    return clusters

def cluster_trajectories_2(trajectories):
    # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # start analysis.
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(trajectories, amount_initial_centers).initialize()
    
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    # number of clusters that can be allocated is 10.
    xmeans_instance = xmeans(trajectories, initial_centers, 10)
    xmeans_instance.process()
    
    # Extract clustering results: clusters
    idxs_per_cluster = xmeans_instance.get_clusters()

    clusters = []
    for i in range(len(trajectories)):
        for j in range(len(idxs_per_cluster)):
            if i in idxs_per_cluster[j]: clusters.append(j)

    return idxs_per_cluster, np.array(clusters)
 
# https://github.com/sascha-kirch/ML_Notebooks/blob/main/Softmax_Temperature.ipynb
def softmax(x, temp):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(np.divide(x,temp)) / np.sum(np.exp(np.divide(x,temp)))

def generate_data_embedding(trajectory_embeddings, normalizing_factor=1, temperature=1):
    embedding = np.sum(trajectory_embeddings, axis=0) / normalizing_factor
    embedding = softmax(embedding, temperature)
    return embedding

def embed_trajectory(gpt, discretizer, observations, actions, rewards, preprocess_fn):
    context = []

    for i in range(len(observations)):
        observation = observations[i]
        action = actions[i]
        reward = rewards[i]

        observation = preprocess_fn(observation)

        # print(observation)
        prefix = make_prefix(discretizer, context, observation, True)
        # print("prefix", prefix.shape)

        out = forward(gpt, prefix)
        # print("out", out.shape)
        context = update_context(context, discretizer, observation, action, reward, len(observations))
        # print("cotext", context)
    
    emb = []
    for context_step in context:
        emb.append(context_step.numpy())
    emb = np.array(emb)
    emb = np.mean(emb, axis=0)[0]

    return emb


def create_complementary_dataset(dataset, idxs, trajectory_length=10):
    observations = []
    actions = []
    rewards = []
    terminals = []
    for i in range(1000):
        if i not in idxs:
            observations += list(dataset.observations[1000*i:1000*i+trajectory_length])
            actions += list(dataset.actions[1000*i:1000*i+trajectory_length])
            rewards += list(dataset.rewards[1000*i:1000*i+trajectory_length])
            terminals += list(dataset.terminals[1000*i:1000*i+trajectory_length])

    new_dataset = d3rlpy.dataset.MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals)
    )
    return new_dataset
    



def main():
    # args = Parser().parse_args('plan')

    #######################
    ####### models ########
    #######################





    # print(args.dataset)

    # dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
    #         'data_config.pkl')


    # gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
    #         epoch=args.gpt_epoch, device=args.device)

    # env = datasets.load_environment(args.dataset)

    # discretizer = dataset.discretizer

    # preprocess_fn = datasets.get_preprocess_fn(env.name)

    # #######################
    # ####### dataset #######
    # #######################

    # # env = datasets.load_environment(args.dataset)
    # discretizer = dataset.discretizer
    # preprocess_fn = datasets.get_preprocess_fn(env.name)

    # # dataset
    dataset_d3, env = d3rlpy.datasets.get_dataset("halfcheetah-medium-v2")

    # env = gym.make('halfcheetah-medium-v2')
    # dataset_d4 = d4rl.qlearning_dataset(env)

    # # checks to see if d3rl & d4rl datasets are equal
    # print(np.allclose(dataset_d3.actions[100], dataset_d4['actions'][100]))

    # # dr4rl has same trajectories, just cut off 1 element before the end
    # for j in range(1000):
    #     for i in range(999):
    #         if dataset_d4['rewards'][j * 999 + i] != dataset_d3.rewards[j * 1000 + i]: print("yo", i)

    # #######################
    # ###### main loop ######
    # #######################

    trajectory_length = 10 # 10 = max

    # embeddings = []
    # for i in range(1000):
    #     observations = dataset_d3.observations[1000*i:1000*i+trajectory_length]
    #     actions = dataset_d3.actions[1000*i:1000*i+trajectory_length]
    #     rewards = dataset_d3.rewards[1000*i:1000*i+trajectory_length]
    #     terminals = dataset_d3.terminals[1000*i:1000*i+trajectory_length]
    #     emb = embed_trajectory(gpt, discretizer, observations, actions, rewards, preprocess_fn)
    #     embeddings.append(emb)
    # embeddings = np.array(embeddings)
    # np.save("embeddings.npy", embeddings)
    # print(embeddings)

    embeddings = np.load("embeddings.npy")

    pca = PCA(n_components=2)
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)
    np.save("pca.py", pca_embeddings)

    idxs_per_cluster, clusters = cluster_trajectories_2(embeddings)
    # print(clusters)
    # return
    np.save("clusters.npy", clusters)

    import matplotlib.pyplot as plt

    d_orig = generate_data_embedding(embeddings)
    unique_clusters = np.unique(clusters)
    
    d_j = []
    complementary_datasets = []
    for j in np.sort(unique_clusters):
        print(j)
        d_j.append(generate_data_embedding(embeddings[clusters != j]))
        plt.scatter(pca_embeddings[clusters == j][:,0], pca_embeddings[clusters == j][:,1], label=j)
        complementary_datasets.append(create_complementary_dataset(dataset_d3, idxs_per_cluster[j], trajectory_length))
    
    original_dataset = create_complementary_dataset(dataset_d3, [], trajectory_length)

    print(complementary_datasets, original_dataset)

    plt.legend()
    plt.show()

    agent_orig = d3rlpy.algos.SAC(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        batch_size=256)

    print(agent_orig)

    agent_orig.fit(original_dataset, n_steps=10000)

    agents_compl = []

    for dset in complementary_datasets:
        agent = d3rlpy.algos.SAC(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            batch_size=256)
        agent.fit(dset, n_steps=10000)
        agents_compl.append(agent)

    action_orig = agent_orig.predict(dataset_d3.observations[0])

    actions_compl = []
    for agent in agents_compl:
        actions_compl.append(agent.predict(dataset_d3.observations[0]))
    
    action_dists = []
    for action in actions_compl:
        action_dists.append(np.linalg.norm(action_orig-action))

    k = 3
    topk = np.argpartition(action_dists, -k)[-k:]

    d_w = {}
    for idx in topk:
        d_w[idx] = wasserstein_distance(d_j[idx], d_orig)

    cluster_assignment = min(d_w, key=d_w.get)
    print("explanation assigned to cluster", cluster_assignment)

    
def assignment_test():
    action_orig = np.random.rand(10)
    d_orig = np.random.rand(5)

    actions_compl = np.random.rand(6,10)
    d_j = np.random.rand(6,5)

    action_dists = []
    for action in actions_compl:
        action_dists.append(np.linalg.norm(action_orig-action))

    print(action_dists)

    k = 3
    topk = np.argpartition(action_dists, -k)[-k:]

    print(topk)

    d_w = {}
    for idx in topk:
        d_w[idx] = wasserstein_distance(d_j[idx], d_orig)

    print(d_w)

    cluster_assignment = min(d_w, key=d_w.get)
    print("explanation assigned to cluster", cluster_assignment)


if __name__ == "__main__":
    # main()
    assignment_test()