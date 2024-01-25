#!/usr/bin/env python3

# from pyclustering.cluster.xmeans import xmeans
import numpy as np

# https://github.com/KazuhisaFujita/X-means/blob/master/simple_xmeans.py
import numpy as np
import math as mt
import sys
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics

from copy import deepcopy

import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from env import idx_to_coords

from env import DynaQAgent, Gridworld

  
def generate_trajectories(num_episodes_dyna=6, num_agents_dyna=5, num_pos_trajs=10, num_neg_trajs=10, max_traj_len=15, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    env = Gridworld()
    agents = []
    for _ in range(num_agents_dyna):
        agent = DynaQAgent(env.state_size(), env.action_size(), 0.1, 0.95)
        agent.train(env, num_episodes_dyna, 5)
        agents.append(agent)
    
    pos_traj = []
    neg_traj = []
    
    while len(pos_traj) < num_pos_trajs or len(neg_traj) < num_neg_trajs:
        agent = np.random.choice(agents)
        traj, r = agent.gen_traj(env, max_traj_len=max_traj_len)
        if r == 0:
            continue
        elif r == 1 and len(pos_traj) < num_pos_trajs:
            pos_traj.append(traj)
        elif r == -1 and len(neg_traj) < num_neg_trajs:
            neg_traj.append(traj)
    return pos_traj, neg_traj

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
    xmeans_instance = XMeans(trajectories)
    xmeans_instance.fit()

    clusters = xmeans_instance.labels
    return clusters

# https://github.com/sascha-kirch/ML_Notebooks/blob/main/Softmax_Temperature.ipynb
def softmax(x, temp):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(np.divide(x,temp)) / np.sum(np.exp(np.divide(x,temp)))

def generate_data_embedding(trajectory_embeddings, normalizing_factor=1, temperature=1):
    embedding = np.sum(trajectory_embeddings, axis=0) / normalizing_factor
    embedding = softmax(embedding, temperature)
    return embedding

def generate_model(env, trajs):
    len_act_space = len(env.action_space)

    # Initialize the transitions
    transition_model = np.zeros(
        (np.prod(env.dim), len_act_space, np.prod(env.dim)))
    reward_model = np.zeros(
        (np.prod(env.dim), len_act_space, np.prod(env.dim)))

    for traj in trajs:
        for (s, a, r, s_) in traj:
            transition_model[s, a, s_] += 1
            reward_model[s, a, s_] += r

    reward_model = reward_model / (transition_model.sum(2)[:, :, None] + 1e-9)
    transition_model = transition_model / \
        (transition_model.sum(2)[:, :, None] + 1e-9)

    # Note that this model has non-zero probability for picking an action in any state

    return transition_model, reward_model


def dp(env, transition_model, reward_model, gamma=0.95, threshold=1e-4):
    values = np.random.randn(np.prod(env.dim))

    while True:
        delta = 0.
        for s in range(values.shape[0]):
            v = values[s]

            values[s] = np.max(list(
                # map(lambda a: (reward_model[s, a] + gamma * transition_model[s, a] * values).sum(), env.action_space)))
                map(lambda a: ((reward_model[s, a] + gamma * values) * transition_model[s, a]).sum(), env.action_space)))

            delta = max(delta, abs(v - values[s]))
        if delta < threshold:
            break

    # Action values
    action_values = np.zeros((np.prod(env.dim), len(env.action_space)))
    for s in range(action_values.shape[0]):
        action_values[s, :] = list(
            # map(lambda a: (reward_model[s, a] + gamma * transition_model[s, a] * values).sum(), env.action_space)))
            map(lambda a: ((reward_model[s, a] + gamma * values) * transition_model[s, a]).sum(), env.action_space))

    # Policy from value
    policy = np.ones(np.prod(env.dim), dtype=int)
    for s in range(values.shape[0]):
        policy[s] = np.argmax(list(
            # map(lambda a: (reward_model[s, a] + gamma * transition_model[s, a] * values).sum(), env.action_space)))
            map(lambda a: ((reward_model[s, a] + gamma * values) * transition_model[s, a]).sum(), env.action_space)))
    return values, action_values, policy


def calc_distance(transition_model1, transition_model2):
    return wasserstein_distance(transition_model1.reshape(-1), transition_model2.reshape(-1))


def take_policy_difference(policy1, policy2):
    return np.array(policy1 != policy2, dtype=int)


def take_traj_difference(traj_idx_list1, traj_idx_list2):
    uniq_idx1, count1 = np.unique(traj_idx_list1, return_counts=True)
    uniq_idx2, count2 = np.unique(traj_idx_list2, return_counts=True)

    count = np.zeros_like(traj_idx_list1)
    count[uniq_idx1] += count1
    count[uniq_idx2] -= count2

    return count


def find_relevant_traj(env, policy, boot_policy_list, offline_data, boot_traj_list, num_pos_trajs=2, num_neg_trajs=2,
                       action_dict={0: 'LEFT', 1: 'UP', 2: 'RIGHT', 3: 'DOWN'}):
    for state_idx in np.arange(np.prod(env.dim)):
        coords = idx_to_coords(state_idx, grid_dim=env.dim)
        print(coords)
        traj_diff_list = []
        for boot_idx, boot_policy in enumerate(boot_policy_list):
            if policy[coords] != boot_policy[coords]:
                traj_diff_list.append(take_traj_difference(
                    np.arange(len(offline_data)), boot_traj_list[boot_idx]))

        traj_attribution = (np.array(traj_diff_list) != 0).sum(axis=0)

        print('State:', state_idx)
        print('Original Action:', action_dict[policy[coords]])
        print('Trajectory Attribution', traj_attribution)

    return


def generate_offline_data(env, agent, num_episodes_dyna=6, num_agents_dyna=5, max_traj_len=15, num_pos_trajs=10, num_neg_trajs=10):
    # Partially training agents to generate trajectories

    agents = []

    print("Training", num_agents_dyna, "Agents")
    for i in range(num_agents_dyna):
        agents.append(deepcopy(agent))

    running_averages = [[0] * num_episodes_dyna] * num_agents_dyna
    episode_lengths = [0] * num_agents_dyna

    for i in tqdm(range(num_agents_dyna)):
        running_averages[i], episode_lengths[i] = agents[i].train(
            num_episodes_dyna, env, 0.1, 0.95, 5, render=False)

    # Generate trajectories using set of partially trained agents

    pos_trajs = []
    neg_trajs = []

    while len(pos_trajs) < num_pos_trajs or len(neg_trajs) < num_neg_trajs:
        sampled_agent = np.random.randint(num_agents_dyna)
        traj, r = agents[sampled_agent].perform(env, max_traj_len)

        if r == 0:
            continue  # Discarding zero reward trajectories

        elif r == 1 and len(pos_trajs) < num_pos_trajs:
            pos_trajs.append(traj)
            print(f'# POS TRAJs {len(pos_trajs)}')
        elif r == -1 and len(neg_trajs) < num_neg_trajs:
            neg_trajs.append(traj)
            print(f'# NEG TRAJs {len(neg_trajs)}')

    print("----------------------------")
    print('Number of trajectories generated:')
    print('+1 trajs:', len(pos_trajs))
    print('-1 trajs:', len(neg_trajs))

    print("----------------------------")

    pos_trajs.sort(key=lambda x: len(x))
    neg_trajs.sort(key=lambda x: -len(x))
    offline_data = pos_trajs + neg_trajs

    return offline_data


def bootstrap(env, offline_data, num_bootstraps):
    boot_traj_list = []
    transition_model_list = []
    reward_model_list = []
    boot_value_list = []
    boot_action_value_list = []
    boot_policy_list = []
    for _ in range(num_bootstraps):
        traj_indices = np.sort(np.random.randint(
            len(offline_data), size=len(offline_data)))
        boot_traj_list.append(traj_indices)

        bootstrap_offline_data = [offline_data[idx] for idx in traj_indices]
        boot_transition_model, boot_reward_model = generate_model(
            env, bootstrap_offline_data)
        boot_values, boot_action_values, boot_policy = dp(
            env, boot_transition_model, boot_reward_model)
        boot_values = boot_values.reshape(env.dim)
        boot_action_values = boot_action_values.reshape(
            *(env.dim), len(env.action_space))
        boot_policy = boot_policy.reshape(env.dim)

        transition_model_list.append(boot_transition_model)
        reward_model_list.append(boot_reward_model)
        boot_value_list.append(boot_values)
        boot_action_value_list.append(boot_action_values)
        boot_policy_list.append(boot_policy)

    return boot_traj_list, transition_model_list, reward_model_list, boot_value_list, boot_action_value_list, boot_policy_list


if __name__ == "__main__":
    # CONVENTION: 1 ROW = 1 TRAJECTORY

    # 3 clusters of embeddings
    trajectory_embeddings = list(np.random.normal(size=50,loc=3,scale=1).reshape(5,10))
    trajectory_embeddings += list(np.random.normal(size=50,loc=-6,scale=1).reshape(5,10))
    trajectory_embeddings += list(np.random.normal(size=50,loc=-50,scale=1).reshape(5,10))
    trajectory_embeddings = np.array(trajectory_embeddings)

    clusters = cluster_trajectories(trajectory_embeddings)

    d_orig = generate_data_embedding(trajectory_embeddings)

    d_j = []
    unique_clusters = np.unique(clusters)
    for j in unique_clusters:
        print(trajectory_embeddings[clusters != j])
        d_j.append(trajectory_embeddings[clusters != j])

    

