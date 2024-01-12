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

    

