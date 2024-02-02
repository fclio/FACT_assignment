from pyclustering.cluster.kmeans import kmeans, kmeans_observer
from pyclustering.cluster.rock import rock
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.agglomerative import agglomerative, type_link
from pyclustering.cluster.optics import optics, ordering_analyser
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster.cure import cure
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

def xmean(traj_embeddings):
    # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # start the analysis.
    amount_initial_centers = 2
    initial_centers = kmeans_plusplus_initializer(traj_embeddings, amount_initial_centers).initialize()

    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    # number of clusters that can be allocated is 10.
    xmeans_instance = xmeans(traj_embeddings, initial_centers, 10)
    xmeans_instance.process()

    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()

    # Print total sum of metric errors
    # print("Total WCE:", xmeans_instance.get_total_wce())
    print("X-means Clusters:", clusters)
    print('Number of clusters', len(clusters))

    return clusters

def kmean(traj_embeddings):
    # Prepare initial centers - amount of initial centers defines the amount of clusters for K-Means.
    amount_initial_centers = 10
    initial_centers = kmeans_plusplus_initializer(traj_embeddings, amount_initial_centers).initialize()

    # Create instance of K-Means algorithm.
    # num_clusters = 10
    kmeans_instance = kmeans(traj_embeddings, initial_centers)

    # Process the K-Means algorithm.
    kmeans_instance.process()

    # Extract clustering results: clusters and their centers
    clusters_k = kmeans_instance.get_clusters()
    centers_k = kmeans_instance.get_centers()

    # Print total sum of metric errors (WCE - within-cluster sum of squared errors)
    # print("Total WCE:", kmeans_instance.get_total_wce())
    # print("Clusters:", clusters_k)
    # print('Number of clusters:', len(clusters_k))
    print("kmean")

    return clusters_k

def dbscans(data):
    # Set DBSCAN parameters
    epsilon = 1.0  # neighborhood radius
    minpts = 5     # minimum number of points in a neighborhood to be considered as a core point

    # Create an instance of DBSCAN
    dbscan_instance = dbscan(data, epsilon, minpts)

    # Process the DBSCAN algorithm
    dbscan_instance.process()

    # Get clustering results: clusters and noise points
    clusters = dbscan_instance.get_clusters()
    noise = dbscan_instance.get_noise()

    # Print the results
    # print("Clusters:", clusters)
    # print("Noise points:", noise)
    # print('Number of clusters:', len(clusters))
    print("dbscans")

    return clusters

def agg(data_points):

    num_clusters = 10
    agg_instance = agglomerative(data_points, num_clusters, type_link.SINGLE_LINK)

    # Perform hierarchical clustering
    agg_instance.process()

    # Extract clustering results: clusters and their centers
    clusters_agg = agg_instance.get_clusters()


    # Print clustering results
    # print("Clusters:", clusters_agg)
    # print('Number of clusters:', len(clusters_agg))
    print("agg")
    return clusters_agg

def optic(data):

    # Create instance of OPTICS algorithm.
    optics_instance = optics(data,eps=1.0, minpts=5)

    # Run OPTICS algorithm.
    optics_instance.process()

    # Get clustering results: clusters and their medoids.
    clusters = optics_instance.get_clusters()

    # Print the results.
    # print("Clusters:", clusters)

    print("optic")
    return clusters

def k_median(data):
    amount_initial_centers = 10
    initial_centers = kmeans_plusplus_initializer(data, amount_initial_centers).initialize()

    # Create instance of K-Medians algorithm.
    kmedians_instance = kmedians(data,initial_centers)

    # Process the K-Medians algorithm.
    kmedians_instance.process()

    # Extract clustering results: clusters and their centers
    clusters_kmedians = kmedians_instance.get_clusters()

    # Print results
    # print("Clusters:", clusters_kmedians)
    # print('Number of clusters:', len(clusters_kmedians))
    print("k_median")
    return clusters_kmedians

def claran(data):

    # Number of clusters you want to find.
    num_clusters = 10

    # Number of local minima that should be generated.
    num_local_minima = 5

    # Maximum number of neighbors examined.
    max_neighbors = 4

    # Create CLARANS instance.
    clarans_instance = clarans(data=data, number_clusters=num_clusters, numlocal=num_local_minima, maxneighbor=max_neighbors)

    # Run CLARANS algorithm.
    clarans_instance.process()

    # Get clustering results.
    clusters = clarans_instance.get_clusters()
    medoids = clarans_instance.get_medoids()

    # Print results.
    # print("Clusters:", clusters)
    # print("Medoids:", medoids)
    print("claran")
    return clusters

def cures(data):
    # Create CURE instance.
    cure_instance = cure(data=data, ccore=True, number_cluster=10)

    # Process CURE algorithm.
    cure_instance.process()

    # Get clustering results.
    clusters = cure_instance.get_clusters()


    # Print results.
    # print("Clusters:", clusters)
    print("cures")

    return clusters

def rocks(data):
    # Set ROCK parameters
    radius = 1.0  # Link radius
    neighborhood = 3  # Minimum number of neighbors

    # Create instance of ROCK algorithm
    rock_instance = rock(data, radius, neighborhood)

    # Run the algorithm
    rock_instance.process()

    # Get clustering results
    clusters = rock_instance.get_clusters()
    print("rock")
    return clusters

def flat_cluster(o_labels):
    # Create a mapping dictionary for true labels
    mapping_dict = {index: label for label, cluster in enumerate(o_labels) for index in cluster}

    # Map true labels to the desired format
    mapped_labels = [mapping_dict.get(index, 0) for index in range(max(mapping_dict, default=-1) + 1)]


    return mapped_labels

def compare(true_cluster, alternative_cluster):
    # Flatten the true labels and predicted labels to get a list of individual labels
    true_labels  = true_cluster
    predicted_labels = alternative_cluster

    # Calculate the Adjusted Rand Index
    ari = adjusted_rand_score(true_labels, predicted_labels)
    print("Adjusted Rand Index (SOM):", ari)

    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    print("NMI:", nmi)
    return ari, nmi

def evaluate_clusters(data):
    clusters = [kmean, dbscans,agg,optic, k_median,claran,cures,rocks]
    true_cluster = flat_cluster(xmean(data))

    for cluster in clusters:
        compare_cluster = flat_cluster(cluster(data))
        print(compare_cluster)
        compare(true_cluster,compare_cluster)


