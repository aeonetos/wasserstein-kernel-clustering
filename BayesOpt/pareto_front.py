import numpy as np
import pandas as pd
import math
import pickle
from functools import partial
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.preprocessing import KernelCenterer
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import validity_measures as vm
import clustering_opt as co

"""
A Pareto-optimal solution set is such that:

- Given a set of solutions, is a set of all the solutions that are not
dominated by any member of the solution set
- The boundary defined by the set of all point mapped
from the Pareto optimal set is called the Pareto-optimal front

Dominance Test: Given x1 and x2
x1 dominates x2, if
- Solution x1 is no worse than x2 in all objectives.
- Solution x1 is strictly better than x2 in at least one objective
"""

def pareto_front(solution_set, direction=None):
    """
    This function returns the Pareto-optimal front given a solution set and the direction of the objectives
    :param solution_set: the solution set
    :param direction: the direction of the objectives. If None, the direction is positive for all objectives, i.e., the objectives are to be maximized. If the direction is negative for an objective, the objective is to be minimized.
    :return: the Pareto-optimal front
    """
    if direction is None:
        direction = np.ones(solution_set.shape[1])
    else:
        solution_set = np.multiply(solution_set, direction)
    # we will use the Pareto dominance test to find the Pareto-optimal solutions
    non_dominated_set = []
    non_dominated_set_index = []
    for j in range(solution_set.shape[0]):
        for i in range(solution_set.shape[0]):
            # we will check if the solution i dominates the solution j
            if (solution_set[i] >= solution_set[j]).all() and (solution_set[i] > solution_set[j]).any():
                break
            if i == solution_set.shape[0] - 1 and j!=i:
                non_dominated_set.append(solution_set[j])
                non_dominated_set_index.append(j)
    non_dominated_set = np.array(non_dominated_set)
    non_dominated_set = np.multiply(non_dominated_set, direction)
    return non_dominated_set, non_dominated_set_index

def N_KMedoids(X, clusters, n_init=6):
    """
    This function runs the KMedoids algorithm n_init times and returns the best clustering solution according to the Fast Goodman-Kruskal index
    :param X: the dataset
    :param clusters: the number of clusters
    :param n_init: the number of times to run the KMedoids algorithm
    :return: the best clustering solution according to the Fast Goodman-Kruskal index
    """
    cluster_model, best_fgk = None, -1 # we initialize the cluster model to return
    # we run the KMedoids algorithm n_init times
    for _ in range(n_init):
        # we create an instance of KMedoids with the number of clusters
        kmedoids = KMedoids(n_clusters=clusters, init='k-medoids++', method='pam').fit(X)
        # we check the current davies-bouldin score
        if np.unique(kmedoids.labels_).shape[0] == clusters:
            # Fast Goodman-Kruskal index
            fgk = vm.FGK(n=60, m=35)
            fgk.fit(X, kmedoids.labels_)
            if fgk.GK_ > best_fgk:
                best_fgk = fgk.GK_
                cluster_model = kmedoids
    return cluster_model

def pareto_clusters(demand_disimilarity, node_disimilarity, wasserstein_distances, pareto_hyp):
    """
    This function computes the kernel clustering solution for a given set of hyperparameters
    :param demand_disimilarity: the disimilarity matrix of the demand
    :param length_disimilarity: the disimilarity matrix of the length
    :param wasserstein_distances: the Wasserstein distance matrix
    :param pareto_hyp: the hyperparameters of the kernel clustering
    :return: the number of components of the PCA, the explained variance of the PCA, the eigenvalues of the PCA, and the clustering model
    """
    gamma_wasserstein, gamma_demand, gamma_nodes = pareto_hyp['gamma_wasserstein'], pareto_hyp['gamma_demand'], pareto_hyp['gamma_nodes']
    # we apply a minimum-shift to the Wassestein distance matrix to yield a analogous Euclidean distance matrix
    # note that the disimilarity of the demand is already a Euclidean distance matrix
    sqr_wasserstein_distance = np.square(wasserstein_distances)
    Dc = sqr_wasserstein_distance - sqr_wasserstein_distance.mean(axis=0)
    Dc = Dc - Dc.mean(axis=1).reshape(-1,1)
    Sc = -0.5 * Dc
    # we obtain the minimum shift to make the matrix positive semidefinite and compute the euclidean distance matrix in the feature space with the shift
    eigenvalues = np.linalg.eigvalsh(Sc)
    if eigenvalues[0] < 0:
        perturbation = np.full(Sc.shape, - 2 * eigenvalues[0])
        np.fill_diagonal(perturbation, 0)
        euclidean_wasserstein = sqr_wasserstein_distance + perturbation
    else:
        euclidean_wasserstein = sqr_wasserstein_distance
    # we compute the composed kernel
    kernel = np.multiply(np.exp(- math.pow(10, gamma_wasserstein) * euclidean_wasserstein), 
                         np.exp(- math.pow(10, gamma_demand) * np.square(demand_disimilarity)) \
                             + np.exp(- math.pow(10, gamma_nodes) * np.square(node_disimilarity)))
    # Create an instance of KernelCenterer
    centerer = KernelCenterer()
    # Fit and transform the kernel matrix
    normalized_kernel = centerer.fit_transform(kernel)
    # we compute the PCA of the normalized kernel
    pca = PCA()
    # Fit the PCA model to your data
    pca.fit(normalized_kernel)
    # Access the principal components (eigenvectors)
    components = pca.components_
    # Access the singular values (eigenvalues)
    singular_values = pca.singular_values_
    # we compute the whitening matrix
    L_m12 = np.diag(np.power(np.sqrt(singular_values), -1))
    U = np.dot(components.T, L_m12)
    # we compute the whitened kernel
    W = np.dot(normalized_kernel, U).T
    # we obtain the feature vectors in the whitened space
    Y = W.T
    n_components = np.where(pca.explained_variance_ratio_ > 1/Y.shape[1])[0][-1] + 1
    # we add the explained variance to the list of explained variance for the pareto front
    explained_var = pca.explained_variance_ratio_[:n_components].sum()
    # we add the eigenvalues to the list of eigenvalues for the pareto front
    eigenvalues = pca.singular_values_
    # we want to keep the n_components number of components of the PCA
    pca = PCA()
    pca.fit(Y)
    pca.components_ = pca.components_[:n_components]
    # we transform the data
    Y_reduced = pca.transform(Y)
    # we cluster the data
    cluster_model = N_KMedoids(Y_reduced, pareto_hyp['K'], n_init=10)
    return n_components, explained_var, eigenvalues, cluster_model

def pareto_consensus(pareto_cluster_models):
    """
    This function computes the consensus matrix for the pareto front
    :param pareto_cluster_models: the clustering models of the pareto front
    :return: the consensus matrix for the pareto front
    """
    # We compute the consensus matrix for the pareto front, [[CI(P0, P0), CI(P0,P1), ... , CI(P0, PN)], ..., [CI(PN, P0), CI(PN,P1), ... , CI(PN, PN)]] where P0, ..., PN are the Pareto optimal solutions
    pareto_consensus_matrix = np.zeros((len(pareto_cluster_models), len(pareto_cluster_models)))
    for i in range(len(pareto_cluster_models)):
        for j in range(i, len(pareto_cluster_models)):
            Y = np.array([pareto_cluster_models[i].labels_, pareto_cluster_models[j].labels_])
            pareto_consensus_matrix[i,j] = vm.consensus_index(Y)
            pareto_consensus_matrix[j,i] = pareto_consensus_matrix[i,j]
    return pd.DataFrame(pareto_consensus_matrix, columns=list(range(len(pareto_cluster_models))), index=list(range(len(pareto_cluster_models))))

def parallel_pareto_clusters(demand_disimilarity, node_disimilarity, wasserstein_distances, pareto_df, cpu):
    """
    This function computes the clustering solutions for the pareto front in parallel
    :param demand_disimilarity: the disimilarity matrix of the demand
    :param length_disimilarity: the disimilarity matrix of the length
    :param wasserstein_distances: the Wasserstein distance matrix
    :param pareto_df: the hyperparameters of the kernel clustering for the pareto front
    :param cpu: the number of cores to use
    :return: the number of components of the kernel PCA, the explained variance of the kernel PCA, the eigenvalues of the kernel PCA, and the clustering model
    """
    partial_pareto_clusters = partial(pareto_clusters, demand_disimilarity, node_disimilarity, wasserstein_distances)
    jobs = [pareto_df.iloc[p] for p in range(pareto_df.shape[0])]    
    
    with Parallel(n_jobs=cpu) as parallel:
        results = parallel(delayed(partial_pareto_clusters)(job) for job in jobs)
    n_components, explained_var, eigenvalues, cluster_models = zip(*results)
    return n_components, explained_var, eigenvalues, cluster_models

if __name__ == '__main__':
    # we read the embeddings
    with open('data/node_embeddings.pkl', 'rb') as handle:
        node_embeddings = pickle.load(handle)
    # we read the overall line length of the grids
    with open('data/length_grids.pkl', 'rb') as handle:
        length_grids = pickle.load(handle)
    # we read the wasserstein model
    with open("data/wasserstein_grids.pickle", "rb") as file:
        wass_grids = pickle.load(file)

    # we get the distances between the embeddings
    grid_ids = node_embeddings['grid'].unique()
    # we stack the node embeddings in the list X
    X = [node_embeddings[node_embeddings['grid']==grid_id].iloc[:,:-1].values for grid_id in grid_ids]

    # we do the same for the node disimilarity
    N = [x.shape[0] for x in X]
    # we compute the disimilarity matrices
    demand_disimilarity = co.penalty_matrix(X, penalize_entry=2)

    node_disimilarity = co.penalty_matrix(N)

    # we get the distances between the embeddings
    wasserstein_distances = wass_grids.min_distances_

    # we read the hyperparameters and the validity indices of the pareto front
    pareto_df = pd.read_csv('data/pareto_front.csv')

    cpu = cpu_count() - 8 # the cores used for the optimization
    pareto_n_components, pareto_explained_var, pareto_eigenvalues, pareto_cluster_models = parallel_pareto_clusters(demand_disimilarity, node_disimilarity, wasserstein_distances, pareto_df, cpu)

    pareto_consensus_df = pareto_consensus(pareto_cluster_models)