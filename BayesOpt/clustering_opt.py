"""
This file contains the clustering optimization algorithm.
The main ideas are:
- We randomly generate weights for producing convex combinations of the kernels.
- We test different values of gamma in the RBF kernel.
- We use Bayesian optimization to tune the gammas for each kernel, given a combination of weights. The objective function is the clustering score.
- To avoid trivial clustering results, we fix a number of clusters, e.g., K = 8.
- The process can then be repeted for different values of K, which yields a Pareto front of solutions, given the scores and the Ks.
"""
# Add extra paths
import sys
import os
sys.path.insert(0, os.path.abspath(''))
# Import libraries
from sklearn_extra.cluster import KMedoids
from validity_measures import FGK, consensus_index
import numpy as np
import pickle
import pandas as pd
import math
import time
from functools import partial
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.decomposition import PCA
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
# we import the class WK_PCA from the module wasserstein_learn, which allows us to perform Wasserstein kernel PCA
from wasserstein_learn import WK_PCA
# we importthe class Wasserstein from the module wassesrstein_learn, which allows us to compute the Wasserstein distance
from wasserstein_learn import Wasserstein
# we import the bayesian optimization module and functions
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

def N_KMedoids(X, clusters, n_init=5):
    """
    This function computes the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a KMedoids clustering solution
    :param X: the dataset
    :param clusters: the number of clusters
    :param n_init: the number of times the KMedoids algorithm is run
    :return: the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a clustering solution
    """
    L, min_ec, fgk_list = [], clusters, [] # we initialize the labels to an empty list and the effective number of components
    # we run the KMedoids algorithm n_init times
    for _ in range(n_init):
        # we create an instance of KMedoids with the number of clusters
        kmedoids = KMedoids(n_clusters=clusters, init='k-medoids++', method='pam').fit(X)
        # we check the current davies-bouldin score
        if np.unique(kmedoids.labels_).shape[0] == clusters:
            # We get the labels of the clusters
            L.append(kmedoids.labels_)
            # we count the effective number of components
            count_labels = np.unique(L[-1], return_counts=True)[1]
            plabels = count_labels/np.sum(count_labels)
            ec = 1/np.sum(plabels**2)
            if min_ec > ec:
                min_ec = ec
            # Fast Goodman-Kruskal index
            fgk = FGK(n=60, m=35)
            fgk.fit(X, L[-1])
            fgk_list.append(fgk.GK_)
            # if the fgk index is below 0.6, we stop the for loop. This avoids the execution of the algorithm for unuseful clusterings
            if fgk.GK_ < 0.6:
                L = [] # we reset the labels so the algorithm does not compute the consensus index and the average of the fast goodman-kruskal indices
                break
    # we compute the consensus index
    if len(L) > 0:
        # we transform the labels into a numpy array
        L = np.array(L)
        CI = consensus_index(L)
        # we compute the average of the fast goodman-kruskal indices
        FGK_index = np.mean(fgk_list)
    else:
        CI, min_ec, FGK_index = 0, 0, -1
    return CI, min_ec, FGK_index

def composed_kernel_clustering(K, disimilarity_demand, node_disimilarity, wasserstein_distance, gamma_wasserstein, gamma_demand, gamma_nodes):
    """
    This function computes the composed kernel and performs KMedoids clustering to get the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a clustering solution
    :param K: the number of clusters
    :param disimilarity_demand: the disimilarity matrix of the demand
    :param length_disimilarity: the disimilarity matrix of the overall line length
    :param wasserstein_distance: the wasserstein distance matrix
    :param gamma_wasserstein: the gamma parameter of the RBF kernel for the wasserstein distance
    :param gamma_demand: the gamma parameter of the RBF kernel for the demand disimilarity
    :param gamma_length: the gamma parameter of the RBF kernel for the length disimilarity
    :return: the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a clustering solution
    """
    # we apply a minimum-shift to the Wassestein distance matrix to yield a analogous Euclidean distance matrix
    # note that the disimilarity of the demand and the disimilarity of the overall line lenght are already Euclidean distance matrices
    sqr_wasserstein_distance = np.square(wasserstein_distance)
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

    kernel = np.multiply(np.exp(- math.pow(10, gamma_wasserstein) * euclidean_wasserstein), 
                         np.exp(- math.pow(10, gamma_demand) * np.square(disimilarity_demand)) \
                                     + np.exp(- math.pow(10, gamma_nodes) * np.square(node_disimilarity)))
    # Create an instance of PCA with the desired number of components
    pca = PCA()
    # Create an instance of KernelCenterer
    centerer = KernelCenterer()
    # Fit and transform the kernel matrix
    normalized_kernel = centerer.fit_transform(kernel)
    # we get the upper triangular part of the kernel matrix, excluding the diagonal. We keep the values in an array
    kernel_upper = normalized_kernel[np.triu_indices(normalized_kernel.shape[0], k = 1)]
    dispersion = np.var(kernel_upper)/(np.abs(np.mean(kernel_upper)) + 1e-15)
    # we discard the kernel if the dispersion is too low
    if dispersion < math.pow(10,-6):
        CI, min_ec, FGK_index = 0, 0, -1  
    else:
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
        # we want to keep the n_components number of components of the PCA
        if n_components > 2:
            pca = PCA()
            pca.fit(Y)
            pca.components_ = pca.components_[:n_components]
            # we transform the data
            Y_reduced = pca.transform(Y)
            # we create an instance of KMedoids with the number of clusters
            CI, min_ec, FGK_index = N_KMedoids(Y_reduced, K)
        else:
            CI, min_ec, FGK_index = 0, 0, -1 
    return CI, min_ec, FGK_index

def penalty_matrix(X, penalize_entry=None):
    if penalize_entry is None:
        weight_penalty = X
    else:
        weight_penalty = [np.sum(X[i][:,penalize_entry]) for i in range(len(X))]
    weight_matrix = np.array([weight_penalty for _ in range(len(weight_penalty))])
    # we then compute the pairwise weight distance
    disimilarity_matrix = np.abs(weight_matrix - weight_matrix.T)
    return disimilarity_matrix      

def get_solutions(results):
    """
    This function takes in the results from the optimizer and returns them as a dataframe.
    """
    # we get the variables and the target value from the first row of the results and store them in a dictionary
    variables = results[0]['params'].keys()
    container = {variable: [results[0]['params'][variable]] for variable in variables}
    container['target'] = [results[0]['target']]
    # we add the variables and target value from the other rows one by one to the dictionary
    for result in results[1:]:
        for variable in variables:
            container[variable].append(result['params'][variable])
        container['target'].append(result['target'])
    # we return the results as a dataframe
    results_df = pd.DataFrame.from_dict(container)
    return results_df

def bayes_optimizer(black_box_fun, utility_fun, opt_obj, n_iter, n_init=50):
    """
    This function takes in a black box function, an utility function, an optimizer, and the number of iterations.
    If the number of initial points is specified, we generate the initial points and register them with the optimizer.
    Then, we add points one by one, using the utility function to select the next one.
    Finally, we return the optimizer and the results as a dataframe.
    """
    # we first generate some initial points
    init_variables, init_targets, CI_index, EC, FGK_index, iter_type = [], [], [], [], [], []
    # we print the initial points
    for j in range(n_init):
        next_point = opt_obj.suggest(utility_fun)
        ci, min_ec, fgk = black_box_fun(**next_point)
        CI_index.append(ci)
        EC.append(min_ec)
        FGK_index.append(fgk)
        init_variables.append(next_point)
        # as CI values range between 0 and 1, we normalize the FGK values to the same range
        # we set a Rawlsian utility function, which is the minimum of the two values
        target = np.min([ci, (fgk+1)*0.5])
        init_targets.append(target)
        iter_type.append('random')
        print("\nInitial iteration: ", j + 1)
        print('Point', next_point)
        print('Target', round(target,4))
        print("Coincidence index: ", round(ci,4), "Fast Goodman-Kruskal index: ", round(fgk,4))
    # we register the initial points with the target values to the optimizer
    for i in range(n_init):
        opt_obj.register(params=init_variables[i], target=init_targets[i])
    # we add points one by one, using the utility function to select the next one
    # we print the iteration points
    for j in range(n_iter):
        next_point = opt_obj.suggest(utility_fun)
        ci, min_ec, fgk = black_box_fun(**next_point)
        EC.append(min_ec)
        FGK_index.append(fgk)
        CI_index.append(ci)
        target = np.min([ci, (fgk+1)*0.5])
        opt_obj.register(params=next_point, target=target)
        iter_type.append('bayes')
        print("\nBayes iteration: ", j + 1)
        print('Point', next_point)
        print('Target', round(target,4))
        print("Coincidence index: ", round(ci,4), "Fast Goodman-Kruskal index: ", round(fgk,4))
    # we get the results as a dataframe and sort them according to the target value
    results = get_solutions(opt_obj.res)
    results['Effective components'] = EC
    results['FGK'] = FGK_index
    results['CI'] = CI_index
    results['Iteration type'] = iter_type
    results.sort_values('target', ascending=False, inplace=True)
    results.reset_index(drop=True, inplace=True)
    return opt_obj, results

def bayes_results(demand_disimilarity, node_disimilarity, wasserstein_distances, hyp_ranges, bayes_iter, bayes_init, base_folder, K):
    """
    This function takes in the results from the optimizer and returns the best result as a dictionary.
    """
    # we define the black box function
    black_box_function = partial(composed_kernel_clustering, K, demand_disimilarity, node_disimilarity, wasserstein_distances)
    # we define the optimizer
    optimizer = BayesianOptimization(f=None, pbounds=hyp_ranges, verbose=2, random_state=1, allow_duplicate_points=True)
    # we use the expected improvement acquisition function
    utility_function = UtilityFunction(kind="ei", xi=0.0)
    # we run the optimizer
    optimizer, results = bayes_optimizer(black_box_function, utility_function, optimizer, n_iter=bayes_iter, n_init=bayes_init)
    results.to_csv(base_folder + 'K{}_results.csv'.format(K), index=False)  
    return

def optimize_kernels(parall_bayes, cpu_count, Ks):
    with Parallel(n_jobs=cpu_count) as parallel:
        parallel(delayed(parall_bayes)(job) for job in Ks)
    return

if __name__ == '__main__':

    grid_type = 'MV'

    # we read the embeddings
    node_embeddings = pd.read_csv('data/node_embeddings_{}.csv'.format(grid_type), index_col=False)
    # we read the overall line length of the grids
    length_grids = pd.read_csv('data/length_grids_{}.csv'.format(grid_type), index_col=False)
    # we read the wasserstein model
    with open("data/wasserstein_grids_{}.pickle".format(grid_type), "rb") as file:
        wass_grids = pickle.load(file)

    cpuc = cpu_count() - 2 # the cores used for the optimization
    print("Cumputing core count: ", cpuc)
    # we get the distances between the embeddings
    grid_ids = node_embeddings['grid'].unique()
    # we stack the node embeddings in the list X
    X = [node_embeddings[node_embeddings['grid']==grid_id].iloc[:,:-1].values for grid_id in grid_ids]
    # we get the number of nodes in each grid
    N = [x.shape[0] for x in X]

    # we compute the disimilarity matrices
    demand_disimilarity = penalty_matrix(X, penalize_entry=0)
    node_disimilarity = penalty_matrix(N)
    # we get the wasserstein distances from the model
    wasserstein_distances = wass_grids.min_distances_

    # gamma_wasserstein is required to be close to the argument that maximizes the dispersion of the kernel matrix. The demand kernel and the length kernel can be more flexible, as we use it to penalize the composed kernel.
    # This ranges are obtained form the robust dispersion of the kernel matrices.
    # hyp_ranges_composed = {'gamma_wasserstein': (0.5, 1.5), 'gamma_demand': (-2, -1), 'gamma_nodes': (-3, -2)} # gamma is in scientific notation
    eps = 0.5
    hyp_ranges_composed = {'gamma_wasserstein': (1.25 - eps, 1.25 + eps), 'gamma_demand': (-1.5 - eps, -1.5 + eps), 'gamma_nodes': (-3.25 - eps, -3.25 + eps)} # gamma is in scientific notation
    bayes_iter, bayes_init = 50, 100 # the number of iterations and initial points for the optimizer

    base_folder = 'data/composed_kernel_clustering_{}/'.format(grid_type)

    partial_bayes_results = partial(bayes_results, demand_disimilarity, node_disimilarity, wasserstein_distances, hyp_ranges_composed, bayes_iter, bayes_init, base_folder)
    Ks = list(range(2, 11)) # the number of clusters
        
    start_time = time.time()
    optimize_kernels(partial_bayes_results, cpuc, Ks)
    print("\n--- %s seconds ---" % (time.time() - start_time))

    print("Hihi")