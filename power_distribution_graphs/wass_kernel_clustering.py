"""Clustering of power distribution graphs with Wasserstein-based kernels.

This script assembles distance matrices derived from node embeddings into
kernel functions and applies Kernel PCA followed by K-medoids clustering.
Hyperparameters are optimised through Bayesian optimisation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('BayesOpt/'))

from validity_measures import FGK, consensus_index
import pickle
from functools import partial
import time
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
import gc
import os
from wasserstein_computations import Wasserstein
import math
from scipy import linalg
# we import the bayesian optimization module and functions
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction


def optimal_dispersion_kernel(distance_vector, test_range = (-4, 2, 25)):
    """Heuristic search for informative RBF kernel widths."""
    if distance_vector.ndim == 2:
        distance_vector = squareform(distance_vector)

    gammas = np.linspace(*test_range, dtype=np.float32)
    dispersions = np.zeros(len(gammas), dtype=np.float32)
    # we enumerate the gammas
    for i, gamma in enumerate(gammas):
        ker = np.exp(- math.pow(10, gamma) * np.square(distance_vector), dtype=np.float32)
        dispersion = np.var(ker)
        dispersions[i] = dispersion
    return gammas, dispersions


def center_kernel(kernel):
    """Center a kernel matrix by subtracting row/column means."""
    # We obtain the centered kernel
    centering_vec_kernel = np.mean(kernel, axis=0, keepdims=True)
    centering_mean_kernel = np.mean(kernel)
    return kernel - centering_vec_kernel - centering_vec_kernel.T + centering_mean_kernel

def nystrom_map(kernel, sampling=5000):
    """
    This function performs the Nyström approximation of the input kernel matrix.
    It returns a map phi that can be used to compute the Nyström approximation of the kernel matrix.
    This map phi is such that the Nyström approximation of the kernel matrix is given by phi @ phi.T.
    """
    if kernel.shape[0]<sampling:
        print("The number of samples is greater than the number of rows in the input kernel matrix.")
        print("The number of samples will be set to the number of rows in the input kernel matrix.")
        sampling = kernel.shape[0]
        random_sampling = np.arange(0, kernel.shape[0]) ## Pick all the indices
    else:
        random_sampling = np.random.choice(np.arange(0, kernel.shape[0]), size=sampling, replace=False) ## Pick a random sample of indices
    K_mm = kernel[random_sampling][:,random_sampling].copy() ## Select the corresponding sub-matrix from the input kernel matrix.
    K_mn = kernel[random_sampling].copy() ## Select the corresponding columns from the input kernel matrix.
    K_nm = K_mn.T.copy() ## Compute the transpose of the selected columns from the kernel matrix.
    try:
        U, s_K, _ = linalg.svd(K_mm)
        n_K = kernel.shape[0]
        s_K_tilde = s_K * (n_K/sampling) 
        U_tilde = np.sqrt(sampling/n_K) * np.dot(K_nm, np.dot(U, np.diag(1/s_K)))
        phi_tilde = np.dot(np.diag(np.sqrt(s_K_tilde)), U_tilde.T )
    except:
        phi_tilde = np.zeros((sampling, kernel.shape[0]))
    return phi_tilde.T

def PCA_map(phi_map):
    pca = PCA()
    # Fit the PCA model to your data
    # The map is returned only when the SVD converges
    try:
        pca.fit(phi_map)
        # we compute the number of components to keep
        n_components = np.where(pca.explained_variance_ratio_ > 1/phi_map.shape[1])[0][-1] + 1
        # we want to keep the n_components number of components of the PCA
        if n_components > 2:
            pca.components_ = pca.components_[:n_components]
            Y_reduced = pca.transform(phi_map)
        else:
            # we return a matrix of zeros with the shape of phi_map
            Y_reduced = np.zeros(phi_map.shape)
    except:
        print("SVD failed")
        # we return a matrix of zeros with the shape of phi_map
        Y_reduced = np.zeros(phi_map.shape)
    return Y_reduced

def kernel_PCA(kernel):
    pca = PCA()
    # Fit the PCA model to your data
    try:
        pca.fit(kernel)
        n_components = np.where(pca.explained_variance_ratio_ > 1/kernel.shape[0])[0][-1] + 1
        # we want to keep the n_components number of components of the PCA
        if n_components > 2:
            # Access the principal components (eigenvectors)
            components = pca.components_[:n_components]
            # Access the singular values (eigenvalues)
            singular_values = pca.singular_values_[:n_components]
            # we compute the whitening matrix
            L_m12 = np.diag(np.power(np.sqrt(singular_values), -1))
            U = np.dot(components.T, L_m12)
            # we compute the whitened kernel
            W = np.dot(kernel, U).T
            # we obtain the feature vectors in the whitened space
            Y = W.T  
        else:
            # we return a matrix of zeros with the shape of phi_map
            Y = np.zeros(kernel.shape)
    except:
        print("SVD failed")
        # we return a matrix of zeros with the shape of phi_map
        Y = np.zeros(kernel.shape)
    return Y    

def N_KMedoids(X, clusters, method, n_init=3, fgk_rejection=0.5, n_fgk=100, m_fgk=35, randomstate=42):
    """
    This function computes the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a KMedoids clustering solution
    :param X: the dataset
    :param clusters: the number of clusters
    :param method: 'pam' or 'alternate' method. The 'alternate' method is suggested for larger datasets
    :param n_init: the number of times the KMedoids algorithm is run
    :param fgk_rejection: the threshold for the fast goodman-kruskal index. If the index is below this threshold, the algorithm stops
    :param n_fgk: the number of pairs of points to sample for each cluster in the fast goodman-kruskal index
    :param m_fgk: the number of times to repeat the sampling procedure in the fast goodman-kruskal index
    :return: the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a clustering solution
    """
    L, min_ec, fgk_list = [], clusters, [] # we initialize the labels to an empty list and the effective number of components
    # we run the KMedoids algorithm n_init times
    for state in range(n_init):
        # we create an instance of KMedoids with the number of clusters
        if not method in ['pam', 'alternate']:
            raise ValueError("Invalid method. Choose either 'pam' or 'alternate'.")
        # the random state is varied for each iteration, to obtain different results
        model_cluster = KMedoids(n_clusters=clusters, init='k-medoids++', method=method, random_state=randomstate + state).fit(X)
        # we check the current davies-bouldin score
        if np.unique(model_cluster.labels_).shape[0] == clusters:
            # We get the labels of the clusters
            L.append(model_cluster.labels_)
            # we count the effective number of components
            count_labels = np.unique(L[-1], return_counts=True)[1]
            plabels = count_labels/np.sum(count_labels)
            ec = 1/np.sum(plabels**2)
            if min_ec > ec:
                min_ec = ec
            # Fast Goodman-Kruskal index
            fgk = FGK(n=n_fgk, m=m_fgk)
            fgk.fit(X, L[-1])
            fgk_list.append(fgk.GK_)
            # if the fgk index is below fgk_rejection, we stop the for loop. This avoids the execution of the algorithm for unuseful clusterings
            if fgk.GK_ < fgk_rejection:
                L = [] # we reset the labels so the algorithm does not compute the consensus index and the average of the fast goodman-kruskal indices
                break
    # we compute the consensus index
    if len(L) == n_init:
        # we transform the labels into a numpy array
        L = np.array(L)
        CI = consensus_index(L)
        # we compute the average of the fast goodman-kruskal indices
        FGK_index = np.mean(fgk_list)
    else:
        CI, min_ec, FGK_index = 0, 0, -1
    return CI, min_ec, FGK_index

def composed_kernel_clustering(K_range, disimilarity_demand, node_disimilarity, wasserstein_distance, cluster_method, gamma_wasserstein, gamma_demand, gamma_nodes, kernel_shift=1e-3, nys_n_samples=5000):
    """
    This function computes the composed kernel and performs KMedoids clustering to get the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a clustering solution
    :param K: the number of clusters
    :param disimilarity_demand: the disimilarity matrix of the demand
    :param node_disimilarity: the disimilarity matrix of the nodes
    :param wasserstein_distance: the wasserstein distance matrix
    :param cluster_method: the clustering method to use, either 'pam' or 'alternate' for KMedoids
    :param gamma_wasserstein: the gamma parameter of the RBF kernel for the wasserstein distance
    :param gamma_demand: the gamma parameter of the RBF kernel for the demand disimilarity
    :param gamma_length: the gamma parameter of the RBF kernel for the length disimilarity
    :return: the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a clustering solution
    """

    ker_wass = np.exp(- math.pow(10, gamma_wasserstein) * np.square(wasserstein_distance), dtype=np.float32)
    ker_demand = np.exp(- math.pow(10, gamma_demand) * np.square(disimilarity_demand), dtype=np.float32)
    ker_nodes = np.exp(- math.pow(10, gamma_nodes) * np.square(node_disimilarity), dtype=np.float32)

    composed_kernel = squareform(np.multiply(ker_wass, ker_demand + ker_nodes))
    # we clear the kernels to free memory
    del ker_wass, ker_demand, ker_nodes
    gc.collect()
    # The diagonal is filled according to the operations performed with the kernels
    np.fill_diagonal(composed_kernel,  (1 + kernel_shift) *  (2 + 2 * kernel_shift))

    # we center the composed kernel
    center_composed_kernel = center_kernel(composed_kernel)
    # we clear composed_kernel
    del composed_kernel
    gc.collect()

    # we initialize the key performance indicators
    CI, min_ec, FGK_index = np.zeros(K_range.size), np.zeros(K_range.size), -np.ones(K_range.size)
    # we obtain the feature map for the Nyström approximation in case the number of samples is less than the number of rows in the kernel matrix.
    # otherwise, we use the kernel PCA directly
    if nys_n_samples < center_composed_kernel.shape[0]:
        phi_wass = nystrom_map(center_composed_kernel, sampling=nys_n_samples)
        print("Nyström decomposition completed")
        # if the SVD fails, we return zeros. This can happen when the matrix presents numerical instabilities for the SVD solver.
        if np.isclose(phi_wass.sum(), 0):
            print("SVD failed")
            del center_composed_kernel, phi_wass
            gc.collect()
        
        else:
            del center_composed_kernel
            gc.collect()

            phi_wass = PCA_map(phi_wass)

            # if all the components are zero, we return trivial values
            if np.isclose(np.sum(np.abs(phi_wass)),0):
                del phi_wass
                
            # if the PCA was successful, we proceed to cluster the data
            else:
                CI, min_ec, FGK_index = [], [], []
                for c in K_range:
                    print("Clustering with {} clusters".format(c))
                    CI_result, min_ec_result, FGK_index_result = N_KMedoids(phi_wass, c, cluster_method)
                    CI.append(CI_result)
                    min_ec.append(min_ec_result)
                    FGK_index.append(FGK_index_result)
                    print("CI: {}, min_ec: {}, FGK: {}".format(CI_result, min_ec_result, FGK_index_result))
                CI, min_ec, FGK_index = np.array(CI), np.array(min_ec), np.array(FGK_index)
    
    else:
        phi_wass = kernel_PCA(center_composed_kernel)
        del center_composed_kernel
        gc.collect()

        if np.isclose(np.sum(np.abs(phi_wass)), 0):
            print("SVD failed")
            del phi_wass
            gc.collect()
        
        else:
            CI, min_ec, FGK_index = [], [], []
            for c in K_range:
                print("Clustering with {} clusters".format(c))
                CI_result, min_ec_result, FGK_index_result = N_KMedoids(phi_wass, c, cluster_method)
                CI.append(CI_result)
                min_ec.append(min_ec_result)
                FGK_index.append(FGK_index_result)
                print("CI: {}, min_ec: {}, FGK: {}".format(CI_result, min_ec_result, FGK_index_result))
            CI, min_ec, FGK_index = np.array(CI), np.array(min_ec), np.array(FGK_index)
    # we return the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices.
    return CI, min_ec, FGK_index

def optimize_kernels(black_box_fun, opt_obj, util_fun, bayes_init, bayes_iter):
    # we first generate some initial points
    init_variables, init_targets, CI_index, EC, FGK_index, iter_type = [], [], [], [], [], []
    # we print the initial points
    bayes_init_time = np.zeros(bayes_init)
    for j in range(bayes_init):
        start_time = time.time()
        next_point = opt_obj.suggest(util_fun)
        print("\nInitial iteration: ", j + 1)
        print('Point', next_point)
        CI_arr, min_ec_arr, FGK_index_arr = black_box_fun(**next_point)
        CI_index.append(CI_arr)
        EC.append(min_ec_arr)
        FGK_index.append(FGK_index_arr)
        init_variables.append(next_point)
        # we set a rawlsian utility function. The FGK index is normalized to the interval [0,1]
        target = np.min([CI_arr, (FGK_index_arr + 1)/2])
        init_targets.append(target)
        iter_type.append('random')
        end_time = time.time()
        bayes_init_time[j] = end_time - start_time
        print('Target', round(target,4))
    # we register the initial points with the target values to the optimizer
    for i in range(bayes_init):
        opt_obj.register(params=init_variables[i], target=init_targets[i])
    # we add points one by one, using the utility function to select the next one
    # we print the iteration points
    bayes_iter_time = np.zeros(bayes_iter)
    for j in range(bayes_iter):
        start_time = time.time()
        next_point = opt_obj.suggest(util_fun)
        print("\nBayes iteration: ", j + 1)
        print('Point', next_point)
        CI_arr, min_ec_arr, FGK_index_arr = black_box_fun(**next_point)
        CI_index.append(CI_arr)
        EC.append(min_ec_arr)
        FGK_index.append(FGK_index_arr)
        target = np.min([CI_arr, (FGK_index_arr + 1)/2])
        opt_obj.register(params=next_point, target=target)
        iter_type.append('bayes')
        end_time = time.time()
        bayes_iter_time[j] = end_time - start_time
        print('Target', round(target,4))
    dict_execution_time = {'init_time': bayes_init_time, 'iter_time': bayes_iter_time}
    return opt_obj, iter_type, CI_index, EC, FGK_index, dict_execution_time

def export_results(optimizer, iter_type, CI_index, EC, FGK_index, clustering_range, grid_type, eps, grid_iteration=0):
    """
    This function exports the results of the optimizer in a dataframe
    The input parameters are as follow
    :optimizer: the optimizer object
    :iter_type: the type of iteration, either 'random' or 'bayes'
    :CI_index: the consensus index values
    :EC: the effective number of components
    :FGK_index: the fast goodman-kruskal index
    :clustering_range: the range of clusters
    :grid_type: the type of grid, either 'LV' or 'MV'
    :eps: the epsilon value used to define the hyperparameter ranges
    For each combination of gamma values there are clustering_range.size values for
    the number of clusters, the CI index, the FGK index, and the EC. 
    """
    gammas = [res['params'] for res in optimizer.res]
    opt_targets = [res['target'] for res in optimizer.res]
    results = {'gamma_wasserstein': [], 'gamma_demand': [], 'gamma_nodes': [], 'iteration_type': [],
                'CI': [], 'min_ec': [], 'FGK_index': [], 'Target': [], 'clusters': []}
    for i, gamma in enumerate(gammas):
        for j in range(clustering_range.size):
            results['gamma_wasserstein'].append(gamma['gamma_wasserstein'])
            results['gamma_demand'].append(gamma['gamma_demand'])
            results['gamma_nodes'].append(gamma['gamma_nodes'])
            results['iteration_type'].append(iter_type[i])
            results['CI'].append(CI_index[i][j])
            results['min_ec'].append(EC[i][j])
            results['FGK_index'].append(FGK_index[i][j])
            results['Target'].append(opt_targets[i])
            results['clusters'].append(clustering_range[j])
    df_results = pd.DataFrame(results)
    # we sort the results by 'Target' in descending order
    df_results.sort_values(by='Target', ascending=False, inplace=True)
    # we save the results to a csv file
    clustering_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clustering', '{}'.format(grid_type))
    df_results.to_csv(os.path.join(clustering_path, 'results_{0}_gamma - Copy ({1}).csv'.format(str(eps*2).replace('.','_'), grid_iteration)), index=False)
    return

def obtain_dissimilarity_matrices(_node_embeddings, _dict_names):
    _node_embeddings = _node_embeddings[_node_embeddings['grid'].isin(_dict_names.values())]
    # we sort the node_embeddings and the length_grids according to the order of dict_names in the column 'grid'	
    order_grid_names = [_dict_names[i] for i in range(len(_dict_names))]
    # Create a categorical ordering for the 'grid' column based on order_grid_names
    _node_embeddings['grid'] = pd.Categorical(_node_embeddings['grid'], categories=order_grid_names, ordered=True)
    # Sort the dataframes based on the new categorical ordering
    _node_embeddings = _node_embeddings.sort_values(by='grid')
    # we compute the penalty matrix by summing the total demand of the grids in the embeddings using groupby
    weight_penalty = _node_embeddings.groupby('grid', observed=True)['demand'].sum().values
    weight_penalty = weight_penalty.reshape((-1,1))
    demand_disimilarity_matrix = np.abs(weight_penalty - weight_penalty.T)
    # we count the nodes in the embeddings
    nodes_penalty = _node_embeddings.groupby('grid', observed=True).size().values
    nodes_penalty = nodes_penalty.reshape((-1,1))
    nodes_disimilarity_matrix = np.abs(nodes_penalty - nodes_penalty.T)

    return demand_disimilarity_matrix, nodes_disimilarity_matrix

# we especify this code to be run only if the file is run as a script
if __name__ == '__main__':

    grid_type = 'LV'
    if not grid_type in ['LV', 'MV']:
        raise ValueError("Invalid grid type. Choose either 'LV' or 'MV'.")
    # We shift the kernel matrices and fill the diagonal with ones + the shift
    clustering_range = np.array([10]) # the number of clusters to test
    bayes_iter, bayes_init = 50, 50 # the number of iterations and initial points for the optimizer
    eps = 0.5 # the epsilon value used to define the hyperparameter ranges
    # optimal zscores for the wasserstein
    zscores = {'LV': -0.5, 'MV': -0.5}

    ##########################################################
    # we read the names of the grids in the embeddings
    # we check if the folder exist, and if not, we raise a warning
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings')):
        raise Warning('The folder embeddings does not exist' + '\n Please, generate the embeddings with node_embeddings.py!')

    with open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings'), 
                'normalized_grids_names_{}.pickle'.format(grid_type)), 'rb') as handle:
        dict_names = pickle.load(handle)


    node_embeddings = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings', 
                'node_embeddings_{}.csv'.format(grid_type)))
    # we export the length_grids dataframe to a csv file

    demand_disimilarity_matrix, nodes_disimilarity_matrix = obtain_dissimilarity_matrices(node_embeddings, dict_names)

    # we clear the node embeddings to free memory
    del node_embeddings
    gc.collect()

    ##########################################################
    ## Kernel definition
    ##########################################################

    # we compute the optimal dispersions for the kernels
    compute_kernel_dispersions = False

    # we check if the folder exist, and if not, we raise a warning
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasserstein_models')):
        raise Warning('The folder wasserstein_models does not exist' \
            + '\n Please, generate the wasserstein models with wasserstein_computations.py!')
    # we read the wasserstein model
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), \
            'wasserstein_models', 'wass_model_{}.pickle'.format(grid_type)), 'rb') as handle:
        wass_grids = pickle.load(handle)
    
    wass_grids.recompute_distance_with_zscore(zscores[grid_type])
    wass_distance = wass_grids.min_distances_.copy()
    # we retrieve the distance from the object and then delete it
    del wass_grids
    gc.collect()
    clustering_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clustering', '{}'.format(grid_type))
    if compute_kernel_dispersions:
        # we check if the gammas folder exist, and if not, we create it
        if not os.path.exists(clustering_path):
            os.makedirs(clustering_path)

        print("\nComputing the optimal dispersion for the Wasserstein kernel\n")
        gammas_wasserstein, dispersion_wasserstein = optimal_dispersion_kernel(wass_distance)
        np.save(os.path.join(clustering_path, 'dispersion_wasserstein.npy'), dispersion_wasserstein)
        np.save(os.path.join(clustering_path, 'gamma_wasserstein.npy'), gammas_wasserstein)

        print("\nComputing the optimal dispersion for the Demand kernel\n")
        gammas_demand, dispersion_demand = optimal_dispersion_kernel(demand_disimilarity_matrix)
        np.save(os.path.join(clustering_path, 'dispersion_demand.npy'), dispersion_demand)
        np.save(os.path.join(clustering_path, 'gamma_demand.npy'), gammas_demand)

        print("\nComputing the optimal dispersion for the Nodes kernel\n")
        gammas_nodes, dispersion_nodes = optimal_dispersion_kernel(nodes_disimilarity_matrix)
        np.save(os.path.join(clustering_path, 'dispersion_nodes.npy'), dispersion_nodes)
        np.save(os.path.join(clustering_path, 'gamma_nodes.npy'), gammas_nodes)

    else:
        # we check if the gammas folder exist, and if not, we raise a warning
        if not os.path.exists(clustering_path):
            raise Warning('The folder {} does not exist'.format(clustering_path) \
                + '\n Please, compute the optimal dispersions first by setting compute_kernel_dispersions = True!')
        # we load the optimal dispersions arrays with numpy
        dispersion_wasserstein = np.load(os.path.join(clustering_path, 'dispersion_wasserstein.npy'))
        dispersion_demand = np.load(os.path.join(clustering_path, 'dispersion_demand.npy'))
        dispersion_nodes = np.load(os.path.join(clustering_path, 'dispersion_nodes.npy'))
        # we also load the gammas arrays
        gammas_wasserstein = np.load(os.path.join(clustering_path, 'gamma_wasserstein.npy'))
        gammas_demand = np.load(os.path.join(clustering_path, 'gamma_demand.npy'))
        gammas_nodes = np.load(os.path.join(clustering_path, 'gamma_nodes.npy'))

    ##########################################################
    ## Nyström approximation
    ##########################################################

    if demand_disimilarity_matrix.ndim == 2:
        demand_disimilarity_matrix = squareform(demand_disimilarity_matrix)
    if nodes_disimilarity_matrix.ndim == 2:
        nodes_disimilarity_matrix = squareform(nodes_disimilarity_matrix)
    if wass_distance.ndim == 2:
        wass_distance = squareform(wass_distance)



    var_gamma = {'gamma_wasserstein': gammas_wasserstein[np.argmax(dispersion_wasserstein)], 'gamma_demand': gammas_demand[np.argmax(dispersion_demand)], 'gamma_nodes': gammas_nodes[np.argmax(dispersion_nodes)]}


    copy_number = 0

    grid_test = [[i,j] for i in [1.0] for j in [-1.0, 1.0]]

    for n, grid_test_i in enumerate(grid_test):

        print("Testing grid: ", grid_test_i)

        displacement_gamma = {'LV': {'gamma_wasserstein': -0.5, 'gamma_demand': -1.0, 'gamma_nodes': 1.5}, 'MV': {'gamma_wasserstein': -0.5, 'gamma_demand': grid_test_i[0], 'gamma_nodes': grid_test_i[1]}}

        hyp_ranges = {'gamma_wasserstein': (var_gamma['gamma_wasserstein'] - eps + displacement_gamma[grid_type]['gamma_wasserstein'], var_gamma['gamma_wasserstein'] + eps + displacement_gamma[grid_type]['gamma_wasserstein']), 
                    'gamma_demand': (var_gamma['gamma_demand'] - eps + displacement_gamma[grid_type]['gamma_demand'], var_gamma['gamma_demand'] + eps + displacement_gamma[grid_type]['gamma_demand']), 
                    'gamma_nodes': (var_gamma['gamma_nodes'] - eps + displacement_gamma[grid_type]['gamma_nodes'], var_gamma['gamma_nodes'] + eps + displacement_gamma[grid_type]['gamma_nodes'])} # gamma is in scientific notation

        optimizer = BayesianOptimization(f=None, pbounds=hyp_ranges, verbose=2, random_state=1, allow_duplicate_points=True)
        # we use the expected improvement acquisition function
        utility_function = UtilityFunction(kind="ei", xi=0.0)
        if grid_type == 'MV':
            clustering_method_kmedoids = 'pam'
        elif grid_type == 'LV':
            clustering_method_kmedoids = 'alternate'

        black_box = partial(composed_kernel_clustering, clustering_range, demand_disimilarity_matrix, nodes_disimilarity_matrix, wass_distance, clustering_method_kmedoids)

        start_time = time.time()
        ############
        opt_obj, iter_type, CI_index, EC, FGK_index, dict_execution_time = optimize_kernels(black_box, optimizer, utility_function, bayes_init, bayes_iter)

        export_results(optimizer, iter_type, CI_index, EC, FGK_index, clustering_range, grid_type, eps, n + copy_number)
        ############
        end_time = time.time()
        print("Execution time: ", end_time - start_time)
        # we save the execution time dictionary to a pickle file
        with open(os.path.join(clustering_path, 'execution_time_{0}_gamma - Copy ({1}).pickle'.format(str(eps*2).replace('.','_'), n + copy_number)), 'wb') as handle:
            pickle.dump(dict_execution_time, handle, protocol=pickle.HIGHEST_PROTOCOL)