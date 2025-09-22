"""Utilities for validating clustering of power distribution graphs.

The script aggregates helper functions and an executable section that computes
context-aware validity indices for clusters of electrical grids.  Distances
between grids are derived from empirical cumulative distribution functions
(CDFs) and combined with K-medoids clustering.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('BayesOpt/'))

import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
import pickle
from validity_measures import FGK
import geopandas as gpd
from statsmodels.distributions.empirical_distribution import ECDF
import time
from sklearn.metrics import pairwise_distances
from wass_kernel_clustering import N_KMedoids, center_kernel, nystrom_map, PCA_map, kernel_PCA
from scipy.spatial.distance import squareform
from wasserstein_computations import Wasserstein
from sklearn_extra.cluster import KMedoids
import math
import gc

def N_KMedoids(X, clusters, grid_type, n_init=3, n_fgk=100, m_fgk=35):
    """
    This function computes the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a KMedoids clustering solution
    :param X: the dataset
    :param clusters: the number of clusters
    :param n_init: the number of times the KMedoids algorithm is run
    :param fgk_rejection: the threshold for the fast goodman-kruskal index. If the index is below this threshold, the algorithm stops
    :param n_fgk: the number of pairs of points to sample for each cluster in the fast goodman-kruskal index
    :param m_fgk: the number of times to repeat the sampling procedure in the fast goodman-kruskal index
    :return: max_fgk_index, labels, medoids
    """
    labels, max_fgk_index, medoids = [], -1, []
    # we run the KMedoids algorithm n_init times
    for _ in range(n_init):
        # we create an instance of KMedoids with the number of clusters
        if grid_type == 'MV':
            clustering_model = KMedoids(n_clusters=clusters, init='k-medoids++', method='pam').fit(X)
        elif grid_type == 'LV':
            clustering_model = KMedoids(n_clusters=clusters, init='k-medoids++', method='alternate').fit(X)
        # we check the current davies-bouldin score
        if np.unique(clustering_model.labels_).shape[0] == clusters:
            fgk = FGK(n=n_fgk, m=m_fgk)
            fgk.fit(X, clustering_model.labels_)
            if fgk.GK_ > max_fgk_index:
                max_fgk_index = fgk.GK_
                labels = clustering_model.labels_
                medoids = clustering_model.medoid_indices_
    return labels, medoids

def composed_kernel(dissimilarities, gammas, kernel_shift=1e-3, nys_n_samples=5000):
    """
    This function computes the composed kernel
    :param disimilarities: dictionary with the dissimilarities considered 
    :param gammas: dictionary with the optimal gamma parameters
    :return: the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a clustering solution
    """

    if dissimilarities['demand_dissimilarity'].ndim == 2:
        dissimilarities['demand_dissimilarity'] = squareform(dissimilarities['demand_dissimilarity'])
    if dissimilarities['nodes_dissimilarity'].ndim == 2:
        dissimilarities['nodes_dissimilarity'] = squareform(dissimilarities['nodes_dissimilarity'])
    if dissimilarities['wasserstein_dissimilarity'].ndim == 2:
        dissimilarities['wasserstein_dissimilarity'] = squareform(dissimilarities['wasserstein_dissimilarity'])

    ker_wass = np.exp(- math.pow(10, gammas['gamma_wasserstein']) * dissimilarities['wasserstein_dissimilarity'], dtype=np.float32)
    ker_demand = np.exp(- math.pow(10, gammas['gamma_demand']) * dissimilarities['demand_dissimilarity'], dtype=np.float32)
    ker_nodes = np.exp(- math.pow(10, gammas['gamma_nodes']) * dissimilarities['nodes_dissimilarity'], dtype=np.float32)

    composed_kernel = squareform(np.multiply(ker_wass, ker_demand + ker_nodes))
    # we clear the kernels to free memory
    del ker_wass, ker_demand, ker_nodes
    gc.collect()
    # The diagonal is filled according to the operations performed with the kernels
    np.fill_diagonal(composed_kernel,  (1 + kernel_shift) * (2 + 2 * kernel_shift) )

    # we center the composed kernel
    center_composed_kernel = center_kernel(composed_kernel)
    # we clear composed_kernel
    del composed_kernel
    gc.collect()

    # we obtain the feature map for the Nyström approximation in case the number of samples is less than the number of rows in the kernel matrix.
    # otherwise, we use the kernel PCA directly
    if nys_n_samples < center_composed_kernel.shape[0]:
        phi_wass = nystrom_map(center_composed_kernel, sampling=nys_n_samples)
        
        del center_composed_kernel
        gc.collect()

        phi_wass = PCA_map(phi_wass)
    
    else:
        phi_wass = kernel_PCA(center_composed_kernel)
        del center_composed_kernel
        gc.collect()
    # we return the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices.
    return phi_wass

def get_grid_dissimilarities(grid_type, _embeddings_path, _wass_path):
    # we read the names of the grids in the embeddings
    with open(os.path.join(_embeddings_path, 'normalized_grids_names_{}.pickle'.format(grid_type)), 'rb') as handle:
        dict_names = pickle.load(handle)

    node_embeddings = pd.read_csv(os.path.join(_embeddings_path, 'node_embeddings_{}.csv'.format(grid_type)))
    # we export the length_grids dataframe to a csv file

    node_embeddings = node_embeddings[node_embeddings['grid'].isin(dict_names.values())]

    # we sort the node_embeddings and the length_grids according to the order of dict_names in the column 'grid'	
    order_grid_names = [dict_names[i] for i in range(len(dict_names))]
    
    # Create a categorical ordering for the 'grid' column based on order_grid_names
    node_embeddings['grid'] = pd.Categorical(node_embeddings['grid'], categories=order_grid_names, ordered=True)

    # Sort the dataframes based on the new categorical ordering
    node_embeddings = node_embeddings.sort_values(by='grid')

    # we compute the penalty matrix by summing the total demand of the grids in the embeddings using groupby
    penalize_entry = 'demand'
    weight_penalty = node_embeddings.groupby('grid', observed=True)[penalize_entry].sum().values
    weight_penalty = weight_penalty.reshape((-1,1))
    demand_disimilarity_matrix = np.abs(weight_penalty - weight_penalty.T)
    # we count the nodes in the embeddings
    nodes_penalty = node_embeddings.groupby('grid', observed=True).size().values
    nodes_penalty = nodes_penalty.reshape((-1,1))
    nodes_disimilarity_matrix = np.abs(nodes_penalty - nodes_penalty.T)

    # we clear the node embeddings to free memory
    del node_embeddings
    gc.collect()

    # we read the wasserstein model
    with open(os.path.join(_wass_path, 'wass_model_{}.pickle'.format(grid_type)), 'rb') as handle:
        wass_grids = pickle.load(handle)
    
    wass_distance = wass_grids.min_distances_.copy()
    # we retrieve the distance from the object and then delete it
    del wass_grids
    gc.collect()
    return wass_distance, demand_disimilarity_matrix, nodes_disimilarity_matrix

def get_grid_properties(folder_path, grid_name):
    """Read node data for a grid and build basic CDF descriptors."""
    gdf_nodes = gpd.read_file(folder_path + '{}_nodes'.format(grid_name))
    voltage = gdf_nodes['voltage'].values
    demand = gdf_nodes['el_dmd'].values
    cdf_voltage = ECDF(voltage)
    cdf_demand = ECDF(demand)
    total_nodes = len(voltage)
    total_demand = sum(demand)
    return {
        'cdf_voltage': cdf_voltage,
        'cdf_demand': cdf_demand,
        'total_nodes': total_nodes,
        'total_demand': total_demand,
        'grid_id': grid_name,
    }

def unpack_grid_properties(grid_results):
    """Transform a list of grid-property dictionaries into a lookup table."""
    return {
        grid_results[j]['grid_id']: {
            k: grid_results[j][k] for k in grid_results[j] if k != 'grid_id'
        }
        for j in range(len(grid_results))
    }

def W1_cdf_based(cdf1, cdf2, x_range):
    """Approximate the 1-Wasserstein distance between two CDFs on ``x_range``."""
    return np.sum(np.abs(cdf1 - cdf2)) * (x_range[-1] - x_range[0]) / x_range.size

def W1_pairwise(cdf_dict, x_range):
    """Compute all pairwise 1-Wasserstein distances between CDFs in ``cdf_dict``."""
    list_keys = list(cdf_dict.keys())
    W1_df = pd.DataFrame(index=list_keys, columns=list_keys)
    W1_values = np.zeros((len(list_keys), len(list_keys)))
    # as the distance matrix is symmetric, we only need to compute the upper triangular part
    pairs_to_compute = [(i, j) for i in range(len(list_keys)) for j in range(len(list_keys)) if j > i]
    jobs = [(cdf_dict[list_keys[p[0]]], cdf_dict[list_keys[p[1]]], x_range) for p in pairs_to_compute]
    # use multiprocessing to compute the Wasserstein-1 distances in parallel
    n_cores = cpu_count() - 2
    with Pool(n_cores) as p:
        W1_distances = p.starmap(W1_cdf_based, jobs)
    p.close()

    # assign the computed distances to the corresponding elements in the matrix
    for p, W1 in zip(pairs_to_compute, W1_distances):
        W1_values[p[0], p[1]] = W1_values[p[1], p[0]] = W1

    np.fill_diagonal(W1_values, 0)
    # we store the values in the dataframe
    W1_df.loc[:] = W1_values
    W1_df = W1_df.apply(pd.to_numeric, errors='coerce')
    return W1_df

def davies_bouldin_distance_based(in_clusters, pairwise_dist, denominator_eps=1e-10):
    """
    This function computes the Davies-Bouldin index for a clustering based on the pairwise distances between the elements in the clusters.
    in_clusters: is a dictionary with the keys of the clusters and the values are the list of grid names in the clusters
    pairwise_dist: is a dataframe with the pairwise distances between the grids
    epsilon: small constant to avoid division by zero
    """
    average_distance_to_medoid = {}
    keys_in_sampled_distances = pairwise_dist.keys()    
    in_clusters = {k: [j for j in in_clusters[k] if j in keys_in_sampled_distances] for k in in_clusters}
    for cluster in in_clusters:
        cluster_distances = pairwise_dist[in_clusters[cluster]].loc[in_clusters[cluster]].copy()
        distances = cluster_distances.sum(axis=1)
        # we obtain the index of the row with the minimum sum of distances, which is the medoid
        medoid = distances.idxmin()
        # we obtain S, the average distance to the medoid
        average_distance_to_medoid[medoid] = cluster_distances.loc[medoid].mean()
    intracluster_distances = pairwise_dist[average_distance_to_medoid.keys()].loc[average_distance_to_medoid.keys()].copy()
    # we build the matrix R = (Scluster_i + Scluster_j) / M_intracluster_ij
    R_matrix = np.zeros((len(average_distance_to_medoid), len(average_distance_to_medoid)))
    for i, med_i in enumerate(average_distance_to_medoid):
        for j, med_j in enumerate(average_distance_to_medoid):
            if j > i:
                R_matrix[i, j] = R_matrix[j, i] = \
                    (average_distance_to_medoid[med_i] + average_distance_to_medoid[med_j]) \
                        / (intracluster_distances.loc[med_i, med_j] + denominator_eps)
            elif j == i:
                R_matrix[i, j] = 0
    # then, we get the vector D = max(R_i) for i different from j
    max_R_values = np.max(R_matrix, axis=1)
    # the Davies-Bouldin index is the average of D
    davies_bouldin_index = np.mean(max_R_values)
    return davies_bouldin_index

def compute_wass_distances(grid_type, random_selection = 25000):
    if grid_type not in ['MV', 'LV']:
        raise ValueError('grid_type must be either MV or LV')
    if grid_type == 'MV':
        folder_mv = os.path.join(grids_path, 'PDGs_{}/'.format(grid_type))
        jobs = [(folder_mv, mv) for mv in grid_names[grid_type].values()]
    elif grid_type == 'LV':
        folder_lv = os.path.join(grids_path, 'PDGs_LV/')
        regions_lv = ['Alps-Periurban', 'Alps-Rural', 'Alps-Urban', 
                      'Jura-Periurban', 'Jura-Rural', 'Jura-Urban', 
                      'Midlands-Periurban', 'Midlands-Rural', 'Midlands-Urban']
        # we check which files are in the folders
        files_lv = {region: os.listdir(os.path.join(folder_lv, region)) for region in regions_lv}
        grids_lv = {region: [f.replace('_edges','') for f in files_lv[region] if 'edges' in f] for region in regions_lv}
        jobs = [(os.path.join(folder_lv, region,''), lv) for region in regions_lv for lv in grids_lv[region] if lv in grid_names['LV'].values()]
        # we get a random sample of 5000 grids without replacement
        jobs = [jobs[i] for i in np.random.choice(range(len(jobs)), random_selection, replace=False)]

    print("Getting grid properties")
    n_cores = cpu_count() - 2
    with Pool(n_cores) as p:
        results = p.starmap(get_grid_properties, jobs)
    p.close()
    grid_properties = unpack_grid_properties(results)

    print("Computing the CDFs")
    if grid_type == 'MV':
        # the support is already known for MV grids
        # we use 100 points between 0 and 4 for the demand
        cdf_demand_range = np.linspace(0, 4, 100)
        # we use 100 points between 0.98 and 1 for the voltage
        cdf_voltage_range = np.linspace(0.98, 1, 100)
    elif grid_type == 'LV':
        print("The process might take a while...")
        # the support is already known for LV grids
        # we use 100 points between 0 and 0.1 for the demand
        cdf_demand_range = np.linspace(0, 0.1, 100)
        # we use 100 points between 0.97 and 1 for the voltage 
        cdf_voltage_range = np.linspace(0.97, 1, 100)

    start_time = time.time()
    cdf_voltage = {g: grid_properties[g]['cdf_voltage'](cdf_voltage_range) for g in grid_properties.keys()}
    W1_voltage = W1_pairwise(cdf_voltage, cdf_voltage_range)
    W1_voltage.to_pickle(os.path.join(context_path, '{}_voltage_wasserstein.pkl'.format(grid_type)))
    end_time = time.time()
    print("Computing time: ", end_time - start_time)
    del cdf_voltage, W1_voltage
    gc.collect()

    start_time = time.time()
    cdf_demand = {g: grid_properties[g]['cdf_demand'](cdf_demand_range) for g in grid_properties.keys()}
    W1_demand = W1_pairwise(cdf_demand, cdf_demand_range)
    W1_demand.to_pickle(os.path.join(context_path, '{}_demand_wasserstein.pkl'.format(grid_type)))
    end_time = time.time()
    print("Computing time: ", end_time - start_time)
    del cdf_demand, W1_demand
    gc.collect()

    grid_ids = list(grid_properties.keys())
    Dgrids = np.array([grid_properties[grid_ids[i]]['total_demand'] for i in range(len(grid_ids))])
    Ngrids = np.array([grid_properties[grid_ids[i]]['total_nodes'] for i in range(len(grid_ids))])
    Dz = (Dgrids - Dgrids.min())/(Dgrids.max() - Dgrids.min())
    Nz = (Ngrids - Ngrids.min())/(Ngrids.max() - Ngrids.min())
    Z = np.array([[Dz[i], Nz[i]] for i in range(Dz.size)])
    # we compute the pairwise distances between the grids
    Z_distances = pd.DataFrame(pairwise_distances(Z), index=grid_ids, columns=grid_ids)
    # we export the distances
    Z_distances.to_pickle(os.path.join(context_path, '{}_z_distances.pkl'.format(grid_type)))
    del Z_distances
    gc.collect()

    return

def read_wass_distances(grid_type):
    if grid_type not in ['MV', 'LV']:
        raise ValueError('grid_type must be either MV or LV')
    # we check if the Wasserstein distance matrices exist, and if not, we raise a warning
    if not os.path.exists(os.path.join(context_path, '{}_voltage_wasserstein.pkl'.format(grid_type))) \
        or not os.path.exists(os.path.join(context_path, '{}_demand_wasserstein.pkl'.format(grid_type))) \
            or not os.path.exists(os.path.join(context_path, '{}_z_distances.pkl'.format(grid_type))):
        raise Warning('The Wasserstein distance matrices for LV grids do not exist' + '\n Please, compute them by setting compute_lv_distances = True!')
    # we read the Wasserstein distance matrices
    distances = {}
    distances['voltage'] = pd.read_pickle(os.path.join(context_path, '{}_voltage_wasserstein.pkl'.format(grid_type)))
    distances['demand'] = pd.read_pickle(os.path.join(context_path, '{}_demand_wasserstein.pkl'.format(grid_type)))
    distances['z_vec'] = pd.read_pickle(os.path.join(context_path, '{}_z_distances.pkl'.format(grid_type)))

    return distances

def cluster_grids_feature_maps(_grid_type, _files_results, _copy_number):
    # we separate the pickle files and the csv files
    csv_files_results = [file for file in _files_results if file.endswith('.csv') and _copy_number in file]
    # we load the csv files
    indices_results = {file.split('.')[0]: pd.read_csv(os.path.join(clustering_path, _grid_type, file)) for file in csv_files_results}
    # we read the grid names from the pickle files 'data/grids_names_{_grid_type}.pickle'
    grid_names = pickle.load(open(os.path.join(embeddings_path, 'normalized_grids_names_{}.pickle'.format(_grid_type)), 'rb'))

    # we cluster the grids
    test_loc = [test for test in range(0,10) if test in indices_results['results - Copy {0}'.format(_copy_number)].index]

    dissimilarities_grids = {'wasserstein_dissimilarity': [], 'demand_dissimilarity': [], 'nodes_dissimilarity': []}

    results = get_grid_dissimilarities(_grid_type, embeddings_path, wass_path)
    dissimilarities_grids['wasserstein_dissimilarity'], dissimilarities_grids['demand_dissimilarity'], dissimilarities_grids['nodes_dissimilarity'] = results
    del results
    gc.collect()

    print("Clustering {} grids".format(_grid_type))
    for test in test_loc:
        gammas = indices_results['results - Copy {0}'.format(_copy_number)].iloc[test][['gamma_wasserstein', 'gamma_demand', 'gamma_nodes']].to_dict()
        print("Computing feature map for {} with gamma: ".format(_grid_type), gammas)
        feature_map = composed_kernel(dissimilarities_grids, gammas)
        start_time = time.time()
        labels, medoids = N_KMedoids(feature_map, 10, _grid_type)
        end_time = time.time()
        print("Time elapsed clustering loc {0}: ".format(test), end_time - start_time)
        # we export the labels
        clustered_grids = pd.DataFrame({'grid_id': [grid_names[i] for i in range(len(grid_names))], 'cluster': labels})
        map_medoid = {i: grid_names[medoids[i]] for i in range(len(medoids))}
        clustered_grids['cluster'] = clustered_grids['cluster'].map(map_medoid)
        clustered_grids.to_csv(os.path.join(clustered_grids_path, 'clustered_grids_{0}_{1} - Copy {2}.csv'.format(_grid_type, str(test),_copy_number)), index=False)
    return

def clusters_davies_bouldin_relative_index(_grid_type, _cluster_grids, _distances):
    davies_bouldin_values = { }
    distances_types = ['voltage', 'demand', 'z_vec']

    start_time = time.time()
    clusters = {}
    for distance_typ in distances_types:
        print("------------------------------------")
        print("Computing Davies-Bouldin index for {} grids with distance type: ".format(_grid_type), distance_typ)
        # we run K-medoids with the feature vectors
        grids_in_index = _distances[distance_typ].index.tolist()
        distances_values = _distances[distance_typ].values
        kmedoids = KMedoids(n_clusters=K_clusters, metric='precomputed', method='alternate', random_state=0).fit(distances_values)
        # we get the labels
        labels = kmedoids.labels_
        # we get the medoid indices
        medoid_indices = kmedoids.medoid_indices_  
        # we get the clusters in a dictionary
        clusters[distance_typ] = {grids_in_index[medoid_indices[k]]: [grids_in_index[i] for i in np.where(labels == k)[0]] for k in range(K_clusters)}
        davies_bouldin_values[distance_typ] = {distance_to_compare: \
                davies_bouldin_distance_based(clusters[distance_typ], _distances[distance_to_compare]) for distance_to_compare in _distances}
    end_time = time.time()
    print("Computing time: ", end_time - start_time)

    for copy_number in _cluster_grids:
        test_loc = [test for test in range(0,10) if test in \
            pd.read_csv(os.path.join(clustering_path, _grid_type, 'results - Copy {0}.csv'.format(copy_number))).index]
        for loc in test_loc:
            # we check if the file exists
            if not os.path.exists(os.path.join(clustered_grids_path, "clustered_grids_{0}_{1} - Copy {2}.csv".format(_grid_type, str(loc), copy_number))):
                raise Warning('The file clustered_grids_{0}_{1} - Copy {2}.csv does not exist'.format(_grid_type, str(loc), copy_number) \
                              + '\n Please, compute the clustering with cluster_grids_feature_maps function!')
            clusters_grids = pd.read_csv(os.path.join(clustered_grids_path, "clustered_grids_{0}_{1} - Copy {2}.csv".format(_grid_type, str(loc), copy_number)))
            keys_in_clusters = clusters_grids.groupby('cluster')['grid_id'].apply(list).to_dict()
            davies_bouldin_values['wasserstein_{}'.format(loc)] = {distance_to_compare: \
                davies_bouldin_distance_based(keys_in_clusters, _distances[distance_to_compare]) for distance_to_compare in _distances}

        # we change the entries of davies_bouldin_values to a dataframe
        davies_bouldin_values = pd.DataFrame(davies_bouldin_values).T
        # we normalize with min-max normalization each row of the dataframe
        davies_bouldin_values = (davies_bouldin_values - davies_bouldin_values.min()) / (davies_bouldin_values.max() - davies_bouldin_values.min())
        # we transpose the dataframe back
        davies_bouldin_values = davies_bouldin_values.T
        # we add two rows to the dataframes. One with the mean of the values and the other with the maximum of the values
        mean_values, max_values = pd.DataFrame(davies_bouldin_values.mean()).T, pd.DataFrame(davies_bouldin_values.max()).T
        # we reindex the dataframes with 'mean' and 'max'
        mean_values.index, max_values.index  = ['mean'], ['max']
        # we concatenate the dataframes
        davies_bouldin_values = pd.concat([davies_bouldin_values, mean_values, max_values])
        # we export the results
        davies_bouldin_values.to_csv(os.path.join(context_path, '{0}_davies_bouldin_values - Copy {1}.csv'.format(_grid_type, copy_number)))

# we especify this code to be run only if the file is run as a script
if __name__ == '__main__':

    # we read the grid names from the pickle files 'embeddings/grids_names_LV.pickle' and 'embeddings/grids_names_MV.pickle'
    embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings')
    grids_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    context_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'context_validity')
    clustering_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clustering')
    wass_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasserstein_models')
    if not os.path.exists(os.path.join(context_path, 'clustered_grids')):
        os.makedirs(os.path.join(context_path, 'clustered_grids'))
    clustered_grids_path = os.path.join(context_path, 'clustered_grids')
    # we check if the folder exist, and if not, we raise a warning
    if not os.path.exists(embeddings_path):
        raise Warning('The folder embeddings does not exist' + '\n Please, create the embeddings with node_embeddings.py!')
    if not os.path.exists(grids_path):
        raise Warning('The folder data does not exist' + '\n Please, download the data from the data repository!')
    
    # we create a directory to store the context-dependent validity results if it does not exist
    if not os.path.exists(context_path):
        os.makedirs(context_path)

    grid_names = {'MV': pickle.load(open(os.path.join(embeddings_path, 'grids_names_MV.pickle'), 'rb')), 
                  'LV': pickle.load(open(os.path.join(embeddings_path, 'grids_names_LV.pickle'), 'rb'))}

    compute_mv_distances,  compute_lv_distances = False, False

    if compute_mv_distances:
        compute_wass_distances('MV')
    else:
        distances_mv = read_wass_distances('MV')

    if compute_lv_distances:
        compute_wass_distances('LV')
    else:
        distances_lv = read_wass_distances('LV')

    cluster_grids = {'LV': ['(0)'], 'MV': ['(0)']}
    perform_clustering = False

    for grid_type in cluster_grids:
        if cluster_grids[grid_type] is not None and perform_clustering:
            files_results = os.listdir(os.path.join(clustering_path, grid_type))
            for copy_number in cluster_grids[grid_type]:
                cluster_grids_feature_maps(grid_type, files_results, copy_number)

    K_clusters = 10
    compute_davies_bouldin = True

    # we compute the Davies-Bouldin index for the clustering results
    if compute_davies_bouldin:
        clusters_davies_bouldin_relative_index('MV', cluster_grids['MV'], distances_mv)
        clusters_davies_bouldin_relative_index('LV', cluster_grids['LV'], distances_lv)