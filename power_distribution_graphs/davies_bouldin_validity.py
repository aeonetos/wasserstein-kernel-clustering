"""Utilities for validating clustering of power distribution graphs.

The script aggregates helper functions and an executable section that computes
context-aware validity indices for clusters of electrical grids.  Distances
between grids are derived from empirical cumulative distribution functions
(CDFs) and combined with K-medoids clustering.
"""

import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
import pickle
import os
import geopandas as gpd
from statsmodels.distributions.empirical_distribution import ECDF
import time
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
import gc


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

    for i in range(len(list_keys)):
        W1_values[i, i] = 0
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
                R_matrix[i, j] = R_matrix[j, i] = (average_distance_to_medoid[med_i] + average_distance_to_medoid[med_j]) / (intracluster_distances.loc[med_i, med_j] + denominator_eps)
            elif j == i:
                R_matrix[i, j] = 0
    # then, we get the vector D = max(R_i) for i different from j
    max_R_values = np.max(R_matrix, axis=1)
    # the Davies-Bouldin index is the average of D
    davies_bouldin_index = np.mean(max_R_values)
    return davies_bouldin_index

# we especify this code to be run only if the file is run as a script
if __name__ == '__main__':

    # we read the grid names from the pickle files 'embeddings/grids_names_LV.pickle' and 'embeddings/grids_names_MV.pickle'
    embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings')
    grids_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    context_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'context_validity')
    clustering_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clustering')
    # we check if the folder exist, and if not, we raise a warning
    if not os.path.exists(embeddings_path):
        raise Warning('The folder embeddings does not exist' + '\n Please, create the embeddings with node_embeddings.py!')
    if not os.path.exists(grids_path):
        raise Warning('The folder data does not exist' + '\n Please, download the data from the data repository!')
    
    # we create a directory to store the context-dependent validity results if it does not exist
    if not os.path.exists(context_path):
        os.makedirs(context_path)

    grid_names = {'MV': pickle.load(open(os.path.join(embeddings_path, 'grids_names_MV.pickle'), 'rb')), 'LV': pickle.load(open(os.path.join(embeddings_path, 'grids_names_LV.pickle'), 'rb'))}

    compute_mv_distances = False

    if compute_mv_distances:
        folder_mv = os.path.join(grids_path, 'PDGs_MV/')

        jobs = [(folder_mv, mv) for mv in grid_names['MV'].values()]

        n_cores = cpu_count() - 2
        with Pool(n_cores) as p:
            results_mv = p.starmap(get_grid_properties, jobs)
        p.close()
        grid_properties_mv = unpack_grid_properties(results_mv)

        cdf_demand_range_mv = np.linspace(0, 4, 100)
        cdf_voltage_range_mv = np.linspace(0.98, 1, 100)
        cdf_demand_mv = {grid_names['MV'][g]: grid_properties_mv[grid_names['MV'][g]]['cdf_demand'](cdf_demand_range_mv) for g in grid_names['MV']}
        cdf_voltage_mv = {grid_names['MV'][g]: grid_properties_mv[grid_names['MV'][g]]['cdf_voltage'](cdf_voltage_range_mv) for g in grid_names['MV']}

        W1_voltage_mv = W1_pairwise(cdf_voltage_mv, cdf_voltage_range_mv)
        W1_demand_mv = W1_pairwise(cdf_demand_mv, cdf_demand_range_mv)

        # we export the Wasserstein distance matrices
        W1_voltage_mv.to_pickle(os.path.join(context_path, 'MV_voltage_wasserstein.pkl'))
        W1_demand_mv.to_pickle(os.path.join(context_path, 'MV_demand_wasserstein.pkl'))

        grid_ids = list(cdf_demand_mv.keys())
        Dgrids_mv = np.array([grid_properties_mv[grid_ids[i]]['total_demand'] for i in range(len(grid_ids))])
        Ngrids_mv = np.array([grid_properties_mv[grid_ids[i]]['total_nodes'] for i in range(len(grid_ids))])
        Dz_mv = (Dgrids_mv - Dgrids_mv.min())/(Dgrids_mv.max() - Dgrids_mv.min())
        Nz_mv = (Ngrids_mv - Ngrids_mv.min())/(Ngrids_mv.max() - Ngrids_mv.min())
        Z_mv = np.array([[Dz_mv[i], Nz_mv[i]] for i in range(Dz_mv.size)])
        # we compute the pairwise distances between the grids
        Z_distances_mv = pd.DataFrame(pairwise_distances(Z_mv), index=grid_ids, columns=grid_ids)
        # we export the distances
        Z_distances_mv.to_pickle(os.path.join(context_path, 'MV_z_distances.pkl'))

    else:
        # we check if the Wasserstein distance matrices exist, and if not, we raise a warning
        if not os.path.exists(os.path.join(context_path, 'MV_voltage_wasserstein.pkl')) \
            or not os.path.exists(os.path.join(context_path, 'MV_demand_wasserstein.pkl')) \
                or not os.path.exists(os.path.join(context_path, 'MV_z_distances.pkl')):
            raise Warning('The Wasserstein distance matrices for MV grids do not exist' + '\n Please, compute them by setting compute_mv_distances = True!')
        distances_mv = {}
        distances_mv['voltage'] = pd.read_pickle(os.path.join(context_path, 'MV_voltage_wasserstein.pkl'))
        distances_mv['demand'] = pd.read_pickle(os.path.join(context_path, 'MV_demand_wasserstein.pkl'))
        distances_mv['z_vec'] = pd.read_pickle(os.path.join(context_path, 'MV_z_distances.pkl'))

    compute_lv_distances = False
    random_selection = 25000

    if compute_lv_distances:
        folder_lv = os.path.join(grids_path, 'PDGs_LV/')
        regions_lv = ['Alps-Periurban', 'Alps-Rural', 'Alps-Urban', 'Jura-Periurban', 'Jura-Rural', 'Jura-Urban', 'Midlands-Periurban', 'Midlands-Rural', 'Midlands-Urban']
        # we check which files are in the folders
        files_lv = {region: os.listdir(os.path.join(folder_lv, region)) for region in regions_lv}
        grids_lv = {region: [f.replace('_edges','') for f in files_lv[region] if 'edges' in f] for region in regions_lv}


        jobs = [(os.path.join(folder_lv, region,''), lv) for region in regions_lv for lv in grids_lv[region] if lv in grid_names['LV'].values()]
        # we get a random sample of 5000 grids without replacement

        jobs = [jobs[i] for i in np.random.choice(range(len(jobs)), random_selection, replace=False)]

        print("Getting grid properties")
        n_cores = cpu_count() - 2
        with Pool(n_cores) as p:
            results_lv = p.starmap(get_grid_properties, jobs)
        p.close()
        grid_properties_lv = unpack_grid_properties(results_lv)

        print("Computing the CDFs")
        print("The process might take a while...")
        cdf_demand_range_lv = np.linspace(0, 0.1, 100)
        cdf_voltage_range_lv = np.linspace(0.97, 1, 100)
        cdf_voltage_lv = {g: grid_properties_lv[g]['cdf_voltage'](cdf_voltage_range_lv) for g in grid_properties_lv.keys()}

        start_time = time.time()
        W1_voltage_lv = W1_pairwise(cdf_voltage_lv, cdf_voltage_range_lv)
        W1_voltage_lv.to_pickle(os.path.join(context_path, 'LV_voltage_wasserstein.pkl'))
        end_time = time.time()
        print("Computing time: ", end_time - start_time)
        del cdf_voltage_lv, W1_voltage_lv
        gc.collect()

        start_time = time.time()
        cdf_demand_lv = {g: grid_properties_lv[g]['cdf_demand'](cdf_demand_range_lv) for g in grid_properties_lv.keys()}
        W1_demand_lv = W1_pairwise(cdf_demand_lv, cdf_demand_range_lv)
        W1_demand_lv.to_pickle(os.path.join(context_path, 'LV_demand_wasserstein.pkl'))
        end_time = time.time()
        print("Computing time: ", end_time - start_time)
        del cdf_demand_lv, W1_demand_lv
        gc.collect()

        grid_ids = list(grid_properties_lv.keys())
        Dgrids_lv = np.array([grid_properties_lv[grid_ids[i]]['total_demand'] for i in range(len(grid_ids))])
        Ngrids_lv = np.array([grid_properties_lv[grid_ids[i]]['total_nodes'] for i in range(len(grid_ids))])
        Dz_lv = (Dgrids_lv - Dgrids_lv.min())/(Dgrids_lv.max() - Dgrids_lv.min())
        Nz_lv = (Ngrids_lv - Ngrids_lv.min())/(Ngrids_lv.max() - Ngrids_lv.min())
        Z_lv = np.array([[Dz_lv[i], Nz_lv[i]] for i in range(Dz_lv.size)])
        # we compute the pairwise distances between the grids
        Z_distances_lv = pd.DataFrame(pairwise_distances(Z_lv), index=grid_ids, columns=grid_ids)
        # we export the distances
        Z_distances_lv.to_pickle(os.path.join(context_path, 'LV_z_distances.pkl'))
        del Z_distances_lv
        gc.collect()

    else:
        # we check if the Wasserstein distance matrices exist, and if not, we raise a warning
        if not os.path.exists(os.path.join(context_path, 'LV_voltage_wasserstein.pkl')) \
            or not os.path.exists(os.path.join(context_path, 'LV_demand_wasserstein.pkl')) \
                or not os.path.exists(os.path.join(context_path, 'LV_z_distances.pkl')):
            raise Warning('The Wasserstein distance matrices for LV grids do not exist' + '\n Please, compute them by setting compute_lv_distances = True!')
        # we read the Wasserstein distance matrices
        distances_lv = {}
        distances_lv['voltage'] = pd.read_pickle(os.path.join(context_path, 'LV_voltage_wasserstein.pkl'))
        distances_lv['demand'] = pd.read_pickle(os.path.join(context_path, 'LV_demand_wasserstein.pkl'))
        distances_lv['z_vec'] = pd.read_pickle(os.path.join(context_path, 'LV_z_distances.pkl'))