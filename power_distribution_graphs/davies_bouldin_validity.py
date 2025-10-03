"""Context-aware validity tools for clustering power distribution graphs.

The module gathers helper routines used during the power-distribution graph
experiments.  In addition to computing Davies-Bouldin indices for different
distance representations, it provides utilities to:

* derive feature maps from Wasserstein-based kernels,
* compute pairwise distances between grid-level empirical CDFs, and
* orchestrate end-to-end evaluations from the command line.

The main block at the end mirrors the workflow described in the repository's
documentation and can be adapted to recompute the intermediate artefacts when
new grids are added.
"""
import os
import sys

# The BayesOpt helpers live outside of this package; extend ``sys.path`` so the
# local imports remain valid when the script is executed as ``__main__``.
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
from wass_kernel_clustering import center_kernel, nystrom_map, PCA_map, kernel_PCA, obtain_grid_dissimilarity
from scipy.spatial.distance import squareform
from sklearn_extra.cluster import KMedoids
import math
import gc

def N_KMedoids(X, clusters, grid_type, n_init=3, n_fgk=100, m_fgk=35):
    """Run several K-Medoids restarts and keep the best Goodman-Kruskal score.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature representation of the grids, either the original embeddings or
        the feature map extracted from a kernel matrix.
    clusters : int
        Number of clusters to form.
    grid_type : {"MV", "LV"}
        Selects the solver used by :class:`~sklearn_extra.cluster.KMedoids`.
        Medium-voltage grids (``MV``) use the PAM solver, whereas low-voltage
        grids (``LV``) rely on the alternate solver which is more memory
        efficient for larger samples.
    n_init : int, default=3
        Number of clustering restarts.  The run with the highest
        Goodman-Kruskal index is returned.
    n_fgk : int, default=100
        Number of pairs used when estimating the FGK index.
    m_fgk : int, default=35
        Number of repetitions of the FGK estimation.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster assignments corresponding to the best FGK score.
    medoids : list[int]
        Indices of the medoids associated with ``labels``.
    """

    labels, max_fgk_index, medoids = [], -1, []
    # Run K-Medoids ``n_init`` times and track the configuration with the best
    # FGK score.
    for _ in range(n_init):
        # Create an instance of K-Medoids with the desired number of clusters.
        if grid_type == 'MV':
            clustering_model = KMedoids(n_clusters=clusters, init='k-medoids++', method='pam').fit(X)
        elif grid_type == 'LV':
            clustering_model = KMedoids(n_clusters=clusters, init='k-medoids++', method='alternate').fit(X)

        # Validate that the run produced the requested number of clusters before
        # computing the FGK index.
        if np.unique(clustering_model.labels_).shape[0] == clusters:
            fgk = FGK(n=n_fgk, m=m_fgk)
            fgk.fit(X, clustering_model.labels_)
            if fgk.GK_ > max_fgk_index:
                max_fgk_index = fgk.GK_
                labels = clustering_model.labels_
                medoids = clustering_model.medoid_indices_

    return labels, medoids

def composed_kernel(dissimilarities, gammas, kernel_shift=1e-3, nys_n_samples=5000):
    """Build a composite kernel from heterogeneous dissimilarities.

    Parameters
    ----------
    dissimilarities : dict[str, np.ndarray]
        Mapping with the entries ``'wasserstein_dissimilarity'``,
        ``'demand_dissimilarity'`` and ``'nodes_dissimilarity'``.  Each value
        can be either a condensed vector or a square distance matrix.
    gammas : dict[str, float]
        Exponential kernel parameters optimised during Bayesian search.
    kernel_shift : float, default=1e-3
        Small positive value added to the diagonals to prevent numerical
        singularities when centring the kernel.
    nys_n_samples : int, default=5000
        Number of samples used for the Nyström approximation.  When the number
        of grids is smaller, the feature map is obtained through Kernel PCA.

    Returns
    -------
    np.ndarray
        Centred feature map ready to be fed into a clustering routine.
    """

    # ``squareform`` expects condensed vectors; if the dissimilarity matrices are
    # still square we convert them first.
    if dissimilarities['demand_dissimilarity'].ndim == 2:
        dissimilarities['demand_dissimilarity'] = squareform(dissimilarities['demand_dissimilarity'])
    if dissimilarities['nodes_dissimilarity'].ndim == 2:
        dissimilarities['nodes_dissimilarity'] = squareform(dissimilarities['nodes_dissimilarity'])
    if dissimilarities['wasserstein_dissimilarity'].ndim == 2:
        dissimilarities['wasserstein_dissimilarity'] = squareform(dissimilarities['wasserstein_dissimilarity'])

    # Build Gaussian kernels for each distance representation.
    ker_wass = np.exp(- math.pow(10, gammas['gamma_wasserstein']) * dissimilarities['wasserstein_dissimilarity'], dtype=np.float32)
    ker_demand = np.exp(- math.pow(10, gammas['gamma_demand']) * dissimilarities['demand_dissimilarity'], dtype=np.float32)
    ker_nodes = np.exp(- math.pow(10, gammas['gamma_nodes']) * dissimilarities['nodes_dissimilarity'], dtype=np.float32)

    # Combine the kernels and restore the square form required by the downstream
    # routines.
    composed_kernel = squareform(np.multiply(ker_wass, ker_demand + ker_nodes))
    # Free intermediate arrays before centring to limit peak memory usage.
    del ker_wass, ker_demand, ker_nodes
    gc.collect()
    # The diagonal needs to reflect the operations performed when combining the
    # three kernels.
    np.fill_diagonal(composed_kernel, (1 + kernel_shift) * (2 + 2 * kernel_shift))

    # Centre the composed kernel to obtain an implicit feature map with zero mean.
    center_composed_kernel = center_kernel(composed_kernel)
    del composed_kernel
    gc.collect()

    # Use Nyström when the number of samples exceeds the requested basis size.
    if nys_n_samples < center_composed_kernel.shape[0]:
        phi_wass = nystrom_map(center_composed_kernel, sampling=nys_n_samples)
        del center_composed_kernel
        gc.collect()
        # Apply PCA in the Nyström feature space to obtain an orthogonal basis.
        phi_wass = PCA_map(phi_wass)
    else:
        # For smaller datasets Kernel PCA can be applied directly.
        phi_wass = kernel_PCA(center_composed_kernel)
        del center_composed_kernel
        gc.collect()

    return phi_wass

def get_grid_dissimilarities(grid_type, _embeddings_path, _wass_path):
    """Load demand, node-count and Wasserstein dissimilarities for a grid type.

    Parameters
    ----------
    grid_type : {"MV", "LV"}
        Grid family whose dissimilarities should be retrieved.
    _embeddings_path : str | os.PathLike
        Directory containing the node embeddings and the canonical grid order.
    _wass_path : str | os.PathLike
        Directory with the serialized Wasserstein models.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Wasserstein distances together with pairwise absolute differences in
        total demand and node counts.
    """

    # Load the canonical ordering of grids used across the experiments.
    with open(os.path.join(_embeddings_path, 'normalized_grids_names_{}.pickle'.format(grid_type)), 'rb') as handle:
        dict_names = pickle.load(handle)

    node_embeddings = pd.read_csv(os.path.join(_embeddings_path, 'node_embeddings_{}.csv'.format(grid_type)))
    # Compute the dissimilarity matrices based on total demand and node counts.
    demand_disimilarity_matrix, nodes_disimilarity_matrix = obtain_grid_dissimilarity(node_embeddings, dict_names)

    # Drop the frame to reduce peak memory usage.
    del node_embeddings
    gc.collect()

    # The Wasserstein distances are stored inside a helper object; extract the
    # matrix and release the model afterwards.
    with open(os.path.join(_wass_path, 'wass_model_{}.pickle'.format(grid_type)), 'rb') as handle:
        wass_grids = pickle.load(handle)

    wass_distance = wass_grids.min_distances_.copy()
    # Retrieve the distances from the object and release the heavy structure.
    del wass_grids
    gc.collect()
    return wass_distance, demand_disimilarity_matrix, nodes_disimilarity_matrix

def get_grid_properties(folder_path, grid_name):
    """Read node data for a grid and summarise demand/voltage statistics.

    Parameters
    ----------
    folder_path : str | os.PathLike
        Directory containing the shapefile for the grid.  The file is expected
        to follow the ``<grid_name>_nodes`` naming convention.
    grid_name : str
        Identifier of the grid whose properties will be extracted.

    Returns
    -------
    dict
        Empirical CDFs for voltage and demand together with aggregated
        statistics used later in the pipeline.
    """
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
    """Transform a list of grid-property dictionaries into a lookup table.

    Parameters
    ----------
    grid_results : list[dict]
        Output of :func:`get_grid_properties` executed across multiple grids.

    Returns
    -------
    dict[str, dict]
        Mapping from grid identifier to the dictionary returned by
        :func:`get_grid_properties` (minus the redundant ``grid_id`` key).
    """
    return {
        grid_results[j]['grid_id']: {
            k: grid_results[j][k] for k in grid_results[j] if k != 'grid_id'
        }
        for j in range(len(grid_results))
    }

def W1_cdf_based(cdf1, cdf2, x_range):
    """Approximate the 1-Wasserstein distance between two empirical CDF values.

    Parameters
    ----------
    cdf1, cdf2 : np.ndarray
        Values of the empirical CDFs evaluated at ``x_range``.
    x_range : np.ndarray
        Support on which the CDFs were evaluated.  The spacing is assumed to be
        uniform which allows approximating the integral by a Riemann sum.
    """
    return np.sum(np.abs(cdf1 - cdf2)) * (x_range[-1] - x_range[0]) / x_range.size

def W1_pairwise(cdf_dict, x_range):
    """Compute all pairwise 1-Wasserstein distances between CDFs in ``cdf_dict``.

    Parameters
    ----------
    cdf_dict : dict[str, np.ndarray]
        Lookup table mapping grid identifiers to empirical CDF evaluations.
    x_range : np.ndarray
        Support on which the CDFs were evaluated.

    Returns
    -------
    pandas.DataFrame
        Symmetric matrix with the pairwise Wasserstein-1 distances.
    """
    list_keys = list(cdf_dict.keys())
    W1_df = pd.DataFrame(index=list_keys, columns=list_keys)
    W1_values = np.zeros((len(list_keys), len(list_keys)))
    # As the distance matrix is symmetric, only compute the upper triangular part.
    pairs_to_compute = [(i, j) for i in range(len(list_keys)) for j in range(len(list_keys)) if j > i]
    jobs = [(cdf_dict[list_keys[p[0]]], cdf_dict[list_keys[p[1]]], x_range) for p in pairs_to_compute]
    # Use multiprocessing to evaluate the Wasserstein distances in parallel.
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
    """Compute the Davies-Bouldin index from arbitrary pairwise distances.

    Parameters
    ----------
    in_clusters : dict[str, list[str]]
        Mapping from medoid identifiers to the grids assigned to them.
    pairwise_dist : pandas.DataFrame
        Pairwise distance matrix between grids.  The index and columns must
        contain all grid identifiers referenced in ``in_clusters``.
    denominator_eps : float, default=1e-10
        Jitter added to the denominator when computing ratios to avoid divisions
        by zero.

    Returns
    -------
    float
        Davies-Bouldin index computed with the provided distances.
    """

    average_distance_to_medoid = {}
    keys_in_sampled_distances = pairwise_dist.keys()
    in_clusters = {k: [j for j in in_clusters[k] if j in keys_in_sampled_distances] for k in in_clusters}
    for cluster in in_clusters:
        cluster_distances = pairwise_dist[in_clusters[cluster]].loc[in_clusters[cluster]].copy()
        distances = cluster_distances.sum(axis=1)
        # Identify the medoid as the element with minimum total distance.
        medoid = distances.idxmin()
        # ``S_i``: average distance from members of the cluster to their medoid.
        average_distance_to_medoid[medoid] = cluster_distances.loc[medoid].mean()

    intracluster_distances = pairwise_dist[average_distance_to_medoid.keys()].loc[average_distance_to_medoid.keys()].copy()
    # Build the matrix R = (S_i + S_j) / M_ij where ``M_ij`` is the distance
    # between medoids ``i`` and ``j``.
    R_matrix = np.zeros((len(average_distance_to_medoid), len(average_distance_to_medoid)))
    for i, med_i in enumerate(average_distance_to_medoid):
        for j, med_j in enumerate(average_distance_to_medoid):
            if j > i:
                R_matrix[i, j] = R_matrix[j, i] = \
                    (average_distance_to_medoid[med_i] + average_distance_to_medoid[med_j]) \
                    / (intracluster_distances.loc[med_i, med_j] + denominator_eps)
            elif j == i:
                R_matrix[i, j] = 0

    # ``D_i``: maximum similarity between cluster ``i`` and any other cluster.
    max_R_values = np.max(R_matrix, axis=1)
    # The Davies–Bouldin index is the average of ``D_i`` across clusters.
    davies_bouldin_index = np.mean(max_R_values)
    return davies_bouldin_index

def compute_wass_distances(grid_type, random_selection = 25000):
    """Compute Wasserstein-based dissimilarities directly from the raw data.

    Parameters
    ----------
    grid_type : {"MV", "LV"}
        Grid family to process.
    random_selection : int, default=25000
        When ``grid_type`` is ``LV`` the dataset is large; this parameter
        controls the number of grids sampled without replacement.

    Returns
    -------
    None
        The resulting distance matrices are stored under ``context_validity``.
    """
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
        # Gather the grids available for each region and sample a manageable
        # subset to keep the computations tractable.
        files_lv = {region: os.listdir(os.path.join(folder_lv, region)) for region in regions_lv}
        grids_lv = {region: [f.replace('_edges','') for f in files_lv[region] if 'edges' in f] for region in regions_lv}
        jobs = [(os.path.join(folder_lv, region,''), lv) for region in regions_lv for lv in grids_lv[region] if lv in grid_names['LV'].values()]
        # Draw a random sample of grids without replacement.
        jobs = [jobs[i] for i in np.random.choice(range(len(jobs)), random_selection, replace=False)]

    print("Getting grid properties")
    n_cores = cpu_count() - 2
    with Pool(n_cores) as p:
        results = p.starmap(get_grid_properties, jobs)
    p.close()
    grid_properties = unpack_grid_properties(results)

    print("Computing the CDFs")
    if grid_type == 'MV':
        # The support is already known for MV grids.
        cdf_demand_range = np.linspace(0, 4, 100)
        cdf_voltage_range = np.linspace(0.98, 1, 100)
    elif grid_type == 'LV':
        print("The process might take a while...")
        # Predefined ranges for LV grids reflecting the different scale.
        cdf_demand_range = np.linspace(0, 0.1, 100)
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
    # Compute Euclidean distances between the normalised demand/node features.
    Z_distances = pd.DataFrame(pairwise_distances(Z), index=grid_ids, columns=grid_ids)
    # Persist the distances for later reuse.
    Z_distances.to_pickle(os.path.join(context_path, '{}_z_distances.pkl'.format(grid_type)))
    del Z_distances
    gc.collect()

    return

def read_wass_distances(grid_type):
    """Load precomputed Wasserstein and auxiliary distance matrices.

    Parameters
    ----------
    grid_type : {"MV", "LV"}
        Grid family to retrieve distances for.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary containing the voltage Wasserstein distances, demand
        Wasserstein distances and the ``z``-vector Euclidean distances.
    """
    if grid_type not in ['MV', 'LV']:
        raise ValueError('grid_type must be either MV or LV')
    # Ensure the expected artefacts exist before attempting to load them.
    if not os.path.exists(os.path.join(context_path, '{}_voltage_wasserstein.pkl'.format(grid_type))) \
        or not os.path.exists(os.path.join(context_path, '{}_demand_wasserstein.pkl'.format(grid_type))) \
            or not os.path.exists(os.path.join(context_path, '{}_z_distances.pkl'.format(grid_type))):
        raise Warning('The Wasserstein distance matrices for LV grids do not exist' + '\n Please, compute them by setting compute_lv_distances = True!')
    # Read the Wasserstein distance matrices and return them as a dictionary.
    distances = {}
    distances['voltage'] = pd.read_pickle(os.path.join(context_path, '{}_voltage_wasserstein.pkl'.format(grid_type)))
    distances['demand'] = pd.read_pickle(os.path.join(context_path, '{}_demand_wasserstein.pkl'.format(grid_type)))
    distances['z_vec'] = pd.read_pickle(os.path.join(context_path, '{}_z_distances.pkl'.format(grid_type)))

    return distances

def cluster_grids_feature_maps(_grid_type, _files_results, _copy_number, _clustering_path, _clustered_grids_path, _embeddings_path, _wass_path):
    """Cluster grids using the feature maps associated with a copy of the run.

    Parameters
    ----------
    _grid_type : {"MV", "LV"}
        Grid family to cluster.
    _files_results : list[str]
        Files available in the ``clustering`` directory for ``_grid_type``.
    _copy_number : str
        Identifier of the optimisation run (e.g. ``"(0)"``) whose parameters
        should be used.
    _clustering_path : str | os.PathLike
        Directory containing the clustering results.
    _clustered_grids_path : str | os.PathLike
        Directory where the clustering assignments will be stored.
    _embeddings_path : str | os.PathLike
        Directory containing the node embeddings and the canonical grid order.
    _wass_path : str | os.PathLike
        Directory with the serialized Wasserstein models.

    Returns
    -------
    None
        Cluster assignments are written to ``context_validity/clustered_grids``.
    """
    # Separate the CSV files corresponding to the requested copy.
    csv_files_results = [file for file in _files_results if file.endswith('.csv') and _copy_number in file]
    # Load the CSV files storing the optimised gamma values.
    indices_results = {file.split('.')[0]: pd.read_csv(os.path.join(_clustering_path, _grid_type, file)) for file in csv_files_results}
    # Read the grid names from the embeddings directory to map indices to labels.
    grid_names = pickle.load(open(os.path.join(_embeddings_path, 'normalized_grids_names_{}.pickle'.format(_grid_type)), 'rb'))

    # Identify which optimisation locations are present in the CSV files.
    test_loc = [test for test in range(0,10) if test in indices_results['results - Copy {0}'.format(_copy_number)].index]

    dissimilarities_grids = {'wasserstein_dissimilarity': [], 'demand_dissimilarity': [], 'nodes_dissimilarity': []}

    results = get_grid_dissimilarities(_grid_type, _embeddings_path, _wass_path)
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
        # Export the labels, replacing cluster ids by the actual medoid grid names.
        clustered_grids = pd.DataFrame({'grid_id': [grid_names[i] for i in range(len(grid_names))], 'cluster': labels})
        map_medoid = {i: grid_names[medoids[i]] for i in range(len(medoids))}
        clustered_grids['cluster'] = clustered_grids['cluster'].map(map_medoid)
        clustered_grids.to_csv(os.path.join(_clustered_grids_path, 'clustered_grids_{0}_{1} - Copy {2}.csv'.format(_grid_type, str(test),_copy_number)), index=False)
    return

def clusters_davies_bouldin_relative_index(_grid_type, _cluster_grids, _distances, _K_clusters):
    """Compare Davies-Bouldin indices across multiple distance representations.

    Parameters
    ----------
    _grid_type : {"MV", "LV"}
        Grid family being analysed.
    _cluster_grids : list[str]
        Identifiers of the optimisation copies whose clustering outputs should
        be incorporated in the analysis.
    _distances : dict[str, pandas.DataFrame]
        Dictionary of distance matrices as returned by
        :func:`read_wass_distances`.
    _K_clusters : int
        Number of clusters used when computing the Davies-Bouldin indices.

    Returns
    -------
    None
        Normalised Davies-Bouldin scores are exported to
        ``context_validity``.
    """

    davies_bouldin_values = { }
    distances_types = ['voltage', 'demand', 'z_vec']

    start_time = time.time()
    clusters = {}
    for distance_typ in distances_types:
        print("------------------------------------")
        print("Computing Davies-Bouldin index for {} grids with distance type: ".format(_grid_type), distance_typ)
        # Run K-medoids on the selected distance matrix.
        grids_in_index = _distances[distance_typ].index.tolist()
        distances_values = _distances[distance_typ].values
        kmedoids = KMedoids(n_clusters=_K_clusters, metric='precomputed', method='alternate', random_state=0).fit(distances_values)
        labels = kmedoids.labels_
        medoid_indices = kmedoids.medoid_indices_
        clusters[distance_typ] = {grids_in_index[medoid_indices[k]]: [grids_in_index[i] for i in np.where(labels == k)[0]] for k in range(_K_clusters)}
        davies_bouldin_values[distance_typ] = {distance_to_compare: \
                davies_bouldin_distance_based(clusters[distance_typ], _distances[distance_to_compare]) for distance_to_compare in _distances}
    end_time = time.time()
    print("Computing time: ", end_time - start_time)

    for copy_number in _cluster_grids:
        test_loc = [test for test in range(0,10) if test in \
            pd.read_csv(os.path.join(clustering_path, _grid_type, 'results - Copy {0}.csv'.format(copy_number))).index]
        for loc in test_loc:
            # Reuse the previously exported cluster assignments when available.
            if not os.path.exists(os.path.join(clustered_grids_path, "clustered_grids_{0}_{1} - Copy {2}.csv".format(_grid_type, str(loc), copy_number))):
                raise Warning('The file clustered_grids_{0}_{1} - Copy {2}.csv does not exist'.format(_grid_type, str(loc), copy_number) \
                              + '\n Please, compute the clustering with cluster_grids_feature_maps function!')
            clusters_grids = pd.read_csv(os.path.join(clustered_grids_path, "clustered_grids_{0}_{1} - Copy {2}.csv".format(_grid_type, str(loc), copy_number)))
            keys_in_clusters = clusters_grids.groupby('cluster')['grid_id'].apply(list).to_dict()
            davies_bouldin_values['wasserstein_{}'.format(loc)] = {distance_to_compare: \
                davies_bouldin_distance_based(keys_in_clusters, _distances[distance_to_compare]) for distance_to_compare in _distances}

        # Convert the dictionary of scores into a dataframe for normalisation.
        davies_bouldin_values = pd.DataFrame(davies_bouldin_values).T
        davies_bouldin_values = (davies_bouldin_values - davies_bouldin_values.min()) / (davies_bouldin_values.max() - davies_bouldin_values.min())
        davies_bouldin_values = davies_bouldin_values.T
        # Summarise the behaviour across representations.
        mean_values, max_values = pd.DataFrame(davies_bouldin_values.mean()).T, pd.DataFrame(davies_bouldin_values.max()).T
        mean_values.index, max_values.index  = ['mean'], ['max']
        davies_bouldin_values = pd.concat([davies_bouldin_values, mean_values, max_values])
        davies_bouldin_values.to_csv(os.path.join(context_path, '{0}_davies_bouldin_values - Copy {1}.csv'.format(_grid_type, copy_number)))

# Execute the workflow only when the module is called as a script.
if __name__ == '__main__':

    # Resolve paths relative to this file.  The directory structure mirrors the
    # one described in the repository README.
    embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings')
    grids_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    context_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'context_validity')
    clustering_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clustering')
    wass_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasserstein_models')
    if not os.path.exists(os.path.join(context_path, 'clustered_grids')):
        os.makedirs(os.path.join(context_path, 'clustered_grids'))
    clustered_grids_path = os.path.join(context_path, 'clustered_grids')
    # Ensure the required artefacts exist before proceeding.
    if not os.path.exists(embeddings_path):
        raise Warning('The folder embeddings does not exist' + '\n Please, create the embeddings with node_embeddings.py!')
    if not os.path.exists(grids_path):
        raise Warning('The folder data does not exist' + '\n Please, download the data from the data repository!')

    # Create the directory used to store validity results when needed.
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
                cluster_grids_feature_maps(grid_type, files_results, copy_number, clustering_path, clustered_grids_path, embeddings_path, wass_path)

    K_clusters = 10
    compute_davies_bouldin = True

    # Compute the Davies–Bouldin index for the clustering results if requested.
    if compute_davies_bouldin:
        clusters_davies_bouldin_relative_index('MV', cluster_grids['MV'], distances_mv, K_clusters)
        clusters_davies_bouldin_relative_index('LV', cluster_grids['LV'], distances_lv, K_clusters)
