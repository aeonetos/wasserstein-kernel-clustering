import codecs
import sys
import os
sys.path.insert(0, os.path.abspath('../BayesOpt/'))
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import contingency_matrix
from scipy.interpolate import interp1d
import pandas as pd
import time
from tqdm import tqdm
from validity_measures import FGK, consensus_index
from scipy.spatial.distance import squareform
from sklearn_extra.cluster import KMedoids
import math
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from functools import partial

def read_data(_filename):
    """
    Read data from a file and preprocess it.

    :param filename: The name of the file to read.
    :return: A tuple containing the data and labels.

    The data has the time series information for each sample.
    The labels are given by the dataset and correspond to the target variable.
    """
    dataFile = codecs.open(_filename)
    text = dataFile.readlines()
    
    data = []
    labels = []

    for line in text:
        new = [float(x) for x in line.split()[1:]]
        data.append(new)
        labels.append(float(line.split()[0]))

    data = np.array(data)
    ys = np.array(labels, dtype=int)
    return data, ys

def get_fourier(_normalized_data, _welch=False):
    """
    Process the data to obtain its NPSD and the RMS of each time series.

    :param _data: The input time series data.
    :param _welch: The flag indicating whether to use Welch's method for computing the PSD.
    :return: A tuple containing the (normalized) power spectral densities and their support.

    The data has the time series information for each sample.
    The normalized power spectral densities are computed from the periodogram.
    The support is the frequency bins corresponding to the power spectral densities.
    """

    if _welch:
        support, psds = signal.welch(_normalized_data, fs=1, nperseg=24, noverlap=12)
    else:
        support, psds = signal.periodogram(_normalized_data)
    # the constant term for the periodogram is removed
    support = support[1:]
    psds = psds[:, 1:]
    # the periodogram is normalized to 1
    normalize = np.sum(psds, axis=1)
    psds += 1e-20
    psds = (psds.transpose() / normalize).transpose()

    return psds, support

def normalize_data(_data):
    """
    Normalize the data to [0, 1]

    :param _data: The input data to be normalized.
    :return: The normalized data.

    The data has the time series information for each sample.
    The normalization is done for the entire dataset.
    """
    return (_data - np.min(_data)) / (np.max(_data) - np.min(_data))

def reduce_variance(_normalized_data, _variance=0.9):
    """
    Reduce the variance of the data using PCA.

    :param _normalized_data: The input normalized data.
    :param _variance: The target variance to retain.
    :return: The data with reduced variance.

    The data is projected onto a lower-dimensional space using PCA,
    and then reprojected back to the original space.
    """
    # we perform PCA on norm_data
    pca = PCA(n_components=_variance)
    # we retransform the data to the original space
    return pca.inverse_transform(pca.fit_transform(_normalized_data))

def inverse_histogram(_mu, _S, _Sinv=np.linspace(0, 1, 1000), _epsilon=1e-14,_method='linear'):
    """

    Compute the inverse histogram of a given distribution.
    
    Parameters
    ----------

    _mu     : histogram
    _S      : support of the histogram
    _Sinv   : support of the quantile function
    _epsilon: small value to avoid division by zero
    _method : name of the interpolation method (linear, quadratic, ...)

    Returns
    -------
    _Sinv    : support of the quantile function
    _q_Sinv  : values of the quantile function
    
    """
    A = _mu>_epsilon
    A[-1] = 0
    Sa = _S[A]

    cdf = np.cumsum(_mu)
    cdfa = cdf[A]
    if (cdfa[-1] == 1):
        cdfa[-1] = cdfa[-1] - _epsilon

    cdfa = np.append(0, cdfa)
    cdfa = np.append(cdfa, 1)

    if _S[0] < 0:
        Sa = np.append(_S[0]-1, Sa)
    else:
        # set it to zero in case of PSDs
        Sa = np.append(0, Sa)
    Sa = np.append(Sa, _S[-1])

    q = interp1d(cdfa, Sa, kind=_method)
    q_Sinv = q(_Sinv)
    return _Sinv, q_Sinv

def get_inverse_histograms(_mus, _S, _Sinv=np.linspace(0, 1, 1000), _epsilon=1e-14,_method='linear'):
    """

    Compute the inverse histogram of a set of distributions.
    
    Parameters
    ----------

    _mus    : set of histograms
    _S      : support of the histograms
    _Sinv   : support of the quantile function
    _epsilon: small value to avoid division by zero
    _method : name of the interpolation method (linear, quadratic, ...)

    Returns
    -------
    _Sinv     : support of the quantile functions
    _q_Sinvs  : values of the quantile functions
    
    """

    _q_Sinvs = []
    for _mu in _mus:
        _, q_Sinv = inverse_histogram(_mu, _S, _Sinv, _epsilon, _method)
        _q_Sinvs.append(q_Sinv)
    return _Sinv, np.array(_q_Sinvs)

def purity_score(y_true, y_pred):
    """
    Calculates the purity score for a clustering result.

    Args:
        y_true (list or np.array): The ground truth labels.
        y_pred (list or np.array): The predicted cluster labels.

    Returns:
        float: The purity score, between 0.0 and 1.0.
    """
    # Create the contingency matrix
    cm = contingency_matrix(y_true, y_pred)
    
    # For each cluster, find the number of samples from the most common class
    # and sum them up.
    purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    
    return purity

def retrieve_results(optimizer, iter_type, CI_index, EC, FGK_index, retained_var):
    """
    Retrieve the results from the optimizer and organize them into a DataFrame.
    
    Parameters
    ----------
    optimizer : object
        The optimizer object containing the results.
    iter_type : list
        The iteration types for each result.
    CI_index : list
        The consensus index for every clustering iteration.
    EC : list
        The effective number of constituents.
    FGK_index : list
        The FGK indices for each result.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the organized results, sorted by the target values.
    """
    gammas = [res['params'] for res in optimizer.res]
    gammas_keys = list(gammas[0].keys())
    opt_targets = [res['target'] for res in optimizer.res]
    results = {'iteration_type': [], 'CI': [], 'min_ec': [], 'FGK_index': [], 'Retained_variance': [], 'Target': []}
    for gamma in gammas_keys:
        results[gamma] = []
    for i, gamma in enumerate(gammas):
        for g in gammas_keys:
            results[g].append(gamma[g])
        results['iteration_type'].append(iter_type[i])
        results['CI'].append(CI_index[i])
        results['min_ec'].append(EC[i])
        results['FGK_index'].append(FGK_index[i])
        results['Target'].append(opt_targets[i])
        results['Retained_variance'].append(retained_var[i])
    df_results = pd.DataFrame(results)
    df_results.sort_values(by=['Target'], ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)
    return df_results

def run_iteration(opt_obj, util_fun, black_box_fun, n_clusters, size_objective):
    """
    Run a single iteration of the optimization process.

    Parameters
    ----------
    opt_obj : object
        The optimization object.
    util_fun : callable
        The utility function to optimize.
    black_box_fun : callable
        The black box function to evaluate.
    n_clusters : int
        The number of clusters.
    size_objective : bool
        Whether to use the size objective.

    Returns
    -------
    tuple
        A tuple containing the results of the iteration.
    """
    start_time = time.time()
    next_point = opt_obj.suggest(util_fun)
    CI_arr, min_ec_arr, FGK_index_arr, retained_variance = black_box_fun(**next_point)
    if size_objective:
        target = np.min([(min_ec_arr - 1)/(n_clusters - 1), (FGK_index_arr + 1)/2, CI_arr])
    else:
        target = np.min([(FGK_index_arr + 1)/2, CI_arr])
    end_time = time.time()
    return next_point, CI_arr, min_ec_arr, FGK_index_arr, retained_variance, target, end_time - start_time

def optimize_kernels(black_box_fun, opt_obj, util_fun, bayes_init, bayes_iter, n_clusters, size_objective=False):
    """
    Optimize the kernel functions using Bayesian optimization.

    Parameters
    ----------
    black_box_fun : callable
        The black box function to evaluate.
    opt_obj : object
        The optimization object.
    util_fun : callable
        The utility function to optimize.
    bayes_init : int
        The number of initial random iterations.
    bayes_iter : int
        The number of Bayesian iterations.
    n_clusters : int
        The number of clusters.
    size_objective : bool
        Whether to use the size objective.

    Returns
    -------
    tuple
        A tuple containing the results of the optimization process.
    """
    init_variables, CI_index, EC, FGK_index, retained_var, init_targets, bayes_init_time, bayes_iter_time,\
          iter_type = [], [], [], [], [], [], [], [], []
    tqdm.write("Random point iterations:")
    # Initial random iterations
    for _ in tqdm(range(bayes_init)):
        next_point, CI_arr, min_ec_arr, FGK_index_arr, variance, target, elapsed =\
              run_iteration(opt_obj, util_fun, black_box_fun, n_clusters, size_objective)
        CI_index.append(CI_arr)
        EC.append(min_ec_arr)
        FGK_index.append(FGK_index_arr)
        retained_var.append(variance)
        init_variables.append(next_point)
        init_targets.append(target)
        iter_type.append('random')
        bayes_init_time.append(elapsed)
    # Register initial points
    for i in range(bayes_init):
        opt_obj.register(params=init_variables[i], target=init_targets[i])
    # Bayesian iterations
    tqdm.write("Bayes point iterations:")
    for _ in tqdm(range(bayes_iter)):
        next_point, CI_arr, min_ec_arr, FGK_index_arr, variance, target, elapsed =\
              run_iteration(opt_obj, util_fun, black_box_fun, n_clusters, size_objective)
        CI_index.append(CI_arr)
        EC.append(min_ec_arr)
        FGK_index.append(FGK_index_arr)
        retained_var.append(variance)
        opt_obj.register(params=next_point, target=target)
        iter_type.append('bayes')
        bayes_iter_time.append(elapsed)
    dict_execution_time = {'init_time': bayes_init_time, 'iter_time': bayes_iter_time}
    return opt_obj, iter_type, CI_index, EC, FGK_index, retained_var, dict_execution_time

def optimal_dispersion_kernel(distance_vector, test_range = (-4, 2, 25)):
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
    # We obtain the centered kernel
    centering_vec_kernel = np.mean(kernel, axis=0, keepdims=True)
    centering_mean_kernel = np.mean(kernel)
    return kernel - centering_vec_kernel - centering_vec_kernel.T + centering_mean_kernel

def kernel_PCA(kernel, _variance=-1, _min_singular_value=1e-8, _diagonal_shift=1e-3):
    """
    Perform Kernel PCA on the given kernel matrix.

    :param kernel: The kernel matrix to perform PCA on.
    :param _variance: The target explained variance ratio (default is -1, which means the Kaiser criterion is used).
    :param _min_singular_value: The minimum singular value to consider (default is 1e-8).
    :return: The feature map derived from the kernel PCA.
    """
    pca = PCA()
    # Fit the PCA model to your data
    kernel = kernel + _diagonal_shift * np.eye(kernel.shape[0])
    pca.fit(kernel)
    explained = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    # we initialize the retained variance 
    retained_variance = 0

    if _variance < 0:
        # Kaiser criterion
        idx = np.where(explained > 1 / kernel.shape[0])[0]
    elif 0 < _variance < 1:
        # Cumulative explained variance
        cumulative_variance = np.cumsum(explained)
        idx_var = np.where(cumulative_variance >= _variance)[0]
        idx_sv = np.where(singular_values > _min_singular_value)[0]
        # only keep components that are above the singular value threshold
        idx_both = np.intersect1d(np.arange(idx_var[0]+1), idx_sv)
        if idx_both.size < 2:
            tqdm.write("kernel_PCA: Not enough components after intersection, returning zeros.")
            return np.zeros(kernel.shape), retained_variance
        idx = idx_both

    if idx.size >= 2:
        # we compute the retained variance
        retained_variance = np.sum(explained[idx])
        components = pca.components_[idx]
        singular_values = pca.singular_values_[idx]
        singular_values[singular_values < _min_singular_value] = _min_singular_value
        L_m12 = np.diag(np.power(np.sqrt(singular_values), -1))
        U = np.dot(components.T, L_m12)
        W = np.dot(kernel, U).T
        Y = W.T 
    else:
        # we return a matrix of zeros with the shape of phi_map
        Y = np.zeros(kernel.shape)
    return Y, retained_variance

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

def italy_kernel_map(sq_wasserstein, gamma_wasserstein, variance=-1, _kernel_shift=1e-3):
    """
    This function computes the kernel map for the Italy dataset using the specified variance.
    :param sq_wasserstein: the squared Wasserstein distance matrix
    :param variance: the variance to use for the kernel PCA (default: -1, means the Kaiser rule is used)
    :param gamma_wasserstein: the gamma parameter for the Wasserstein kernel
    :param kernel_shift: the kernel shift parameter
    :return: the kernel map
    """
    ker_wass = np.exp(- math.pow(10, gamma_wasserstein) * sq_wasserstein, dtype=np.float32)

    # The diagonal is filled according to the operations performed with the kernels
    np.fill_diagonal(ker_wass,  (1 + _kernel_shift))

    # we center the composed kernel
    center_kernel_wass = center_kernel(ker_wass)

    # we obtain the kernel PCA map
    phi_wass, retained_variance = kernel_PCA(center_kernel_wass, _variance=variance)

    return phi_wass, retained_variance

def melbourne_kernel_map(Dsq, gamma_wasserstein, gamma_mass, gamma_euclid, variance=-1, _kernel_shift=1e-3):
    """
    This function computes the kernel map for the Melbourne dataset using the specified variance.
    :param Dsq: the dictionary with the squared distance matrices
    :param gamma_wasserstein: the gamma parameter for the Wasserstein kernel
    :param gamma_mass: the gamma parameter for the mass kernel
    :param gamma_euclid: the gamma parameter for the Euclidean kernel
    :param variance: the variance to use for the kernel PCA (default: -1, means the Kaiser rule is used)
    :param kernel_shift: the kernel shift parameter
    :return: the kernel map
    """
    ker_wass = np.exp(- math.pow(10, gamma_wasserstein) * Dsq['wasserstein'], dtype=np.float32)
    ker_mass = np.exp(- math.pow(10, gamma_mass) * Dsq['mass'], dtype=np.float32)
    ker_euclid = np.exp(- math.pow(10, gamma_euclid) * Dsq['euclid'], dtype=np.float32)

    k_composed = np.multiply(ker_mass.copy() + ker_euclid.copy(), ker_wass.copy())

    # The diagonal is filled according to the operations performed with the kernels
    np.fill_diagonal(k_composed,  2 * (1 + _kernel_shift) ** 2)

    # we center the composed kernel
    center_kernel_composed = center_kernel(k_composed)

    # we obtain the kernel PCA map
    phi_wass, retained_variance = kernel_PCA(center_kernel_composed, _variance=variance)

    return phi_wass, retained_variance

def objective_function_italy(cluster_number, sq_wasserstein, cluster_method, variance, gamma_wasserstein, kernel_shift=1e-3):
    """
    This function computes the composed kernel and performs KMedoids clustering to get the consensus index, 
        the effective number of components, and the average of the fast goodman-kruskal indices of a clustering solution
    :param cluster_number: the number of clusters
    :param sq_wasserstein: the squared wasserstein distance matrix
    :param cluster_method: the clustering method to use, either 'pam' or 'alternate' for KMedoids
    :param variance: the variance explained by the PCA components
    :param gamma_wasserstein: the gamma parameter of the RBF kernel for the wasserstein distance
    :return: the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a clustering solution
    """

    # we initialize the key performance indicators
    CI, min_ec, FGK_index = 0, 0, -1
    # we obtain the kernel PCA map
    phi_wass, retained_variance = italy_kernel_map(sq_wasserstein, gamma_wasserstein, variance, _kernel_shift=kernel_shift)

    if not(np.isclose(np.sum(np.abs(phi_wass)), 0)):
        CI, min_ec, FGK_index = N_KMedoids(phi_wass, cluster_number, cluster_method)
    # we return the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices.
    return CI, min_ec, FGK_index, retained_variance

def bayes_optimize_italy(gamma_wasserstein_max, sq_wasserstein, eps, bayes_init, bayes_iter, k_clusters, variance=-1):
    """
    This function performs Bayesian optimization for the kernel parameters.
    :param gamma_wasserstein_max: The gamma parameter that maximizes the off-diagonal kernel variance for the Wasserstein kernel.
    :param sq_wasserstein: The squared Wasserstein distance matrix.
    :param eps: The epsilon value for the optimization.
    :param bayes_init: The number of initial points for the Bayesian optimization.
    :param bayes_iter: The number of iterations for the Bayesian optimization.
    :param k_clusters: The number of clusters for the KMedoids algorithm.
    :param variance: The variance explained by the PCA components.
    """
    print("Bayesian optimization for kernel parameters")
    hyp_ranges = {'gamma_wasserstein': (gamma_wasserstein_max - eps, gamma_wasserstein_max + eps)} # gamma is in scientific notation

    optimizer = BayesianOptimization(f=None, pbounds=hyp_ranges, verbose=2, random_state=0, allow_duplicate_points=True)
    # we use the expected improvement acquisition function
    utility_function = UtilityFunction(kind="ei", xi=0.0)

    black_box = partial(objective_function_italy, k_clusters, sq_wasserstein, 'alternate', variance)

    start_time = time.time()
    ############
    _, iter_type, CI_index, EC, FGK_index, variance, dict_execution_time = optimize_kernels(black_box, optimizer, utility_function, bayes_init, bayes_iter, k_clusters, size_objective=True)

    end_time = time.time()
    print("Total execution time: ", end_time - start_time)

    # we retrieve the results
    return retrieve_results(optimizer, iter_type, CI_index, EC, FGK_index, variance), dict_execution_time

def objective_function_melbourne(cluster_number, Dsq, cluster_method, variance, gamma_wasserstein, gamma_mass, gamma_euclid, kernel_shift=1e-3):
    """
    This function computes the composed kernel and performs KMedoids clustering to get the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a clustering solution
    :param cluster_number: the number of clusters
    :param Dsq: the distance matrix
    :param cluster_method: the clustering method to use, either 'pam' or 'alternate' for KMedoids
    :param variance: the variance explained by the PCA components
    :param gamma_wasserstein: the gamma parameter of the RBF kernel for the wasserstein distance
    :param gamma_mass: the gamma parameter of the RBF kernel for the mass distance
    :param gamma_euclid: the gamma parameter of the RBF kernel for the euclidean distance
    :return: the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices of a clustering solution
    """

    # we initialize the key performance indicators
    CI, min_ec, FGK_index = 0, 0, -1

    # we obtain the kernel PCA map
    phi_wass, retained_variance = melbourne_kernel_map(Dsq, gamma_wasserstein, gamma_mass, gamma_euclid, variance, _kernel_shift=kernel_shift)

    if not(np.isclose(np.sum(np.abs(phi_wass)), 0)):
        CI, min_ec, FGK_index = N_KMedoids(phi_wass, cluster_number, cluster_method)
    # we return the consensus index, the effective number of components, and the average of the fast goodman-kruskal indices.
    return CI, min_ec, FGK_index, retained_variance

def bayes_optimize_melbourne(gammas, Dsq, eps, bayes_init, bayes_iter, k_clusters, variance=-1):
    """
    This function performs Bayesian optimization for the kernel parameters.
    :param gammas: The gammas dictionary containing the kernel parameters.
    :param Dsq: The dictionary with the squared distance matrices.
    :param eps: The epsilon value for the optimization.
    :param bayes_init: The number of initial points for the Bayesian optimization.
    :param bayes_iter: The number of iterations for the Bayesian optimization.
    :param k_clusters: The number of clusters for the KMedoids algorithm.
    :param variance: The variance explained by the PCA components.
    """
    print("Bayesian optimization for kernel parameters")
    hyp_ranges = {'gamma_wasserstein': (gammas['gamma_wasserstein'] - eps, gammas['gamma_wasserstein'] + eps), 
                'gamma_mass': (gammas['gamma_mass'] - eps, gammas['gamma_mass'] + eps), 
                'gamma_euclid': (gammas['gamma_euclid'] - eps, gammas['gamma_euclid'] + eps)} # gamma is in scientific notation

    optimizer = BayesianOptimization(f=None, pbounds=hyp_ranges, verbose=2, random_state=0, allow_duplicate_points=True)
    # we use the expected improvement acquisition function
    utility_function = UtilityFunction(kind="ei", xi=0.0)

    black_box = partial(objective_function_melbourne, k_clusters, Dsq, 'alternate', variance)

    start_time = time.time()
    ############
    _, iter_type, CI_index, EC, FGK_index, variance, dict_execution_time = optimize_kernels(black_box, optimizer, utility_function, bayes_init, bayes_iter, k_clusters)

    end_time = time.time()
    print("Total execution time: ", end_time - start_time)

    # we retrieve the results
    return retrieve_results(optimizer, iter_type, CI_index, EC, FGK_index, variance), dict_execution_time

def purity_clustering(_y_true, _k_clusters, _data, _n_init=5, _method='kmedoids'):
    purity_scores = np.zeros(_n_init)
    if not(_method in ['kmedoids', 'gmm', 'hierarchical']):
        raise ValueError("Method not recognized. Use 'kmedoids', 'gmm' or 'hierarchical'.")
    if _method == 'kmedoids':
        for i in range(_n_init):
            y_pred = KMedoids(n_clusters=_k_clusters, random_state=i, init='k-medoids++', method='alternate').fit_predict(_data)
            purity_scores[i] = purity_score(_y_true, y_pred)
    elif _method == 'gmm':
        for i in range(_n_init):
            y_pred = GaussianMixture(n_components=_k_clusters, random_state=i).fit_predict(_data)
            purity_scores[i] = purity_score(_y_true, y_pred)
    elif _method == 'hierarchical':
        y_pred = AgglomerativeClustering(n_clusters=_k_clusters).fit_predict(_data)
        # as the hierarchical clustering is deterministic, we can use the same score for all initializations
        purity_scores = np.ones(_n_init) * purity_score(_y_true, y_pred)
    purity_scores = np.array(purity_scores)
    return purity_scores.mean(), purity_scores.std()