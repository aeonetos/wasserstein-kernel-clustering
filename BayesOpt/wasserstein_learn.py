import numpy as np
import ot
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.cluster import KMeans
import pickle
from joblib import Parallel, delayed
from multiprocessing import cpu_count, Pool
from functools import partial
import time
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import KernelCenterer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids
import pandas as pd
import os
import sys

def unwrap_wasserstein_self(arg, **kwarg):
    return Wasserstein.map_distance(*arg, **kwarg)

# Class for approximated Wasserstein distance analysis
class Wasserstein:
    """
    A class that implements the approximated Wasserstein distance analysis through multiple distribution templates with additional functionalities.

    The Wasserstein class provides methods to compute Wasserstein distances, perform bootstrapping for confidence intervals,
    calculate CVaR, compute relative errors, and train templates for linear optimal transport mappings (LotMap).

    Attributes:
        - distances_: A list of distance matrices obtained from pairwise comparisons using different templates.
        - min_distances_: The minimum distance matrix computed from all pairwise comparisons.
        - templates_: A list of templates obtained from training on LotMap analysis.
        - phi_maps_: A list of feature maps obtained from LotMap analysis.
        - W_exact_: The exact Wasserstein distances computed from sampled pairs.
        - pairs_exact_: The pairs of samples used to compute exact Wasserstein distances.
        - rel_error_: The relative errors between exact and approximated Wasserstein distances.
        - mean_ci_: The confidence interval for the mean of the relative errors.
        - cvar_ci_: The confidence interval for the Conditional Value-at-Risk (CVaR) of the relative errors.
        - self.penalty_matrix_: The penalty matrix used to penalize the Wasserstein distances.
        - self.penalized_wasserstein_: The penalized Wasserstein distances.
    """
    def __init__(self):
        self.distances_ = None
        self.min_distances_ = None
        self.templates_ = None
        self.phi_maps_ = None
        self.W_exact_ = None
        self.pairs_exact_ = None
        self.rel_error_ = None
        self.mean_ci_ = None
        self.cvar_ci_ = None
        self.penalty_matrix_ = None 
        self.penalized_wasserstein_ = None 

    # Function to calculate CVaR
    def compute_cvar(self, data, alpha):
        sorted_data = np.sort(data)
        cutoff_index = int(np.ceil(alpha * len(sorted_data)))
        cvar = np.mean(sorted_data[cutoff_index:])
        return cvar
    
    def cvar_confidence_interval(self, alphas, confidence_level=0.95, n_bootstrap_samples=1000):
        intervals_cvar = {}
        # Define number of bootstrap samples
        for alpha in alphas:
            # Perform bootstrapping
            cvar_values = []
            for _ in range(n_bootstrap_samples):
                bootstrap_sample = np.random.choice(self.rel_error_, size=len(self.rel_error_), replace=True)
                # We use alpha < 0 to denote -CVaR_alpha(-X)
                if alpha <0:
                    cvar = -self.compute_cvar(-bootstrap_sample, -alpha)  # alpha = 1- confidence_level for CVaR
                # This is CVaR_alpha(X)
                else:
                    cvar = self.compute_cvar(bootstrap_sample, alpha)
                cvar_values.append(cvar)

            # Sort CVaR values
            sorted_cvar_values = np.sort(cvar_values)

            # Calculate lower and upper percentiles for confidence interval
            lower_percentile = (1 - confidence_level) / 2
            upper_percentile = 1 - lower_percentile
            lower_ci = sorted_cvar_values[int(lower_percentile * n_bootstrap_samples)]
            upper_ci = sorted_cvar_values[int(upper_percentile * n_bootstrap_samples)]
            intervals_cvar[alpha] = (lower_ci, upper_ci)
        self.cvar_ci_ = intervals_cvar

    def mean_confidence_interval(self, confidence_level=0.95):
        # Calculate the sample mean and standard deviation
        sample_mean, sample_std = np.mean(self.rel_error_), np.std(self.rel_error_)
        # Calculate the critical value
        critical_value = stats.t.ppf((1 + confidence_level) / 2, df=len(self.rel_error_) - 1)
        # Calculate the margin of error
        margin_of_error = critical_value * (sample_std / np.sqrt(len(self.rel_error_)))
        # Calculate the confidence interval
        confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
        # Output the results
        self.mean_ci_ = confidence_interval

    def relative_error(self, distance_matrix, abs_bound=True, q_bound=0.05):
        Warr = np.array([self.W_exact_[pair] for pair in self.pairs_exact_])
        mdarr = np.array([distance_matrix[pair] for pair in self.pairs_exact_])

        Warr_sort = np.sort(Warr)
        abs_error = np.abs(Warr-mdarr)
        rel_error = (mdarr - Warr)/(np.abs(Warr)+1e-15)

        if abs_bound:
            diff_w = np.diff(Warr_sort)
            wb = np.quantile(diff_w, q_bound) # absolute approximation bound error
            self.rel_error_ = rel_error[abs_error > wb].copy()
        else:
            self.rel_error_ = rel_error

    def compute_W2(self, xs, xt):
        M = ot.dist(xs, xt)
        return ot.emd2([], [], M)
    
    def sample_exact_distances(self, X, num_samples=1000):
        # Define the range list
        range_list = range(0, len(X))
        # Generate unique pair combinations
        pairs = list(combinations(range_list, 2))

        # Sample numbers within the range without replacement
        sample_ids = np.random.choice(range(len(pairs)), size=num_samples, replace=False)
        sample_pairs = [pairs[p] for p in sample_ids]

        Wdist = {} # compute exact Wasserstein distances between distributions in sample_pairs
        for pair in sample_pairs:
            Wdist[pair] = np.sqrt(self.compute_W2(X[pair[0]], X[pair[1]]))
        self.W_exact_ = Wdist
        self.pairs_exact_ = sample_pairs

    def lot_map(self, X, p):
        V=list()
        M = len(X)
        X0 = self.templates_[p]
        N = X0.shape[0]
        for ind in range(M):
            Ni=X[ind].shape[0]
            C=ot.dist(X[ind],X0)
            b=np.ones((N,))/float(N)
            a=np.ones((Ni,))/float(Ni)
            p=ot.emd(a,b,C) # exact linear program
            V.append(np.matmul((N*p).T,X[ind])-X0)
        V=np.asarray(V)
        return np.asarray([(1/np.sqrt(N)) * (V[i].flatten()) for i in range(V.shape[0])])

    def normalize_samples(self, X, normalization='min-max'):
        # We normalize the samples with min-max normalization
        # We do that with a pandas dataframe with the concatenated sample distributions
        # We then normalize the dataframe and convert it back to a numpy array
        distributions = pd.DataFrame()
        for i in range(len(X)):
            df_x = pd.DataFrame(X[i])
            df_x['label'] = i
            distributions = pd.concat([distributions, df_x], ignore_index=True, axis=0)
        # Then, we normalize all the columns except the 'label' column
        dist_cols = distributions.columns
        # we remove the 'label' column from the list of columns to be normalized
        dist_cols = dist_cols[dist_cols!='label']
        if normalization == 'min-max':
            distributions[dist_cols] = (distributions[dist_cols] - distributions[dist_cols].min()) / (distributions[dist_cols].max() - distributions[dist_cols].min())
        elif normalization == 'z-score':
            distributions[dist_cols] = (distributions[dist_cols] - distributions[dist_cols].mean()) / distributions[dist_cols].std()
        # Finally, we convert the dataframe back to a list of numpy arrays
        else:
            print("Normalization method not recognized.")
        X_norm = []
        for i in range(len(X)):
            X_norm.append(distributions[distributions['label']==i][dist_cols].values)
        return X_norm
    
    def train_template(self, X, n_threshold=3000):
        N=int(np.asarray([x.shape[0] for x in X]).mean())
        if len(X) <= n_threshold:
            Xc = np.concatenate(X)
        else:
            ch = np.random.choice(len(X), n_threshold,  replace=False)
            Xc = np.concatenate([X[i] for i in ch])
        # we run K-means on the concatenated data
        # with N centroids we get a template
        template_points = KMeans(n_clusters=N, n_init='auto').fit(Xc)
        # we use the tepmlate points as the first template
        self.templates_ = [template_points.cluster_centers_]
        self.phi_map_ = [self.lot_map(X, 0)]

    def add_templates(self,X,K):
        if K>0:
            # As self.phi_map_[0] can be a massive array, it might be necessary to reduce the dimensionality of the data
            # For this purpose, we can use PCA to reduce the dimensionality of the data
            # We use a threshold on the size of self.phi_map_[0] to decide whether to use PCA or not
            if self.phi_map_[0].size > 1e6:
                # We reduce the dimensionality of the data to 99.9% of the variance
                pca_phi = PCA(n_components=0.999)
                pca_phi.fit(self.phi_map_[0])
                pca_phi_map = pca_phi.transform(self.phi_map_[0])
                kmeans = KMeans(n_clusters=K, n_init='auto').fit(pca_phi_map)
                # we have to reshape the templates to the original shape of the data after PCA transformation
                self.templates_ += [pca_phi.inverse_transform(kmeans.cluster_centers_[i]).reshape(self.templates_[0].shape) for i in range(K)]
            else:
                pca_phi_map = self.phi_map_[0]
                # we try KMedoids instead of KMeans
                # importing KMedoids from sklearn_extra.cluster
                kmedoids = KMedoids(n_clusters=K).fit(pca_phi_map)
                self.templates_ += [X[i] for i in kmedoids.medoid_indices_]
                # kmeans = KMeans(n_clusters=K, n_init='auto').fit(pca_phi_map)
                # self.templates_ += [kmeans.cluster_centers_[i].reshape(self.templates_[0].shape) for i in range(K)]

    def map_distance(self,X,k):
        p_map = self.lot_map(X, k)
        distances_k = pdist(p_map, metric='euclidean')
        return squareform(distances_k), p_map
    
    def pairwise_distances(self,X,zscore=0):
        # The first is precomputed, so we can directly obtain the distance matrix associated with it
        distances = pdist(self.phi_map_[0], metric='euclidean')
        sq_distances = squareform(distances)
        # Iteratively computes other distance matrices approximations with their feature maps and store them in the object attributes
        distance_matrices = [sq_distances]
        phi_maps = [self.phi_map_[0]]
        if len(self.templates_)>1:
            for k in range(1, len(self.templates_)):
                distance_matrix, p_map = self.map_distance(X,k)
                distance_matrices.append(distance_matrix)
                phi_maps.append(p_map)
        self.distances_ = distance_matrices
        self.phi_map_ = phi_maps
        # Compute the minimum distance matrix
        self.compute_minimum_distances(zscore)

    def parallel_pairwise_distances(self,X,njobs=-1,zscore=0):
        # The first is precomputed, so we can directly obtain the distance matrix associated with it
        distances = pdist(self.phi_map_[0], metric='euclidean')
        sq_distances = squareform(distances)
        # Iteratively computes other distance matrices approximations and store them in the object attributes
        distance_matrices = [sq_distances]
        phi_maps = [self.phi_map_[0]]
        results = []
        num = range(1, len(self.templates_))
        results = Parallel(n_jobs= njobs, backend="threading")\
            (delayed(unwrap_wasserstein_self)(i) for i in zip([self]*len(num), [X]*len(num), num))
        
        distance_result, phi_maps_result = zip(*results)
        self.distances_ = distance_matrices + list(distance_result)
        self.phi_map_ = phi_maps + list(phi_maps_result)

        # Compute the minimum distance matrix
        self.compute_minimum_distances(zscore)

    def compute_minimum_distances(self, zcore=0):
        # Compute the minimum distance matrix
        min_dist = self.distances_[0]
        if len(self.distances_) > 1:
            distances = np.array(self.distances_)
            avg = np.average(distances, axis=0)
            std = np.std(distances, axis=0)
            min_dist = zcore * std + avg
        # Assign the minimum distance matrix to the object attributes
        self.min_distances_ = min_dist

    def penalty_matrix(self, X, penalty_type='entry', penalize_entry=None):
        if penalty_type=='entry':
            weight_penalty = [np.sum(X[i][:,penalize_entry]) for i in range(len(X))]
            weight_matrix = np.array([weight_penalty for i in range(len(weight_penalty))])
            # we then compute the pairwise weight distance
            self.penalty_matrix_ = np.abs(weight_matrix - weight_matrix.T)
        elif penalty_type=='volume':
            # we get the volume of the node embeddings, for which the volume is given by the .shape[0] attribute
            volumes = [X[i].shape[0] for i in range(len(X))]
            # we then compute the pairwise volume distance, which is given by the absolute difference of the volumes, i.e., |v_i - v_j| for i,j=1,...,n
            # we first create a matrix of size n x n with the volumes
            volume_matrix = np.array([volumes for i in range(len(volumes))])
            # we then compute the pairwise volume distance
            self.penalty_matrix_ = np.abs(volume_matrix - volume_matrix.T)            

    def penalize_wasserstein(self, lbda, composition_type='multiplicative'):
        if composition_type=='additive':
            self.penalized_wasserstein_ = lbda * self.penalty_matrix_ + (1 - lbda) * self.min_distances_
        elif composition_type=='multiplicative':
            self.penalized_wasserstein_ = lbda * np.multiply(self.penalty_matrix_,self.min_distances_) + (1 - lbda) * self.min_distances_
    
# Class for Nystrom approximation 
class Nystrom:
    """
    The Nystrom class implements the Nystrom approximation algorithm for solving large eigendecomposition problems with kernel matrices.
    It provides the following attributes and methods:
    
    Attributes
    - n_samples_ : Number of sample columns to be selected.
    - n_components_ : Number of components retained from PCA.
    - kernel_matrix_ : Input kernel matrix.
    - random_sampling_ : Indices of the randomly selected columns.
    - K_nm_ : Sub-matrix with the selected colums.
    - mn_1_ : Matrix used for additive normalization.
    - K_mm_inv_ : Inverse of the selected kernel sub-matrix.
    - alpha_ : Constant terms for kernel normalization.
    - U_ : Projection matrix.
    - W_ : Projected data.

    Methods
    - transform(self, ker_col)
        This method takes a column vector (or vectors) `ker_col` as input and projects the PCA transformation to it.
        It returns the projected data.
    - fit_transform(self)
        This method performs the Nystrom approximation and computes the projection matrix.
        It randomly samples `n_samples_` indices from the input kernel matrix, computes various intermediate matrices,
        performs PCA, and computes the projection matrix and the projected data.

    """
    def __init__(self, kernel_matrix, n_samples, n_components=1e20):
        self.n_samples_ = n_samples
        self.n_components_ = n_components # the default value is set to a very large number, so the PCA retains all the components.
        self.kernel_matrix_ = kernel_matrix 
        self.random_sampling_ = None 
        self.K_nm_ = None 
        self.mn_1_ = None 
        self.K_mm_inv_ = None 
        self.alpha_ = None
        self.s_K_p_t_ = None
        self.U_ = None 
        self.W_ = None 
        self.explained_variance_ = 1.0 

    def transform(self, ker_col):
        # additive normalization of the kernel column based on the Nystrom approximation.
        nys_kernel = ker_col - np.dot(np.dot(np.dot(self.mn_1_, self.K_nm_), self.K_mm_inv_), ker_col) + self.alpha_
        PCA_projection = np.dot(self.U_.T, nys_kernel)
        return PCA_projection
    
    def fit_transform(self):
        self.random_sampling_ = np.random.choice(np.arange(0, self.kernel_matrix_.shape[0]), size=self.n_samples_, replace=False) ## Randomly sample indices from the input kernel matrix.
        K_mm = self.kernel_matrix_[self.random_sampling_][:,self.random_sampling_].copy() ## Select the corresponding sub-matrix from the input kernel matrix.
        K_mn = self.kernel_matrix_[self.random_sampling_].copy() ## Select the corresponding columns from the input kernel matrix.
        self.K_nm_ = K_mn.T.copy() ## Compute the transpose of the selected columns from the kernel matrix.
        n_data = self.kernel_matrix_.shape[0] ## Get the number of data points for which their pairwise kernel value is defined in the kernel matrix. 
        n_1 = np.ones((n_data,n_data)) / n_data ## Define a matrix of size n_data x n_data with entries 1/n_data.
        nm_1 = np.ones((n_data,self.n_samples_)) / n_data ## Define a matrix of size n_data x n_samples with entries 1/n_data.
        self.mn_1_ = np.ones((self.n_samples_, n_data)) / n_data ## Define a matrix of size n_samples x n_data with entries 1/n_data.
        self.K_mm_inv_ = np.linalg.inv(K_mm).copy() ## Inverse of the selected kernel sub-matrix.
        K_p = np.dot(np.dot(self.K_nm_, self.K_mm_inv_), K_mn) ## Approximated kernel matrix obtained from the Nystrom method.
        K_mn_t = K_mn - np.dot(K_mn, n_1) - np.dot(self.mn_1_, K_p) + np.dot(np.dot(self.mn_1_, K_p), n_1) ## Intermediate matrix computation.
        K_mm_t = K_mm - np.dot(self.mn_1_, self.K_nm_) - np.dot(K_mn, nm_1) + np.dot(np.dot(self.mn_1_, K_p), nm_1) ## Intermediate matrix computation.
        K_nm_t = K_mn_t.T.copy() ## Transpose of the intermediate matrix.
        v_n_1 = (np.ones(n_data)/n_data).reshape(-1,1) ## vector of size n_data x 1 with entries 1/n_data.
        self.alpha_ = - np.dot(K_mn, v_n_1) + np.dot(np.dot(self.mn_1_, K_p),v_n_1) ## Constant terms for kernel normalization

        pca1 = PCA()
        pca1.fit(K_mm_t)
        c_K_mm_t = pca1.components_.T
        s_K_mm_t = pca1.singular_values_ 

        # pseudo-inverse
        bound1 = np.min(np.where(np.isclose(s_K_mm_t,0))[0])
        s_K_mm_t_reduced = s_K_mm_t[:bound1]
        c_K_mm_t_reduced = c_K_mm_t[:,:bound1]
        
        K_mm_t_12 = np.dot(np.dot(c_K_mm_t_reduced, np.diag(1/np.sqrt(s_K_mm_t_reduced))),c_K_mm_t_reduced.T) ## Intermediate matrix computation
        K_p_t = (1/n_data) * np.dot(np.dot(np.dot(K_mm_t_12,K_mn_t), K_nm_t), K_mm_t_12) ## Intermediate matrix computation

        pca2 = PCA()
        pca2.fit(K_p_t)

        c_K_p_t = pca2.components_.T
        self.s_K_p_t_ = pca2.singular_values_ 
        
        # pseudo-inverse
        where_zero = np.where(np.isclose(self.s_K_p_t_,0))[0]
        if where_zero.size > 0:
            bound2 = np.min(where_zero)
        else:
            bound2 = c_K_p_t.shape[1]
        if self.n_components_ < bound2:
            V = c_K_p_t[:,:self.n_components_]
            self.explained_variance_ = np.sum(self.s_K_p_t_[:self.n_components_])/np.sum(self.s_K_p_t_)
        else:
            V = c_K_p_t[:,:bound2]

        self.U_ = np.dot(K_mm_t_12, V) ## Projection matrix
        self.W_ = np.dot(K_nm_t,self.U_).T # projection to the PCs

# Class for Wasserstein Kernel PCA 
class WK_PCA:
    """
    A class that implements a customized version of Kernel PCA (KPCA) with additional functionalities, suitable for Wasserstein distances.
    
    Attributes:
        - kernel_matrix_: The kernel matrix computed from input data or provided by the user.
    """
    def __init__(self):
        self.gamma_ = None
        self.kernel_matrix_ = None
        self.perturbation_ = None
        self.normalized_kernel_ = None
        self.random_sampling_ = None
        self.dispersion_ = None
        self.robust_dispersion_ = None # robust dispersion is the dispersion computed with robust estimatiors of location and scale
        self.dimensions_ = None
        self.alpha_ = None
        self.explained_variance_ = 1
        self.U_ = None
        self.W_ = None

    def gaussian_kernel(self, distances, gamma, perturbation_method='minimum-shift'):
        if perturbation_method == 'minimum-shift':
            distances2 = np.square(distances)
            Dc = distances2 - distances2.mean(axis=0)
            Dc = Dc - distances2.mean(axis=1).reshape(-1,1)
            Sc = -0.5 * Dc
            eigenvalues = np.linalg.eigvalsh(Sc)
            perturbation = np.full(Sc.shape, - 2 * eigenvalues[0])
            np.fill_diagonal(perturbation, 0)
            D = Dc + perturbation
        else:
            D = np.square(distances)
        
        self.gamma_ = gamma
        self.kernel_matrix_ = np.exp(- self.gamma_ * D)

    def kernel_quality_features(self, lbdas):
        v = self.kernel_matrix_[np.triu_indices(self.kernel_matrix_.shape[0], k=1)]
        mean = np.mean(v)
        var = np.var(v)
        dispersion = var/(mean + 1e-20)
        sort_eigenvalues = (-np.sort(-np.real(lbdas)))
        explained_var = sort_eigenvalues/ (np.sum(sort_eigenvalues) + 1e-20) # avoids division by zero
        dimensions = explained_var[explained_var >= 1/explained_var.size].size
        median = np.median(v)
        mad = np.median(np.abs(v - median)) * 1.4826
        robust_dispersion = (mad ** 2)/(median + 1e-20) # the median absolute deviation (MAD) should be squared to be comparable to the variance
        return dispersion, robust_dispersion, dimensions
    
    def normalized_kernel_from_distance(self, distances): # distances is a matrix of column vectors
        if distances.ndim == 1:
            distances  = distances.reshape(-1, 1)
        
        # we perturbe the distance matrix for consistency.
        # to do so, we add a small constant to the minimum value of each column of the 'distances' matrix.
        # we first identify the argmin values of each column
        argmin = np.argmin(distances, axis=0)
        # we then add a small constant to all the values of the columns except the argmin values
        # for that, we create a matrix with the same shape as 'distances' and fill it with the small constant
        perturbation = np.full(distances.shape, self.perturbation_)
        # we then set the values of the argmin indices to 0
        perturbation[argmin, np.arange(perturbation.shape[1])] = 0
        # we then add the perturbation matrix to the distance matrix
        D = distances + perturbation
        # we then compute the kernel matrix
        k_from_d = np.exp(- self.gamma_ * np.square(D))

        k_from_d_mean = np.mean(k_from_d, axis=0, keepdims=True)
        normalized_k_from_d = k_from_d - k_from_d_mean + self.aplha_
        return normalized_k_from_d
    
    def distances_to_feature(self, distances):
        kernel_vec = self.normalized_kernel_from_distance(distances)
        return np.dot(self.U_.T, kernel_vec)
    
    def KPCA(self, method='exact', n_samples = 10000, n_components=1e20):
        self.n_components_ = n_components
        if method == 'exact':
            # Create an instance of PCA with the desired number of components
            pca = PCA()
            # Create an instance of KernelCenterer
            centerer = KernelCenterer()
            # constant part of the kernel normalization
            self.aplha_ = np.mean(self.kernel_matrix_) - np.mean(self.kernel_matrix_, axis=1, keepdims=True)
            # Fit and transform the kernel matrix
            self.normalized_kernel_ = centerer.fit_transform(self.kernel_matrix_)
            # Fit the PCA model to your data
            pca.fit(self.normalized_kernel_)
            # Access the principal components (eigenvectors)
            components = pca.components_
            # Access the singular values (eigenvalues)
            singular_values = pca.singular_values_
            # The kernel dispersion is measured with the uncentered kernel matrix, while the dimensions are measured with the centered kernel matrix.
            self.dispersion_, self.robust_dispersion_, self.dimensions_ = self.kernel_quality_features(singular_values)

            if n_components < singular_values.size:
                m_components = components[:n_components].copy()
                m_values = singular_values[:n_components].copy()
            else:
                m_components = components.copy()
                m_values = singular_values.copy()

            self.explained_variance_ = np.sum(m_values)/np.sum(singular_values)
            L_m12 = np.diag(np.power(np.sqrt(m_values), -1))
            self.U_ = np.dot(m_components.T, L_m12)
            self.W_ = np.dot(self.normalized_kernel_, self.U_).T
        
        # TODO: The Nystrom method is giving unstable results. Need to fix it.

        elif method == 'nystrom':
            self.n_samples_ = n_samples
            self.random_sampling_ = np.random.choice(np.arange(0, self.kernel_matrix_.shape[0]), size=self.n_samples_, replace=False) ## Randomly sample indices from the input kernel matrix.
            K_mm = self.kernel_matrix_[self.random_sampling_][:,self.random_sampling_].copy() ## Select the corresponding sub-matrix from the input kernel matrix.
            K_mn = self.kernel_matrix_[self.random_sampling_].copy() ## Select the corresponding columns from the input kernel matrix.
            self.K_nm_ = K_mn.T.copy() ## Compute the transpose of the selected columns from the kernel matrix.
            n_data = self.kernel_matrix_.shape[0] ## Get the number of data points for which their pairwise kernel value is defined in the kernel matrix. 
            n_1 = np.ones((n_data,n_data)) / n_data ## Define a matrix of size n_data x n_data with entries 1/n_data.
            nm_1 = np.ones((n_data,self.n_samples_)) / n_data ## Define a matrix of size n_data x n_samples with entries 1/n_data.
            self.mn_1_ = np.ones((self.n_samples_, n_data)) / n_data ## Define a matrix of size n_samples x n_data with entries 1/n_data.
            self.K_mm_inv_ = np.linalg.inv(K_mm).copy() ## Inverse of the selected kernel sub-matrix.
            K_p = np.dot(np.dot(self.K_nm_, self.K_mm_inv_), K_mn) ## Approximated kernel matrix obtained from the Nystrom method.
            K_mn_t = K_mn - np.dot(K_mn, n_1) - np.dot(self.mn_1_, K_p) + np.dot(np.dot(self.mn_1_, K_p), n_1) ## Intermediate matrix computation.
            K_mm_t = K_mm - np.dot(self.mn_1_, self.K_nm_) - np.dot(K_mn, nm_1) + np.dot(np.dot(self.mn_1_, K_p), nm_1) ## Intermediate matrix computation.
            K_nm_t = K_mn_t.T.copy() ## Transpose of the intermediate matrix.
            v_n_1 = (np.ones(n_data)/n_data).reshape(-1,1) ## vector of size n_data x 1 with entries 1/n_data.
            self.alpha_ = - np.dot(K_mn, v_n_1) + np.dot(np.dot(self.mn_1_, K_p),v_n_1) ## Constant terms for kernel normalization

            pca1 = PCA()
            pca1.fit(K_mm_t)
            c_K_mm_t = pca1.components_.T
            s_K_mm_t = pca1.singular_values_ 

            # pseudo-inverse
            where_zero1 = np.where(np.isclose(s_K_mm_t,0))[0]
            if where_zero1.size > 0:
                bound1 = np.min(where_zero1)
            else:
                bound1 = c_K_mm_t.shape[1]
            s_K_mm_t_reduced = s_K_mm_t[:bound1]
            c_K_mm_t_reduced = c_K_mm_t[:,:bound1]
            
            K_mm_t_12 = np.dot(np.dot(c_K_mm_t_reduced, np.diag(1/np.sqrt(s_K_mm_t_reduced))),c_K_mm_t_reduced.T) ## Intermediate matrix computation
            K_p_t = (1/n_data) * np.dot(np.dot(np.dot(K_mm_t_12,K_mn_t), K_nm_t), K_mm_t_12) ## Intermediate matrix computation

            pca2 = PCA()
            pca2.fit(K_p_t)

            c_K_p_t = pca2.components_.T
            self.s_K_p_t_ = pca2.singular_values_ 
            
            # pseudo-inverse
            where_zero2 = np.where(np.isclose(self.s_K_p_t_,0))[0]
            if where_zero2.size > 0:
                bound2 = np.min(where_zero2)
            else:
                bound2 = c_K_p_t.shape[1]
            if self.n_components_ < bound2:
                V = c_K_p_t[:,:self.n_components_]
                self.explained_variance_ = np.sum(self.s_K_p_t_[:self.n_components_])/np.sum(self.s_K_p_t_)
            else:
                V = c_K_p_t[:,:bound2]

            self.dispersion_, self.robust_dispersion_, self.dimensions_ = self.kernel_quality_features(self.s_K_p_t_)
            self.U_ = np.dot(K_mm_t_12, V) ## Projection matrix
            self.W_ = np.dot(K_nm_t,self.U_).T # projection to the PCs


def get_embedding_arrays(embeddings, grid_id):
    # we create a list of arrays with the node embeddings of each grid
    return embeddings[embeddings['grid']==grid_id].iloc[:,:-1].values, grid_id

# we especify this code to be run only if the file is run as a script
if __name__ == '__main__':

    grid_type = 'LV'
    write_embeddings = False
    
    n_read = 48
    
    if write_embeddings:
        # we read the data from the nodes embeddings
        embeddings = pd.read_csv('data/node_embeddings_{}.csv'.format(grid_type), index_col=False)

        # embeddings is a dataframe with the node embeddings of different grids
        # the file has six columns, where the column 'grid' indicates the grid ID, and the other columns have the values of the node embeddings
        # for each grid, we would like to have a 2D numpy array with the node embeddings
        # we first get the unique grid IDs
        grid_ids = embeddings['grid'].unique()
        # we then create an array with the node embeddings of each grid

        # we count the time it takes to compute the embeddings
        start_time = time.time()
        partial_get_embeddings = partial(get_embedding_arrays, embeddings)
        
        if n_read > 1:
            print("Reading the embeddings with parallel processing")
            jobs = [i for i in grid_ids]
            with Parallel(n_jobs=n_read) as parallel:
                result_embeddings = parallel(delayed(partial_get_embeddings)(job) for job in jobs)

            # we unzip the result, to a list X with the embeddings and the grid_ids list. Each entry of results_embeddings is a tuple with the embeddings and the grid_id
            X, grid_ids = zip(*result_embeddings)
            
            print("Exporting the embeddings to a pickle file")

            dict_names = {i:grid_ids[i] for i in range(len(grid_ids))}
            with open("data/grids_names_{}.pickle".format(grid_type), "wb") as file:
                pickle.dump(dict_names, file)
            with open("data/array_embeddings_{}.pickle".format(grid_type), "wb") as file:
                pickle.dump(X, file)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time:", elapsed_time)
        else:
            X = [embeddings[embeddings['grid']==grid_id].iloc[:,:-1].values for grid_id in grid_ids]
            dict_names = {i:grid_ids[i] for i in range(len(grid_ids))}
            with open("data/grids_names_{}.pickle".format(grid_type), "wb") as file:
                pickle.dump(dict_names, file)
            with open("data/array_embeddings_{}.pickle".format(grid_type), "wb") as file:
                pickle.dump(X, file)
    else:
        # we read the embeddings
        with open('data/array_embeddings_{}.pickle'.format(grid_type), 'rb') as handle:
            X = pickle.load(handle)
        # we read the overall line length of the grids
        with open('data/grids_names_{}.pickle'.format(grid_type), 'rb') as handle:
            dict_names = pickle.load(handle)

    ########################
    # Fix the of the problem
    K = 39
    jobs = 24
    cvar_alphas = [-0.95, -0.9, 0.9, 0.95]
    zscore = 0.00

    # Compute Wasserstein distances through planes
    wass_model = Wasserstein()
    print("Normalize the data")
    X = wass_model.normalize_samples(X)
    print("Getting the reference template and the LOT map")
    wass_model.train_template(X)
    print("Computing pairwise distances for the reference template")
    wass_model.pairwise_distances(X, zscore)
    print("Adding", K, "templates")
    wass_model.add_templates(X, K)
    print("Computing pairwise distances for", K, "templates")
    wass_model.parallel_pairwise_distances(X, jobs, zscore)

    # Check the solution's quality
    wass_model.sample_exact_distances(X)
    min_distances = wass_model.min_distances_.copy()
    wass_model.relative_error(min_distances)
    wass_model.mean_confidence_interval()
    wass_model.cvar_confidence_interval(cvar_alphas)

    # Print the CI for the mean and for CVaRs under different cummulative probability
    print("Confidence interval for average error: ", wass_model.mean_ci_)
    print(wass_model.cvar_ci_)

    # Store the object to a file using pickle
    with open("wasserstein_grids.pickle", "wb") as file:
        pickle.dump(wass_model, file)