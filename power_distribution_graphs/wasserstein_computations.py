"""Approximate Wasserstein distance computations for graph embeddings."""

import pickle
from joblib import Parallel, delayed
from multiprocessing import cpu_count, Pool
from functools import partial
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import ot
from scipy.spatial.distance import pdist, squareform
from sklearn_extra.cluster import KMedoids
import gc
import psutil
import os


def unwrap_wasserstein_self(arg, **kwarg):
    """Helper to unpack arguments when using :func:`Parallel`."""
    return Wasserstein.map_parallel_distance(*arg, **kwarg)

# Class for approximated Wasserstein distance analysis
class Wasserstein:
    """
    A class that implements the approximated Wasserstein distance analysis through multiple distribution templates with additional functionalities.

    The Wasserstein class provides methods to compute Wasserstein distances.

    Attributes:
        - zscore_: The z-score used to compute the minimum distance matrix.
        - avg_distances_: The average distance estimation obtained from the multiple distribution templates.
        - std_distances_: The standard deviation of the distance estimation obtained from the multiple distribution templates.
        - min_distances_: The distance obtained from the multiple distribution templates with the z-score.
        - initial_distances_: The initial distance matrix obtained from the first template.
        - templates_: The templates obtained from the K-medoids clustering.
        - templates_indices_: The indices of the templates in the node embeddings array.
        - phi_maps_: A list of feature maps obtained from LotMap analysis.
        - W_exact_: The exact Wasserstein distances computed from sampled pairs.
        - pairs_exact_: The pairs of samples used to compute exact Wasserstein distances.
    """
    def __init__(self):
        self.zscore_ = 0
        self.avg_distances_ = None
        self.std_distances_ = None
        self.min_distances_ = None
        self.initial_distances_ = None
        self.templates_ = None
        self.templates_indices_ = None
        self.phi_maps_ = None
        self.W_exact_ = None
        self.pairs_exact_ = None

    def compute_W2(self, xs, xt):
        """Compute the squared 2-Wasserstein distance between two point clouds."""
        M = ot.dist(xs, xt)
        return ot.emd2([], [], M)
    
    def sample_exact_distances(self, X, num_samples=1000):
        """
        The function samples pairs of distributions to compute the exact Wasserstein distances between them.
        The sampling consists of the following steps:
        - We define the set to sample, such that the templates are excluded from the set.
        - We sample pairs without replacement from the set of distributions.
        - We check if all the pairs are unique. The pair (i,j) and (j,i) are considered the same pair.
        - While the number of unique pairs is less than the number of samples, we sample new pairs.
        - We compute the exact Wasserstein distances between distributions in the sample pairs.
        - We store the exact Wasserstein distances and the sample pairs in the object attributes.
        """
        # We sample pairs without replacement from the set of distributions
        # we leave out the teplates from the set of distributions to sample the pairs, as the error will be zero for the templates
        sampling_set = [i for i in range(len(X)) if i not in self.templates_indices_]
        sampled_pairs = np.array([np.random.choice(sampling_set, size=2, replace=False) for _ in range(num_samples)])
        # we check if all the pairs are unique. The pair (i,j) and (j,i) are considered the same pair
        sorted_pairs = np.sort(sampled_pairs, axis=1)
        unique_pairs = np.unique(sorted_pairs, axis=0)
        # while the number of unique pairs is less than the number of samples, we sample new pairs
        while unique_pairs.shape[0] < num_samples:
            new_pairs = np.array([np.random.choice(sampling_set, size=2, replace=False) for _ in range(num_samples - len(unique_pairs))])
            new_sorted_pairs = np.sort(new_pairs, axis=1)
            new_unique_pairs = np.unique(new_sorted_pairs, axis=0)
            unique_pairs = np.vstack((unique_pairs, new_unique_pairs))
        # we change unique_pairs to a list of tuples
        unique_pairs = [tuple(unique_pairs[j]) for j in range(unique_pairs.shape[0])]

        Wdist = {} # compute exact Wasserstein distances between distributions in sample_pairs
        for pair in unique_pairs:
            Wdist[pair] = np.sqrt(self.compute_W2(X[pair[0]], X[pair[1]]))
        self.W_exact_ = Wdist
        self.pairs_exact_ = unique_pairs

    def lot_map(self, X, k):
        V=list()
        M = len(X)
        X0 = self.templates_[k]
        N = X0.shape[0]
        for ind in range(M):
            Ni=X[ind].shape[0]
            C=ot.dist(X[ind],X0)
            b=np.ones((N,))/float(N)
            a=np.ones((Ni,))/float(Ni)
            p=ot.emd(a,b,C) # exact linear program
            V.append(np.matmul((N*p).T,X[ind])-X0)
        V=np.asarray(V, dtype=np.float32)
        return np.asarray([(1/np.sqrt(N)) * (V[i].flatten()) for i in range(V.shape[0])], dtype=np.float32)
    
    def train_template(self, X, n_threshold=50000):
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
            pca_phi_map = self.phi_map_[0]
            kmedoids = KMedoids(n_clusters=K).fit(pca_phi_map)
            self.templates_ += [X[i] for i in kmedoids.medoid_indices_]
            self.templates_indices_ = kmedoids.medoid_indices_

    def map_distance(self,X,k):
        p_map = self.lot_map(X, k)
        distances_k = np.float32(pdist(p_map, metric='euclidean'))
        return distances_k, p_map

    def map_parallel_distance(self,X,k,grid_type):
        # we evaluate the time it takes to compute the distance matrix
        start = time.time()
        p_map = self.lot_map(X, k)
        distances_k = np.float32(pdist(p_map, metric='euclidean'))
        end = time.time()
        print("Elapsed time to compute the distance matrix for template", k, ":", end - start)
        del p_map
        gc.collect()
        # we obtain the exact distances between the templates and the other distributions
        # using get_distance_row_or_column_in_vector to get the entries of the distance matrix that correspond to the template
        entries_exact = get_distance_row_or_column_in_vector(distances_k, self.templates_indices_[k-1])
        exact_distances = distances_k[entries_exact]
        # we export exact_distances to a numpy array to free memory
        np.save(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasserstein_models'),
                             'exact_distances_parallel_{0}/exact_distances_{1}.npy'.format(grid_type, k)), exact_distances)
        del exact_distances
        # we return the distances
        return distances_k
    
    def initial_pairwise_distances(self):
        self.initial_distances_ = np.float32(pdist(self.phi_map_[0], metric='euclidean'))
        self.phi_map_ = [self.phi_map_[0]]

    def parallel_pairwise_distances(self,X, grid_type, njobs=-1,zscore=0):
        """
        Compute pairwise distances between templates and data computing Wasserstein distances in parallel.
        This function calculates distances between each template and the data matrix X.
        It also computes average and standard deviation of these distances, also considering the initial distance matrix.
        The final minimum distance matrix is obtained by adding the average and the standard deviation, adjusted by zscore.

        Parameters:
        -----------
        X : list of numpy arrays with the node embeddings of the grids.

        grid_type : str the type of grid, either 'MV' or 'LV'

        njobs : int, default=-1
            Number of parallel jobs. -1 means using all available processors.
        
        zscore : float, default=0
            Z-score normalization factor for the distances. 
            It scales the standard deviation to adjust the distance matrix.

        Returns:
        --------
        None
            Updates object attributes with computed distances:
            - avg_distances_: average distance from each template to data points.
            - std_distances_: standard deviation of distances.
            - min_distances_: final minimum distance matrix after adjusting with zscore.
        """
        # The first is precomputed, so we can directly obtain the distance matrix associated with it
        # Iteratively computes other distance matrices approximations and store them in the object attributes
        self.zscore_ = zscore
        num = range(1, len(self.templates_))

        # we monitor the memory usage
        print("Memory usage before parallel loop:", psutil.virtual_memory().percent)

        start = time.time()

        distance_result = Parallel(n_jobs= njobs, backend="threading")\
            (delayed(unwrap_wasserstein_self)(i) for i in zip([self]*len(num), [X]*len(num), num, [grid_type]*len(num)))
        end = time.time()
        print("Elapsed time parallel loop:", end - start)        
        
        # we monitor the memory usage
        print("Memory usage after parallel loop:", psutil.virtual_memory().percent)


        # we store the distances in the object attributes
        distance_result.insert(0, self.initial_distances_)

        # Compute the minimum distance matrix, without storing the distances nor the feature maps in the object attributes
        distance_result = np.vstack(distance_result, dtype=np.float32)

        # we monitor the memory usage
        print("Memory usage before computing the avg and the std:", psutil.virtual_memory().percent)

        avg = np.mean(distance_result, axis=0, dtype=np.float32)
        self.avg_distances_ = np.array(avg, dtype=np.float32)
        # here we compute the std of the distances using the same array distance_result
        np.subtract(distance_result, avg, out=distance_result, dtype=np.float32)
        np.square(distance_result, out=distance_result, dtype=np.float32)
        # as the sum will reduce the dimensionality of the array, we need to define another array to store the sum
        std = np.sum(distance_result, axis=0, dtype=np.float32)
        # we delete distance_result to free memory
        del distance_result
        gc.collect()
        # we compute the std of the distances
        np.sqrt(std/len(self.templates_), out=std, dtype=np.float32)
        self.std_distances_ = np.array(std, dtype=np.float32)
        # we obtain the final minimum distance matrix by adding the average and the std
        np.add(avg, self.zscore_*std, out=avg, dtype=np.float32)
        # we delete the std array to free memory
        self.min_distances_ = avg
        del std, avg
        gc.collect()
        # we monitor the memory usage
        print("Memory usage after computing the avg and the std:", psutil.virtual_memory().percent)

        # we read the exact distances from the files
        exact_distances = [np.load(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasserstein_models'),
                                'exact_distances_parallel_{0}/exact_distances_{1}.npy'.format(grid_type,k))) for k in range(1, len(self.templates_))]

        # we replace the approximated distances with the exact distances when they are available
        # there is no need to access the square form of the distance matrix, as we have the indices of the templates and the exact distances
        # we call the function get_distance_row_or_column_in_vector to get the entries of the distance matrix that correspond to the templates
        for i in range(len(self.templates_indices_)-1):
            entries = get_distance_row_or_column_in_vector(self.min_distances_, self.templates_indices_[i])
            self.min_distances_[entries] = exact_distances[i]
            # we set the standard deviation of the exact distances to zero
            self.std_distances_[entries] = 0
            # we set the average of the exact distances to the exact distances
            self.avg_distances_[entries] = exact_distances[i]
        # we delete the exact distances to free memory
        del exact_distances
        gc.collect()

        # we monitor the memory usage
        print("Memory usage after replacing the exact distances:", psutil.virtual_memory().percent)

        # Assign the minimum distance matrix to the object attributes
    
    def recompute_distance_with_zscore(self, zscore=0):
        # Recompute the minimum distance matrix with the z-score
        self.min_distances_ = np.array(self.avg_distances_ + zscore*self.std_distances_, dtype=np.float32)
        self.zscore_ = zscore

def normalize_sample(x, x_min, x_max):
    """
    The function normalizes the samples of the array x using the min-max normalization.
    """
    # we normalize the samples of the array x using the min-max normalization
    x_norm = ((x - x_min) / (x_max - x_min)).copy()
    return x_norm

def get_distance_row_or_column_in_vector(distance_vector, i):
    """
    For a given distance matrix D of size N x N, we have the respective compact vector of size N(N-1)/2.
    The entry (i,j) of the matrix D is equivalent to the entry k of the compact vector, where 
    k = i*N - i*(i+3)/2 + j - 1.
    Therefore, the row i of the matrix D is equivalent to the entries k = i*N - i*(i+3)/2 + j - 1, for j = i+1, ..., N-1.
    Moreover, the column i of the matrix D is equivalent to the entries k = j*N - j*(j+3)/2 + i - 1, for j = 0, ..., i-1.
    """
    N = round(1/2 + np.sqrt(1/4 + 2*distance_vector.size))
    return [round(j*N - j*(j+3)/2 + i - 1) for j in range(i)] + [round(i*N - i*(i+3)/2 + j - 1) for j in range(i+1, N)]


def get_distance_position_in_vector(distance_vector, position_tuple):
    """
    For a given distance matrix D of size N x N, we have the respective compact vector of size N(N-1)/2.
    The entry (i,j) of the matrix D is equivalent to the entry k of the compact vector, where 
    k = i*N - i*(i+3)/2 + j - 1.
    """
    i, j = position_tuple
    N = 1/2 + np.sqrt(1/4 + 2*distance_vector.size)
    return round(i*N - i*(i+3)/2 + j - 1)

def remove_exact_distances(directory):
    # Get a list of all files in the directory
    all_files = os.listdir(directory)

    # Remove each file
    for file_name in all_files:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):  # Check if it is a file (not a directory)
            os.remove(file_path)
    return

def get_embedding_arrays(embeddings, grid_id):
    # we create a list of arrays with the node embeddings of each grid
    return embeddings[embeddings['grid']==grid_id].iloc[:,:-1].values, grid_id


# we especify this code to be run only if the file is run as a script
if __name__ == '__main__':

    grid_type = 'MV'
    # we raise an error if the grid_type is not MV or LV
    if grid_type not in ['MV', 'LV']:
        raise Warning('The grid_type must be MV or LV')
    # we check if the folders exist, and if not, we create them
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings')):
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings'))
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasserstein_models')):
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasserstein_models'))
    
    # we define the paths
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings')
    wasserstein_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wasserstein_models')
    ########################################################
    # STEP 1: Get the embeddings
    ########################################################

    
    write_embeddings = False
    
    n_cores = cpu_count() - 2
    # n_cores = 1


    if write_embeddings:
        # we read the data from the nodes embeddings
        embeddings = pd.read_csv(os.path.join(data_path, 'node_embeddings_{}.csv'.format(grid_type)), index_col=False)

        # embeddings is a dataframe with the node embeddings of different grids
        # the file has six columns, where the column 'grid' indicates the grid ID, and the other columns have the values of the node embeddings
        # for each grid, we would like to have a 2D numpy array with the node embeddings
        # we first get the unique grid IDs
        grid_ids = embeddings['grid'].unique()
        # we then create an array with the node embeddings of each grid

        # we count the time it takes to compute the embeddings
        start_time = time.time()
        partial_get_embeddings = partial(get_embedding_arrays, embeddings)
        
        if n_cores > 1:
            print("Reading the embeddings with parallel processing")
            jobs = [i for i in grid_ids]
            with Parallel(n_jobs=n_cores) as parallel:
                result_embeddings = parallel(delayed(partial_get_embeddings)(job) for job in jobs)

            # we unzip the result, to a list X with the embeddings and the grid_ids list. Each entry of results_embeddings is a tuple with the embeddings and the grid_id
            X, grid_ids = zip(*result_embeddings)
            
            print("Exporting the embeddings to a pickle file")

            dict_names = {i:grid_ids[i] for i in range(len(grid_ids))}
            with open(os.path.join(data_path, "grids_names_{}.pickle".format(grid_type)), "wb") as file:
                pickle.dump(dict_names, file)
            with open(os.path.join(data_path, "array_embeddings_{}.pickle".format(grid_type)), "wb") as file:
                pickle.dump(X, file)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time:", elapsed_time)
        else:
            X = [embeddings[embeddings['grid']==grid_id].iloc[:,:-1].values for grid_id in grid_ids]
            dict_names = {i:grid_ids[i] for i in range(len(grid_ids))}
            with open(os.path.join(data_path, "grids_names_{}.pickle".format(grid_type)), "wb") as file:
                pickle.dump(dict_names, file)
            with open(os.path.join(data_path, "array_embeddings_{}.pickle".format(grid_type)), "wb") as file:
                pickle.dump(X, file)
    else:
        # we check if the array embeddings file exists
        if not os.path.exists(os.path.join(data_path, 'array_embeddings_{}.pickle'.format(grid_type))):
            raise Warning('The file array_embeddings_{}.pickle does not exist. Please set write_embeddings to True to create the file.'.format(grid_type))
        # we read the embeddings
        with open(os.path.join(data_path, 'array_embeddings_{}.pickle'.format(grid_type)), 'rb') as handle:
            X = pickle.load(handle)
        # we read the overall line length of the grids
        with open(os.path.join(data_path, 'grids_names_{}.pickle'.format(grid_type)), 'rb') as handle:
            dict_names = pickle.load(handle)

    ########################################################
    # STEP 2: Normalize the embeddings
    ########################################################

    normalize_computation = False

    if normalize_computation:
        no_voltage_variation = {'grid_id': []}

        X_min = np.min(np.stack([np.min(x_vec, axis=0) for x_vec in X]), axis=0)
        X_max = np.max(np.stack([np.max(x_vec, axis=0) for x_vec in X]), axis=0)

        X_norm_results = []
        index_no_voltage_variation = []
        for i, x in enumerate(X):
            x_normalized = normalize_sample(x, X_min, X_max)
            if x_normalized.size == 0:
                no_voltage_variation['grid_id'].append(dict_names[i])
                index_no_voltage_variation.append(i)
            else:
                X_norm_results.append(x_normalized)

        # we save the names of the grids with no voltage variation
        no_voltage_variation_df = pd.DataFrame(no_voltage_variation)
        no_voltage_variation_df.to_csv(os.path.join(data_path, 'no_voltage_variation_{}.csv'.format(grid_type)), index=False)
        # we update the dictionary of grid names by deleting the grids with no voltage variation
        for i in index_no_voltage_variation:
            del dict_names[i]
        # we update the keys of the dictionary in order to have a continuous sequence of integers
        dict_names = {i:dict_names[k] for i, k in enumerate(dict_names.keys())}
        # we delete the unused variables to free memory
        del no_voltage_variation_df, index_no_voltage_variation, X
        gc.collect()

        with open(os.path.join(data_path, "normalized_grids_names_{}.pickle".format(grid_type)), "wb") as file:
            pickle.dump(dict_names, file)

        with open(os.path.join(data_path, "normalized_array_embeddings_{}.pickle".format(grid_type)), "wb") as file:
            pickle.dump(X_norm_results, file)

    else:
        if not os.path.exists(os.path.join(data_path, "normalized_array_embeddings_{}.pickle".format(grid_type))) \
            or not os.path.exists(os.path.join(data_path, "normalized_grids_names_{}.pickle".format(grid_type))):
            raise Warning('The normalized embeddings files do not exist. Please set normalize_computation to True to create the files.')
        # we read the normalized embeddings
        with open(os.path.join(data_path, "normalized_array_embeddings_{}.pickle".format(grid_type)), "rb") as handle:
            X_norm_results = pickle.load(handle)
        with open(os.path.join(data_path, "normalized_grids_names_{}.pickle".format(grid_type)), "rb") as handle:
            dict_names = pickle.load(handle)


    ########################################################
    # STEP 3: Compute the Wasserstein distances
    ########################################################

    # we set if we want to compute the Wasserstein distances or read them from a file
    compute_wasserstein = True
    
    if compute_wasserstein:

        # we set the number of additional reference templates
        K=24
        # we sample the exact distances
        sampling_exact_distances = 30000
        jobs = cpu_count()
        jobs = min(jobs, K)
        zscore = 0

        # we check if the folder exact_distances_parallel_{grid_type} exists, and if not, we create it
        if not os.path.exists(os.path.join(wasserstein_path, 'exact_distances_parallel_{}'.format(grid_type))):
            os.makedirs(os.path.join(wasserstein_path, 'exact_distances_parallel_{}'.format(grid_type)))

        # Compute Wasserstein distances through planes
        wass_model = Wasserstein()

        print("Getting the reference template and the LOT map")
        start_time_1 = time.time()
        wass_model.train_template(X_norm_results)
        end_time_1 = time.time()
        print("Elapsed time to train the template:", end_time_1 - start_time_1)
        print("\nComputing pairwise distances for the reference template")
        start_time_2 = time.time()
        wass_model.initial_pairwise_distances()
        end_time_2 = time.time()
        print("Elapsed time to compute the pairwise distances:", end_time_2 - start_time_2)
        # we write a txt file with the execution time of the initial plane
        with open(os.path.join(wasserstein_path, 'initial_plane_time_{}.txt'.format(grid_type)), 'w') as file:
            file.write(str(end_time_2 - start_time_1))
            
        print("\nAdding", K, "templates")
        start_time_3 = time.time()
        wass_model.add_templates(X_norm_results, K)
        end_time_3 = time.time()
        print("Elapsed time to add the templates:", end_time_3 - start_time_3)
        print("\nComputing pairwise distances for the reference template with parallel processing")

        # we check if there is the file with the exact distances from the previous computations
        if os.path.exists(os.path.join(wasserstein_path, 'exact_distances_parallel_{}'.format(grid_type))):
            # before computing the parallel distances, we remove the exact distances from the previous computations
            remove_exact_distances(os.path.join(wasserstein_path, 'exact_distances_parallel_{}'.format(grid_type)))
        # then we compute the parallel pairwise distances
        start_time_4 = time.time()
        wass_model.parallel_pairwise_distances(X_norm_results, grid_type, jobs, zscore)
        end_time_4 = time.time()
        
        print("Elapsed time to compute parallel pairwise distances:", end_time_4 - start_time_4)
        # we write a txt file with the execution time for the addition of planes
        with open(os.path.join(wasserstein_path, 'additional_distances_time_{}.txt'.format(grid_type)), 'w') as file:
            file.write(str(end_time_4 - start_time_3))

        print("Sampling {} exact distances".format(sampling_exact_distances))
        start_time = time.time()
        wass_model.sample_exact_distances(X_norm_results, num_samples=sampling_exact_distances)
        end_time = time.time()
        print("Elapsed time to sample exact distances:", end_time - start_time)

        # we store the wass_model object in a pickle file
        with open(os.path.join(wasserstein_path, 'wass_model_{}.pickle'.format(grid_type)), 'wb') as handle:
            pickle.dump(wass_model, handle)
        # and delete the wass_model object to free memory

    else:
        # we read the wass_model object
        with open(os.path.join(wasserstein_path, 'wass_model_{}.pickle'.format(grid_type)), 'rb') as handle:
            wass_model = pickle.load(handle)

    # ########################################################
    # # STEP 4: Evaluate the quality of the approximations
    # ########################################################
    evaluate_approximations = True

    if evaluate_approximations:
        # we evaluate the quality of the approximations
        zscore = -0.5

        Warr = np.array([wass_model.W_exact_[pair] for pair in wass_model.pairs_exact_])

        # we the distances from the min_distances_ array
        wass_model.recompute_distance_with_zscore(zscore)
        # we compute the relative errors
        distance_position = partial(get_distance_position_in_vector, wass_model.min_distances_)
        mdarr = np.array([wass_model.min_distances_[distance_position(pair)] for pair in wass_model.pairs_exact_])

        # the average overestimation of the Wasserstein can be computed using the heaviside function
        overestimation = np.heaviside(mdarr - Warr, 0)
        avg_overestimation = np.mean(overestimation * (mdarr - Warr)/Warr)
        # the average underestimation of the Wasserstein can be computed using the heaviside function
        underestimation = np.heaviside(Warr - mdarr, 0)
        avg_underestimation = np.mean(underestimation * (Warr - mdarr)/Warr)
        # we print the average overestimation and underestimation
        print("\nAverage overestimation for the multiplane distances:", avg_overestimation)
        print("Average underestimation for the multiplane distances:", avg_underestimation)
        # we compute the average relative error
        rel_error = np.mean(np.abs(Warr - mdarr)/Warr)
        print("Average relative error for the multiplane distances:", rel_error, "\n")

        # we also analyze the estimation error for the initial distances
        distance_position = partial(get_distance_position_in_vector, wass_model.initial_distances_)
        init_arr = np.array([wass_model.initial_distances_[distance_position(pair)] for pair in wass_model.pairs_exact_])

        # the average overestimation of the Wasserstein can be computed using the heaviside function
        overestimation = np.heaviside(init_arr - Warr, 0)
        avg_overestimation = np.mean(overestimation * (init_arr - Warr)/Warr)
        # the average underestimation of the Wasserstein can be computed using the heaviside function
        underestimation = np.heaviside(Warr - init_arr, 0)
        avg_underestimation = np.mean(underestimation * (Warr - init_arr)/Warr)
        # we print the average overestimation and underestimation
        print("Average overestimation for the initial distances:", avg_overestimation)
        print("Average underestimation for the initial distances:", avg_underestimation)
        # we compute the average relative error
        rel_error = np.mean(np.abs(Warr - init_arr)/Warr)
        print("Average relative error for the initial distances:", rel_error)