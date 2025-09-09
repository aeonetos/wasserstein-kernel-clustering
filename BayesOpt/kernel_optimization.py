# Add extra paths
import sys
import os
sys.path.insert(0, os.path.abspath(''))
# we import the class WK_PCA from the module wasserstein_learn, which allows us to perform Wasserstein kernel PCA
from wasserstein_learn import WK_PCA
# we importthe class Wasserstein from the module wassesrstein_learn, which allows us to compute the Wasserstein distance
from wasserstein_learn import Wasserstein
# we import the bayesian optimization module and functions
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
# we import the rest of the libraries
import pickle
import numpy as np
import math
from functools import partial
import time 
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed

def kernel_dispersion(distances, gamma):
    """
    This function takes in a matrix of distances and a value for gamma and returns the dispersion of the kernel matrix.
    """
    km = WK_PCA()
    km.gaussian_kernel(distances, math.pow(10, gamma))      
    km.KPCA()
    return km.robust_dispersion_

def composed_kernel_dispersion(ker_demand, ker_node, ker_edge, b0, b1, b2, b3, b4, b5, b6):
    km = WK_PCA()
    km.kernel_matrix_ = b0 * ker_edge.kernel_matrix_ + b1 * ker_node.kernel_matrix_ + b2 * ker_demand.kernel_matrix_ \
        + b3 * np.multiply(ker_edge.kernel_matrix_, ker_node.kernel_matrix_) + b4 * np.multiply(ker_demand.kernel_matrix_, ker_node.kernel_matrix_) \
            + b5 * np.multiply(ker_demand.kernel_matrix_, ker_edge.kernel_matrix_) + b6 * np.multiply(ker_demand.kernel_matrix_, np.multiply(ker_edge.kernel_matrix_, ker_node.kernel_matrix_))
    km.KPCA()   
    return km.robust_dispersion_

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

def random_optimizer(black_box_fun, n_init=300):
    # # we generate a regular grid
    # b = np.linspace(0,1,spacing)
    # # we define a meshgrid
    # x1, x2, x3, x4, x5, x6, x7 = np.meshgrid(b,b,b,b,b,b,b)
    # # we exclude the points that do not satisfy the condition that the sum of the weights is 1
    # x_sum = x1 + x2 + x3 + x4 + x5 + x6 + x7
    # # we get only the combination of weights that sum to 1
    # x1c, x2c, x3c, x4c, x5c, x6c, x7c = x1[np.where(x_sum == 1)], x2[np.where(x_sum == 1)], x3[np.where(x_sum == 1)], x4[np.where(x_sum == 1)], x5[np.where(x_sum == 1)], x6[np.where(x_sum == 1)], x7[np.where(x_sum == 1)]
    # # we want to stack the 1-dimensional arrays x1c, x2c, ..., x7c into a matrix Xc with 7 columns and rows equal to x1c.size = x2c.size = ... = x7c.size
    # Xc = np.column_stack((x1c, x2c, x3c, x4c, x5c, x6c, x7c))
    # # we initialize the target values
    # targets = []
    # # we compute the target values in parallel
    # num_cores = multiprocessing.cpu_count() - 4
    # print("Computing the target values in parallel...")
    # targets = Parallel(n_jobs=num_cores)(delayed(black_box_fun)(*Xc[j]) for j in range(Xc.shape[0]))
    # # we stack the Xc values with the target values in a result matrix
    # results = np.column_stack((Xc, targets))
    # results_df = pd.DataFrame(results, columns=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'target'])
    # results_df.sort_values('target', ascending=False, inplace=True)  
    # we generate a regular grid
    # b = np.linspace(0,1,spacing)
    # # we define a meshgrid
    # x1, x2, x3 = np.meshgrid(b,b,b)
    # # we exclude the points that do not satisfy the condition that the sum of the weights is 1
    # x_sum = x1 + x2 + x3
    # # we get only the combination of weights that sum to 1
    # x1c, x2c, x3c = x1[np.where(x_sum == 1)], x2[np.where(x_sum == 1)], x3[np.where(x_sum == 1)]

    # we generate random points
    # Reference here: sampling from standard simplex (This is a Dirichlet distribution)
    # https://mathoverflow.net/questions/76255/random-sampling-a-linearly-constrained-region-in-n-dimensions
    X = np.random.exponential(scale=1, size=(n_init, 7))
    Xc = X / X.sum(axis=1).reshape((-1,1))
    # we initialize the target values
    targets = []
    # we compute the target values in parallel
    num_cores = multiprocessing.cpu_count() - 4
    print("Computing the target values in parallel...")
    targets = Parallel(n_jobs=num_cores)(delayed(black_box_fun)(*Xc[j]) for j in range(Xc.shape[0]))
    # we stack the Xc values with the target values in a result matrix
    results = np.column_stack((Xc, targets))
    results_df = pd.DataFrame(results, columns=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'target'])
    results_df.sort_values('target', ascending=False, inplace=True)  
    return results_df

def bayes_optimizer(black_box_fun, utility_fun, opt_obj, n_iter, n_init=0, warm_start=None):
    """
    This function takes in a black box function, an utility function, an optimizer, and the number of iterations.
    If the number of initial points is specified, we generate the initial points and register them with the optimizer.
    Then, we add points one by one, using the utility function to select the next one.
    Finally, we return the optimizer and the results as a dataframe.
    """
    if warm_start is None:
        # we first generate some initial points
        init_variables, init_targets = [], []
        # we print the initial points
        print('Initial points:')
        for i in range(n_init):
            next_point = opt_obj.suggest(utility_fun)
            #################
            b_keys = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
            if all([b in next_point.keys() for b in b_keys]):
                b_scale = sum([next_point[key] for key in b_keys])
                for key in b_keys:
                    next_point[key] = (next_point[key])/b_scale
            #################
            target = black_box_fun(**next_point)
            init_variables.append(next_point)
            init_targets.append(target)
            print('\nPoint {}: {}'.format(i+1, next_point))
            print('Target {}: {}'.format(i+1, target))
        # we register the initial points with the target values to the optimizer
        for i in range(n_init):
            opt_obj.register(params=init_variables[i], target=init_targets[i])
        # we add points one by one, using the utility function to select the next one
        # we print the iteration points
        print('\nIteration points:')
        for _ in range(n_iter):
            next_point = opt_obj.suggest(utility_fun)
            #################
            b_keys = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
            if all([b in next_point.keys() for b in b_keys]):
                b_scale = sum([next_point[key] for key in b_keys])
                for key in b_keys:
                    next_point[key] = next_point[key]/b_scale
            #################
            target = black_box_fun(**next_point)
            opt_obj.register(params=next_point, target=target)
            print('\nPoint {}: {}'.format(n_init+1+_, next_point))
            print('Target {}: {}'.format(n_init+1+_, target))
        # we get the results as a dataframe and sort them according to the target value
        results = get_solutions(opt_obj.res)
        results.sort_values('target', ascending=False, inplace=True)
    else:
        # we get a list of dictionaries with the variables 'lambda_penalty', 'gamma', 'pca_variance' for each row in the results dataframe
        variables = warm_start.columns.to_list()
        # we remove ['target'] from the variables list
        obj_values = ['target']
        variables = [v for v in variables if v not in obj_values]
        warm_points = warm_start[variables].to_dict('records')
        # we get a dictionary with the target values, the running time, and the number of clusters for each row in the results dataframe
        warm_targets = warm_start[obj_values].to_dict('list')
        # we register the initial points with the target values to the optimizer
        for i in range(len(warm_points)):
            opt_obj.register(params=warm_points[i], target=warm_targets['target'][i])
        # we add points one by one, using the utility function to select the next one
        # we print the iteration points
        print('\nIteration points:')
        for _ in range(n_iter):
            next_point = opt_obj.suggest(utility_fun)
            #################
            b_keys = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
            if all([b in next_point.keys() for b in b_keys]):
                b_scale = sum([next_point[key] for key in b_keys])
                for key in b_keys:
                    next_point[key] = (next_point[key])/b_scale
            #################
            target = black_box_fun(**next_point)
            opt_obj.register(params=next_point, target=target)
            print('\nPoint {}: {}'.format(_, next_point))
            print('Target {}: {}'.format(_, target))
        # we get the results as a dataframe and sort them according to the target value
        results = get_solutions(opt_obj.res)
        results.sort_values('target', ascending=False, inplace=True)    
    return opt_obj, results

if __name__ == '__main__':
    
    kernel_type = 'composed'

    if kernel_type == 'single':
        name = 'demand'
        # we only allow a warm start if the results are available
        warm = False

        if warm:
            warm = os.path.isfile('BayesOpt/kernels/kernel_{}_opt.csv'.format(name))
        
        if name == 'demand':
            # we read the embeddings
            with open('Embeddings/node_embeddings.pkl', 'rb') as handle:
                node_embeddings = pickle.load(handle)

            with open("wasserstein_node.pickle", "rb") as file:
                wasserstein = pickle.load(file)

            # we get the data
            grid_ids = node_embeddings['grid'].unique()
            X = [node_embeddings[node_embeddings['grid']==grid_id].iloc[:,:-1].values for grid_id in grid_ids]
            wasserstein.penalty_matrix(X, penalty_type='entry', penalize_entry=0)
            distances = wasserstein.penalty_matrix_
        else:
            with open("wasserstein_{}.pickle".format(name), "rb") as file:
                wasserstein = pickle.load(file)
            distances = wasserstein.min_distances_
        
        black_box_function = partial(kernel_dispersion, distances)

        # we define the ranges of the hyperparameters
        hyp_ranges = {'gamma': (-5, 6)} # gamma is in scientific notation
        # we define the optimizer
        optimizer = BayesianOptimization(f=None, pbounds=hyp_ranges, verbose=2, random_state=1, allow_duplicate_points=True)
        # we use the expected improvement acquisition function
        utility_function = UtilityFunction(kind="ei", xi=0.0)

        if warm:
            # we read the results from the csv
            read_results = pd.read_csv('BayesOpt/kernels/kernel_{}_opt.csv'.format(name))

            # we count the running time
            start_time = time.time()
            # we run the optimizer
            optimizer, results = bayes_optimizer(black_box_function, utility_function, optimizer, 10, warm_start=read_results)
        else:
            # we count the running time
            start_time = time.time()
            # we run the optimizer
            optimizer, results = bayes_optimizer(black_box_function, utility_function, optimizer, 20, n_init=100)        

        # we print the running time
        print("--- %s seconds ---" % (time.time() - start_time))
        # we display the top 5 solutions
        print(results.head(5))
        # save the results as a csv
        results.to_csv('BayesOpt/kernels/kernel_{}_opt.csv'.format(name), index=False)
    
    elif kernel_type == 'composed':
        
        optimization = 'random'
        
        ##################
        # best gammas according to kernel dispersion
        # gamma_edge = 3.336 ; gamma_node = 1.987, gamma_demand = -0.291
        ##################

        with open('Embeddings/node_embeddings.pkl', 'rb') as handle:
            node_embeddings = pickle.load(handle)
        with open("wasserstein_node.pickle", "rb") as file:
            wass_node = pickle.load(file)
        with open("wasserstein_edge.pickle", "rb") as file:
            wass_edge = pickle.load(file)

        grid_ids = node_embeddings['grid'].unique()
        X = [node_embeddings[node_embeddings['grid']==grid_id].iloc[:,:-1].values for grid_id in grid_ids]
        wass_node.penalty_matrix(X, penalty_type='entry', penalize_entry=0)
        demand_distances = wass_node.penalty_matrix_
        node_distances = wass_node.min_distances_
        edge_distances = wass_edge.min_distances_

        km_node = WK_PCA()
        km_node.gaussian_kernel(node_distances, math.pow(10, 1.987))      
        km_node.KPCA()
        km_edge = WK_PCA()
        km_edge.gaussian_kernel(edge_distances, math.pow(10, 3.336))      
        km_edge.KPCA()  
        km_demand = WK_PCA()
        km_demand.gaussian_kernel(demand_distances, math.pow(10, -0.291))      
        km_demand.KPCA()          

        black_box_function = partial(composed_kernel_dispersion, km_demand, km_node, km_edge)

        if optimization == 'bayes':
            # we define the ranges of the hyperparameters
            hyp_ranges = {'b0': (0, 1), 'b1': (0, 1), 'b2': (0, 1), 'b3': (0, 1), 'b4': (0, 1), 'b5': (0, 1), 'b6': (0, 1)} # gamma is in scientific notation

            # we define the optimizer
            optimizer = BayesianOptimization(f=None, pbounds=hyp_ranges, verbose=2, random_state=1, allow_duplicate_points=True)
            # we use the expected improvement acquisition function
            utility_function = UtilityFunction(kind="ei", xi=0.0)
            
            # we count the running time
            start_time = time.time()
            # we run the optimizer
            optimizer, results = bayes_optimizer(black_box_function, utility_function, optimizer, 30, n_init=100)        

            # we print the running time
            print("--- %s seconds ---" % (time.time() - start_time))
            # we display the top 5 solutions
            print(results.head(5))        
            results.to_csv('BayesOpt/kernels/kernel_composed_opt.csv', index=False)
        
        elif optimization == 'random':
            # we count the running time
            start_time = time.time()
            # we run the random optimizer
            results = random_optimizer(black_box_function)
            # we print the running time
            print("--- %s seconds ---" % (time.time() - start_time))
            # we display the top 5 solutions
            print(results.head(5))        
            results.to_csv('BayesOpt/kernels/kernel_composed_opt.csv', index=False)