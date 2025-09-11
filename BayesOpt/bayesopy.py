from bayes_opt import BayesianOptimization
import math
from bayes_opt import UtilityFunction
import pandas as pd
import time

# Let's start by defining our function, bounds, and instantiating an optimization object.
def black_box_function(x, y, z):
    # we want to count the evaluation time of the function
    init_time = time.time()
    evaluation =  math.exp(-z) * (-x ** 2 - (y - 1) ** 2 + 1) + x + z ** 2
    # we sleep for 0.1 seconds to simulate a more complex function
    time.sleep(0.005)
    # we get the elapsed time
    elap = (time.time() - init_time) * 1e3 # we convert the time to milliseconds
    return evaluation, elap

def get_solutions(results, elapsed_time):
    """
    This function takes in the results from the optimizer and returns a dataframe with the results.
    First, we get the variables and the target value from the first row of the results.
    Then, we add the variables and target value from the other rows one by one.
    Finally, we return the results as a dataframe.
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
    results_df['func_eval (ms)'] = elapsed_time
    return results_df

def bayes_optimizer(black_box_fun, util_fun, opt_obj, n_init, n_iter):
    """
    This function takes in the utility function, the optimizer, the number of initial points, and the number of iterations.
    First, we generate the initial points and register them with the optimizer.
    Then, we add points one by one, using the utility function to select the next one.
    Finally, we sort the results and return the optimizer and the results.
    """
    # we first generate some initial points
    init_variables, init_targets, elapsed_time = [], [], []
    for i in range(n_init):
        next_point = opt_obj.suggest(util_fun)
        target, elapsed = black_box_fun(**next_point)
        init_variables.append(next_point)
        init_targets.append(target)
        elapsed_time.append(elapsed)
    # we register the initial points with the target values to the optimizer
    for i in range(n_init):
        opt_obj.register(params=init_variables[i], target=init_targets[i])
    # we add points one by one, using the utility function to select the next one
    for _ in range(n_iter):
        next_point = opt_obj.suggest(util_fun)
        target, elapsed = black_box_fun(**next_point)
        elapsed_time.append(elapsed)
        opt_obj.register(params=next_point, target=target)
    # we get the results as a dataframe and sort them according to the target value
    results = get_solutions(opt_obj.res, elapsed_time)
    results.sort_values('target', ascending=False, inplace=True)
    return opt_obj, results

if __name__ == '__main__':
    optimizer = BayesianOptimization(
        f=None,
        pbounds={'x': (-2, 2), 'y': (-3, 3), 'z': (-4, 4)},
        verbose=2,
        random_state=1,
        allow_duplicate_points=True,
    )
    # we use the expected improvement acquisition function
    utility = UtilityFunction(kind="ei", xi=0.0)

    # we count the running time
    start_time = time.time()
    # we run the optimizer
    optimizer, results = bayes_optimizer(black_box_function, utility, optimizer, 100, 100)
    # we print the running time
    print("--- %s seconds ---" % (time.time() - start_time))
    # we display the top 5 solutions
    print(results.head(5))