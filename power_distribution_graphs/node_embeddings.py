import os
import sys
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
from multiprocessing import cpu_count, Pool
import time

def parallel_grid_embedding(grid_files_path, njobs=20, bunches=20):
    # we pick bunches of files in the jobs list. The last bunch may have less files, so we add the rest of the files
    nbunches = len(grid_files_path) // bunches
    jobs = [[grid_files_path[i*bunches:(i+1)*bunches]] for i in range(nbunches)]
    jobs += [[grid_files_path[nbunches*bunches:]]]
    # we do parallel computing
    with Pool(njobs) as p:
        results = p.starmap(grids_embeddings, jobs)
    p.close()
    # we concatenate the results
    node_embeddings = pd.concat([result[0] for result in results], ignore_index=True, axis=0)
    grid_no_power_flow = []
    for result in results:
        grid_no_power_flow += result[1]
    grid_no_power_flow = pd.Series(grid_no_power_flow)
    return node_embeddings, grid_no_power_flow

def grids_embeddings(grid_files_path):
    node_embeddings = pd.DataFrame()
    grid_no_power_flow = []
    for grid_file_path in grid_files_path:
        # we get the id of the grid
        grid_id = grid_file_path.split('/')[-1]
        # we read the geodataframe of the nodes
        nodes, edges = gpd.read_file(grid_file_path + '_nodes'), gpd.read_file(grid_file_path + '_edges')
        # we consolidate the geometry of the nodes
        nodes[['x', 'y']] = pd.DataFrame([nodes['geometry'].x, nodes['geometry'].y]).T
        # we set the columns key, u, v as multiindex in edges
        # we set the osmid as index in nodes
        nodes = nodes.set_index('osmid')
        # we transform the type of the index to string
        nodes.index = nodes.index.astype(str)
        edges = edges.set_index(['key', 'u', 'v'])
        # we also change the type of the index to string
        edges.index = edges.index.set_levels([edges.index.levels[0], edges.index.levels[1].astype(str), edges.index.levels[2].astype(str)])
        edges.columns.tolist()
        # we get the index of the rows with nan values in the edges dataframe
        nan_rows = edges[edges.isnull().any(axis=1)].index
        # and we drop the rows with nan values
        if len(nan_rows) > 0:
            edges.drop(nan_rows, inplace=True)
        # we check if the grid has power flow computations
        if 'load' in edges.columns:
            # we add a column with the power_flow of each edge
            # this is the product of the load and the s_nom
            edges['power_flow'] = edges['load'] * edges['s_nom']
            # we select the edge properties that we want to consider
            edges_features = []
            centrality = {}
            # we build the graphs from the nodes and edges geodataframes
            for edge_feature in edges_features:
                g = ox.graph_from_gdfs(nodes, edges[[edge_feature] + ['geometry']])
                # we can transform the multigraph into a graph, changing the edges of the type (0, u, v) to (u, v)
                if edge_feature in ['s_nom', 'b', 'load']:
                    g_graph = nx.Graph([(v, k, {edge_feature: 1/d[edge_feature]}) for u, v, k, d in g.edges(keys=True, data=True)])
                else:
                    g_graph = nx.Graph([(v, k, {edge_feature: d[edge_feature]}) for u, v, k, d in g.edges(keys=True, data=True)])
                # we want to get the centrality of the graph g_graph for each edge considering the edge feature 
                centrality[edge_feature] = nx.closeness_centrality(g_graph, distance=edge_feature)
            # then, each entry of centrality is a dictionary
            # we can transform it into a pandas dataframe
            node_df = pd.DataFrame(centrality)
            # morevoer, we can add the demand and the voltage of each node
            node_df['demand'] = pd.Series(nodes['el_dmd'].to_dict())
            node_df['voltage'] = pd.Series(nodes['voltage'].to_dict())
            # we check if there are none values in the dataframe
            if node_df.isnull().values.any():
                # if there are none values, we replace them with 0
                node_df = node_df.fillna(0)
            # we also put the name of the grid in the dataframe
            node_df['grid'] = grid_id
            # we update the node_embedding and length_sum dataframes
            node_embeddings = pd.concat([node_embeddings, node_df], ignore_index=True, axis=0)
        else:
            # if there are no power flow computations
            # we add the grid_id to the list of grids without power flow computations
            grid_no_power_flow.append(grid_id)
    return node_embeddings, grid_no_power_flow

if __name__ == '__main__':
    # we get the name of the grids in the path
    grid_type = 'LV'
    # we raise a warning if the grid_type is not MV or LV
    if grid_type not in ['MV', 'LV']:
        raise Warning('The grid_type must be MV or LV')
    # we get the number of cpus
    # if the nbojs is bigger than one, we do parallel computing
    njobs = cpu_count() - 2
    # njobs = 1
    # we check if the folders exist, and if not, we raise a warning
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\PDGs_{}'.format(grid_type))):
        raise Warning('The folder data/PDGs_{} does not exist'.format(grid_type) + '\n Please, download the data from the data repository!')

    if grid_type == 'MV':
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\PDGs_MV')
        files = os.listdir(path)
        # we get the file names such that they contain the string 'edges'
        files = [file for file in files if 'edges' in file]
        files = [file.replace('_edges', '') for file in files]
        # and we add the path to the file name
        files = [path + '/' + file for file in files]

    elif grid_type == 'LV':
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\PDGs_LV')
        files = os.listdir(path)
        # we first get the names of the folders in the path
        folders = os.listdir(path)
        # then, for each folder, we get the names of the files
        # and we add the path to the file name
        # we remove the strings '_edges' and '_nodes' from the string in files
        # and we remove the duplicates
        files = []
        for folder in folders:
            folder_files = os.listdir(path + '/' + folder)
            # we get the file names such that they contain the string 'edges'
            folder_files = [file for file in folder_files if 'edges' in file]
            # we delete the string '_edges' from the file names
            folder_files = [file.replace('_edges', '') for file in folder_files]
            # then, we add the path to the file names
            files += [path + '/' + folder + '/' + file for file in folder_files]

    # Start the timer
    start_time = time.time()
    if njobs == 1:
        node_embeddings, grid_no_power_flow = grids_embeddings(files)
        grid_no_power_flow = pd.Series(grid_no_power_flow)
    else:
        node_embeddings, grid_no_power_flow = parallel_grid_embedding(files, njobs=njobs)
    export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings')
    # we export the node embeddings dataframe to a csv file
    node_embeddings.to_csv(os.path.join(export_path, 'node_embeddings_{}.csv'.format(grid_type)), index=False)
    # we export the list of grids without power flow computations to a csv file
    grid_no_power_flow.to_csv(os.path.join(export_path, 'grid_no_power_flow_{}.csv'.format(grid_type)), index=False)
    # Stop the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print("\n", elapsed_time)
    print('Done!')