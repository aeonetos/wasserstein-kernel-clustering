# Wasserstein-based Kernel Clustering

This repository contains the code used to reproduce the experiments of the
paper **“Wasserstein-based Kernel Principal Component Analysis for Clustering
Applications.”**  Two families of datasets are considered:

* **Time-series:** electricity demand and pedestrian counts.
* **Power distribution graphs:** low- and medium-voltage grid models.

The experiments explore how kernels constructed from Wasserstein distances can
be used within Kernel PCA and subsequent clustering algorithms.

## Repository structure

```text
BayesOpt/                  Bayesian optimisation helpers
power_distribution_graphs/ Scripts for graph experiments
time_series/               Notebooks and utilities for time-series experiments
```

### Time-series experiments

The `time_series` directory provides two Jupyter notebooks:

* `italypower.ipynb`
* `melbournepedestrian.ipynb`

Both notebooks rely on the helper functions in
`time_series/time_toolbox.py`.  The workflow typically consists of:

1. Loading a dataset.
2. Computing power spectral densities and Wasserstein distances.
3. Generating a Wasserstein kernel and extracting a feature map via Kernel
   PCA.
4. Clustering the feature map with K-medoids and evaluating consensus-based
   validity indices.

### Power distribution graph experiments

The `power_distribution_graphs` directory contains scripts for handling graph
data and running clustering experiments:

* `node_embeddings.py` – computes node-level features for each grid.
* `wasserstein_computations.py` – approximates Wasserstein distances between
  grids using the Lot–mapping approach.
* `wass_kernel_clustering.py` – builds composite kernels from the distances and
  performs clustering with Bayesian optimisation of hyperparameters.
* `davies_bouldin_validity.py` – recomputes Wasserstein-based dissimilarities,
  clusters the resulting feature maps and assesses the solutions with the
  Davies–Bouldin index.

Each script can be executed directly and contains inline comments describing
the required input files and the produced output.

## Requirements

Python dependencies are listed in `requirements.txt`.  The code was developed
with Python 3.10; installing the requirements into a virtual environment is
recommended:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the experiments

### Time-series

Open the desired notebook under `time_series/` and run the cells in order.  The
datasets must be downloaded from the UCR Time Series Archive.

### Power distribution graphs

The datasets must be downloaded from https://doi.org/10.5281/zenodo.15167589

1. Run `power_distribution_graphs/node_embeddings.py` to build node-level
   embeddings for each grid.
2. Use `power_distribution_graphs/wasserstein_computations.py` to compute
   Wasserstein distance matrices.  This step can leverage parallel processing
   and may take considerable time.
3. Execute `power_distribution_graphs/wass_kernel_clustering.py` to build the
   composed kernels and perform clustering.  Hyperparameters are tuned through
   Bayesian optimisation.
4. Optionally, evaluate the clustering quality with
   `power_distribution_graphs/davies_bouldin_validity.py`.  Setting the
   `compute_mv_distances`/`compute_lv_distances` flags to `True` recomputes the
   Wasserstein-1 distance matrices from the raw data and stores them in
   `power_distribution_graphs/context_validity/`.  Cluster assignments are
   cached under `power_distribution_graphs/context_validity/clustered_grids/`
   before generating Davies–Bouldin summaries.

## Citing

If you find this repository useful, please cite the associated paper.
 
