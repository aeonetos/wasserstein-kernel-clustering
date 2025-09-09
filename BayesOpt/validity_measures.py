from itertools import product
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from numpy import linalg as LA

# we define a function for the consensus index
def consensus_index(Y):
    """
    This function computes the consensus index of a clustering solution
    :param Y: the clustering solutions (each row is a clustering solution). Is a numpy array of shape (M, N), where M is the number of clustering solutions and N is the number of points in the dataset.
    :return: the consensus index
    """
    M = Y.shape[0]
    AMI = [adjusted_mutual_info_score(Y[i], Y[j], average_method='max') for i in range(M) for j in range(i)]
    return sum(AMI)/(M * (M-1)/2)

class FGK:
    """
    This class computes the Fast Goodman-Kruskal index (FGK) of a clustering solution.
    The index should be use to compare clustering solutions with the same number of clusters.
    """
    def __init__(self, n=100, m=35):
        """
        :param n: the number of pairs of points to sample for each cluster
        :param m: the number of times to repeat the sampling procedure
        The attributes of the class are:
        - n: the number of pairs of points to sample for each cluster
        - m: the number of times to repeat the sampling procedure
        - GK_scores_: the array of the GK index for each repetition
        - GK_: the mean of the GK index
        - GK_std_mean_: the standard deviation of the mean of the GK index
        - GK_CI95_: the 95% confidence interval for the mean of the GK index
        """
        self.n, self.m = n, m
        self.GK_scores_ = np.zeros(self.m)
        self.GK_ = 0
        self.GK_std_mean_ = 0
        self.GK_CI95_ = (0, 0)
    
    def fit(self, X, y):
        """
        This function computes the Fast Goodman-Kruskal index (FGK) of a clustering solution
        :param X: the dataset
        :param y: the labels of the points in the dataset
        :return: the FGK index, the mean of the FGK index, the standard deviation of the mean of the FGK index, and the 95% confidence interval for the mean
        The valid quadruples are n x n quadruples of points, where n is the number of pairs of points to sample.
        """
        # we initialize the array of the FGK index
        GK_scores = np.zeros(self.m)
        # we fix the seed for reproducibility
        np.random.seed(42)
        for i in range(self.m):
            # 1. we sample n labels from the labels of the points in the dataset with replacement
            unique_labels, cluster_size = np.unique(y, return_counts=True)
            points_in_cluster = {label: np.where(y == label)[0] for label in unique_labels}
            p_cluster_same = cluster_size[cluster_size>1]/cluster_size[cluster_size>1].sum()
            p_cluster_diff = cluster_size/cluster_size.sum()
            # we sample n points for the pairs in the same cluster, considering clusters with more than one point
            labels_same = np.random.choice(unique_labels[cluster_size>1], self.n, replace=True, p=p_cluster_same)
            labels_diff = np.array([np.random.choice(unique_labels, 2, replace=False, p=p_cluster_diff) for _ in range(self.n)])
            # 2. we sample n pairs of points from the points in the dataset
            pairs_same = np.array([np.random.choice(points_in_cluster[l], 2, replace=False) for l in labels_same])
            pairs_diff = np.array([[np.random.choice(points_in_cluster[l[0]]), np.random.choice(points_in_cluster[l[1]])] for l in labels_diff])
            # we ensure that the pairs are unique and that the number of pairs is n
            while pairs_same.shape[0] < self.n:
                labels_same = np.random.choice(unique_labels[cluster_size>1], self.n-pairs_same.shape[0], replace=True, p=p_cluster_same)
                pairs_same = np.concatenate((pairs_same, np.array([np.random.choice(points_in_cluster[l], 2, replace=False) for l in labels_same])), axis=0)
                pairs_same.sort(axis=1)
                pairs_same = np.unique(pairs_same, axis=0)
            while pairs_diff.shape[0] < self.n:
                labels_diff = np.array([np.random.choice(unique_labels, 2, replace=False, p=p_cluster_diff) for _ in range(self.n-pairs_diff.shape[0])])
                pairs_diff = np.concatenate((pairs_diff, np.array([[np.random.choice(points_in_cluster[l[0]]), np.random.choice(points_in_cluster[l[1]])] for l in labels_diff])), axis=0)
                pairs_diff.sort(axis=1)
                pairs_diff = np.unique(pairs_diff, axis=0)
            # 3. We compute the distances for the pairs within the same cluster and for the pairs in different clusters
            distance_same = LA.norm(X[pairs_same[:,0]] - X[pairs_same[:,1]], axis=1)
            distance_diff = LA.norm(X[pairs_diff[:,0]] - X[pairs_diff[:,1]], axis=1)
            # 4. We check the concordant and discordant quadruples
            pair_distances = np.array(list(product(distance_same, distance_diff)))
            total = pair_distances.shape[0]
            concordant = np.sum(pair_distances[:,0] < pair_distances[:,1])
            discordant = total - concordant
            GK_scores[i] = (concordant - discordant)/total
        self.GK_scores_ = GK_scores
        # we get the mean and standard deviation of the mean of the GK index
        self.GK_, self.GK_std_mean_ = GK_scores.mean(), GK_scores.std()/np.sqrt(self.m)
        # we get the 95% confidence interval for the mean
        self.GK_CI95_ = (self.GK_ - 1.96*self.GK_std_mean_, self.GK_ + 1.96*self.GK_std_mean_)
        return