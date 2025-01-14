import os
import numpy as np
import torch
import time as t
import random

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

def batch_pairwise_distances(U: torch.tensor, V: torch.tensor, distance: str = "euclidean"):
    """Compute pairwise distances between two batches of feature vectors using the specified distance metric..
    ----
    Parameters:
        U (torch.tensor): first feature vector (shape: [batch_size_1, feature_dim]).
        V (torch.tensor): second feature vector (shape: [batch_size_2, feature_dim]).
        distance (str): Distance metric to use ("euclidean" or "manhattan"). Defaults to "euclidean".
    Returns:
        tensor of pairwise distances (shape: [batch_size_1, batch_size_2]). """

    if distance == "euclidean":
        D = torch.cdist(U, V, p = 2) # p=2 specifies Euclidean distance
    elif distance == "manhattan":
        D = torch.cdist(U, V, p = 1)  # p=1 specifies Manhattan distance
    else:
        raise ValueError(f"Unsupported distance metric: {distance}. Use 'euclidean' or 'manhattan'.")

    return D


class ManifoldEstimator():
    """Estimates the manifold of given feature vectors."""

    def __init__(self, features, distance = "euclidean", row_batch_size=20000, col_batch_size=20000,
                 nhood_sizes=[50], clamp_to_percentile=None, eps=1e-5):
        """Estimate the manifold of given feature vectors.

            Args:
                features (np.array/tf.Tensor): Matrix of feature vectors to estimate their manifold.
                distance (str): Distance metric to use ("euclidean" or "manhattan"). Defaults to "euclidean".
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
                clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                    the given percentile.
                eps (float): Small number for numerical stability.
        """
        batch_size = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        self.D = np.zeros([batch_size, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros(
            [row_batch_size, batch_size], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, batch_size, row_batch_size):
            end1 = min(begin1 + row_batch_size, batch_size)
            row_batch = features[begin1:end1]

            for begin2 in range(0, batch_size, col_batch_size):
                end2 = min(begin2 + col_batch_size, batch_size)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0:end1 - begin1,
                               begin2:end2] = batch_pairwise_distances(row_batch,
                                                                       col_batch, distance = distance)

            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(
                distance_batch[0:end1 - begin1, :], seq, axis=1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(
            self,
            eval_features,
            distance = "euclidean",
            return_realism=False,
            return_neighbors=False):
        """Evaluate if new feature vectors are at the manifold.
        """
        num_eval = eval_features.shape[0]
        num_ref = self.D.shape[0]
        distance_batch = np.zeros(
            [self.row_batch_size, num_ref], dtype=np.float32)
        batch_predictions = np.zeros(
            [num_eval, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval,], dtype=np.float32)
        nearest_indices = np.zeros([num_eval,], dtype=np.int32)

        for begin1 in range(0, num_eval, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1 - begin1,
                               begin2:end2] = batch_pairwise_distances(feature_batch,
                                                                       ref_batch, distance = distance)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of
            # neighborhood size k.
            samples_in_manifold = distance_batch[0:end1 -
                                                 begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(
                samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(
                self.D[:, 0] / (distance_batch[0:end1 - begin1, :] + self.eps), axis=1)
            nearest_indices[begin1:end1] = np.argmin(
                distance_batch[0:end1 - begin1, :], axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions


def knn_precision_recall_features(
        ref_features,
        eval_features,
        nhood_sizes=[50],
        distance = "euclidean",
        row_batch_size=20000,
        col_batch_size=20000,
        num_gpus=1):
    """Calculates k-NN precision and recall for two sets of feature vectors.

        Args:
            ref_features (np.array/tf.Tensor): Feature vectors of reference samples.
            eval_features (np.array/tf.Tensor): Feature vectors of generated samples.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.
            distance (str): Distance metric to use ("euclidean" or "manhattan"). Defaults to "euclidean".
            row_batch_size (int): Row batch size to compute pairwise distances
                (parameter to trade-off between memory usage and performance).
            col_batch_size (int): Column batch size to compute pairwise distances.
            num_gpus (int): Number of GPUs used to evaluate precision and recall.
        Returns:
            State (dict): Dict that contains precision and recall calculated from
            ref_features and eval_features.
    """
    state = dict()
    num_data = ref_features.shape[0]
    num_features = ref_features.shape[1]

    # Initialize ManifoldEstimators.
    ref_manifold = ManifoldEstimator(
        ref_features,
        distance,
        row_batch_size,
        col_batch_size,
        nhood_sizes)
    eval_manifold = ManifoldEstimator(
        eval_features,
        distance,
        row_batch_size,
        col_batch_size,
        nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    start = t.time()

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features, distance)
    state['precision'] = precision.mean(axis=0)

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features, distance)
    state['recall'] = recall.mean(axis=0)

    print('Evaluated k-NN precision and recall in: %gs' % (t.time() - start))

    return state

def get_precision_recall(
        ref: torch.tensor,
        eval: torch.tensor,
        ks: list = [50],
        distance: str = "euclidean"):
    """
    Compute precision and recall between datasets.
    ----
    Parameters:
        ref (torch.tensor): First data set of comparison.
        eval (torch.tensor): Second dataset to use for comparison.
        ks (list): Number of neighbors used to estimate the data manifold.
        distance (str): Distance metric to use ("euclidean" or "manhattan"). Defaults to "euclidean".
    Returns:
        tuple with precision and recall.
    """

    # Calculate k-NN precision and recall.
    precision_recall_state = knn_precision_recall_features(
        ref, eval,ks, distance)

    precision = precision_recall_state['precision']
    recall = precision_recall_state['recall']

    return (precision, recall)
    

def get_realism_score(ref: torch.tensor, eval: torch.tensor, distance: str = "euclidean"):
    """
    Compute realism score between datasets.
    ----
    Parameters:
        ref (torch.tensor): First data set of comparison.
        eval (torch.tensor): Second dataset to use for comparison.
        distance (str): Distance metric to use ("euclidean" or "manhattan"). Defaults to "euclidean".
    Returns:
        Maximum realism score.
    """

    # Estimate manifold of real images.
    print('Estimating manifold of real data...')
    ref_manifold = ManifoldEstimator(ref, distance=distance, clamp_to_percentile=50)

    # Estimate quality of individual samples.
    _, realism_scores = ref_manifold.evaluate(eval, distance=distance, return_realism=True)

    return realism_scores
