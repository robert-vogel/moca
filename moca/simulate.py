"""Simulate data

By: Robert Vogel
"""
import numbers

import numpy as np
# from scipy.special import ndtri

from moca import stats


def _sample_network(n, seed=None):
    """Sample a simple 1-d spring network

    The network matrix is symmetry, with positive off-
    diagonal elements.  Let network by the n by n matrix,
    then it follows that

    i != j
    network_{ij} > 0

    network_{ii} > -\sum_{j != i} network_{ij}

    Args:
        n: (int)
            dimension of the matrix
        seed:
            any satisfactory input to np.random.default_rng

    Returns:
        ((n, n) np.ndarray)
    """
    rng = np.random.default_rng(seed=seed)

    network = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            if i < j:
                network[i, j] = rng.uniform()
                network[j, i] = network[i, j]

            network[i, i] -= network[i, j]

        # these the spring connecting are the springs
        # connecting a mass to the fixed points
        network[i, i] -= rng.uniform(low=2, high=4)

    return network


def make_corr_matrix(m_classifiers, independent=False,
                     seed=None):
    """Make correlation matrix from network
    
    Args:
        m_classifiers: (int)
            number of classifier predictions to simuluate
        independent: (bool)
            True: then samples are drawn independently
            False: sample are correlated, correlations
                are draw randomly
        seed:
            any satisfactory input to np.random.default_rng

    Return:
        ((m_classifiers, m_classifiers) np.ndarray) random
        correlation matrix
    """
    rng = np.random.default_rng(seed=seed)

    max_network_iteration = 100

    if m_classifiers < 3:
        raise ValueError("Unmet requirement: m_classifiers >= 3")

    if independent:
        return np.eye(m_classifiers)

    # Sample a random, fully connected, and symmetric 
    # spring network, keep sampling until matrix is full
    # rank

    network = _sample_network(m_classifiers, seed=rng)

    i = 0
    while (np.linalg.matrix_rank(network) != m_classifiers
           and i < max_network_iteration):

        network = _sample_network(m_classifiers, seed=rng)
        i += 1

    if i == max_network_iteration:
        raise ValueError(f"Rank {m_classifiers} not found.")

    # Recall that covariance = -0.5 network ^{-1} noise matrix
    corr = -np.linalg.inv(network)
    
    # normalize covariance matrix so that it is a
    # correlation matrix
    for i in range(m_classifiers):
        for j in range(m_classifiers):

            if i == j:
                continue
    
            corr[i, j] = corr[i,j] / np.sqrt(corr[i,i]*corr[j,j])

    for i in range(m_classifiers):
        corr[i, i] = 1

    return corr


def gaussian_scores(n_samples, n_pos, auc, corr_matrix, seed=None):
    """Base classifier predictions and sample labels

    Args: 
        n_samples: (int)
            number of samples to draw
        n_pos: (int)
            number of samples from the positive class
        auc: ((M classifiers,) np.ndarray)
            array of base classifier AUC
        corr_matrix: ((M classifier, M classifier) np.ndarray)
            class conditioned correlation matrix, assumes same
            matrix for each class, default is the identity 
            matrix
        seed:
            any satisfactory input to np.random.default_rng

    Returns:
        scores: ((m classifiers, n sample) np.ndarray)
        labels: ((n sample,) np.ndarray)
    """
    if n_pos <= 0 or n_pos > n_samples:
        raise ValueError(("Unmet requirement: n_samples > n_pos"
                            "and n_pos > 0"))

    if not stats.is_auc(auc):
        raise ValueError("Not valid AUC value(s).")

    rng = np.random.default_rng(seed=seed)

    delta = stats.auc_to_delta(auc,
                               np.diag(corr_matrix),
                               np.diag(corr_matrix))

    labels = np.zeros(n_samples)
    labels[:n_pos] = 1

    rng.shuffle(labels)

    scores = np.zeros(shape=(auc.size, n_samples))
    _mu = np.zeros(auc.size)

    for j, label in enumerate(labels):
        s = rng.multivariate_normal(_mu,
                                    corr_matrix,
                                    size=1)
        if label == 1:
            s += delta

        scores[:, j] = s

    return scores, labels


def rank_scores(n_samples, n_pos, auc, corr_matrix, seed=None):
    """Simulate rank scores and class labels.

    Args: 
        n_samples: (int)
            number of samples to draw
        n_pos: (int)
            number of samples from the positive class
        prevalence: (float)
            fraction of samples from the positive class
        auc: ((M classifiers,) np.ndarray)
            array of base classifier AUC
        corr_matrix: ((M classifier, M classifier) np.ndarray)
            class conditioned correlation matrix, assumes same
            matrix for each class, default is the identity 
            matrix
        seed:
            any satisfactory input to np.random.default_rng
    
    Returns:
        scores: ((m classifiers, n sample) np.ndarray)
            rank order sample scores for each classifier
        labels: ((n sample,) np.ndarray)
            binary labels, (0,1)
    """
    scores, labels = gaussian_scores(n_samples, n_pos,
                                     auc, corr_matrix,
                                     seed=seed)

    return stats.rank_transform(scores), labels
