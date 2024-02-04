
import numpy as np
from . import stats


def kfold(data, labels, kfolds, ascending=False, seed=None):
    """Cross validation subsets of rank and labeled data.

    Args:
        data: ((M methods, N sample) np.ndarray)
        labels: ((N sample,) np.ndarray) 
        kfolds: (int) number of cross-validation folds 
        ascending: (bool) if True or False then sample ranks are in ascending 
            or descending order of samples scores, respectively.
        seed: Random number generator seed

    Returns:
        python generator of length kfolds, each iteration returns a 
            2-tuple of dicts:
            (
            training_dataset =
                dict(
                data=((M base classifiers, training_idxs) 
                            np.ndarray),
                        labels=((training_idxs) np.ndarray)
                ), 
            test_dataset = 
                dict(
                    data=((M base classifiers, test_idxs) 
                            np.ndarray),
                    labels=((test_idxs) np.ndarray)
                )
            )
    """
    N = data.shape[1]

    if N != labels.size:
        raise ValueError

    for train_idx, test_idx in sample_idx(N, kfolds, seed=seed):
        yield ({
                "data":stats.rank_transform(data[:, train_idx], 
                            ascending=ascending),
                "labels": labels[train_idx]
                },{
                "data":stats.rank_transform(data[:, test_idx], 
                            ascending=ascending),
                "labels":labels[test_idx]
                })


def stratified_kfold(data, labels,
        kfolds, ascending=False, seed=None):
    """Stratified K fold CV returns rank values.
    
    Args:
        data: ((M methods, N samples) np.ndarray) of data
        labels: ((N samples,) np.ndarray) of binary class labels
        kfolds: (int)
        ascending: (bool)
        seed: arguments of rng.default_rng
    """
    rng = np.random.default_rng(seed=seed)

    idx=np.arange(labels.size)

    label_vals= np.unique(labels)
    if len(label_vals) != 2:
        raise ValueError("Require binary class labels")

    # get the indexes of samples according to their label value
    idx_by_label = []
    for label_val in label_vals:
        idx_by_label.append(idx[labels == label_val])

    label_counts = [len(w) for w in idx_by_label]
    # train / test index generators for each class label
    split_by_label = [sample_idx(label_counts[0],kfolds,seed=rng),
                      sample_idx(label_counts[1], kfolds, 
                                 reverse_order=True,
                                 seed=rng)]
    
    # recall that each sample_idx returns a generator.
    # each evaluation of the generator produces tuple
    # (training data idxs, testing data idxs)

    for k_idxs_by_label in zip(*split_by_label):

        train_idx = np.array([], dtype=np.int)
        test_idx = np.array([], dtype=np.int)

        for i, label_idx in enumerate(k_idxs_by_label):
            train_idx = np.hstack([train_idx, 
                                idx_by_label[i][label_idx[0]]])
            test_idx = np.hstack([test_idx, 
                                idx_by_label[i][label_idx[1]]])

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        yield ({
                "data":stats.rank_transform(data[:, train_idx], 
                            ascending=ascending),
                "labels": labels[train_idx]
                },{
                "data":stats.rank_transform(data[:, test_idx], 
                            ascending=ascending),
                "labels":labels[test_idx]
                })


def sample_idx(N, kfolds,
               reverse_order=False,
               shuffle=True,
               seed=None):
    """Get the indexes of n fold cross validation.

    Args:
        N : The number of samples in the data, must be
            greater than or equal to 3 x kfolds.  (int)
        kfolds : The number of folds for cross validation, 
            must be greater than 1. (int)
        seed: any seed acceptable to np.random.default_rng(seed=seed),
            note that when a Generator instance is passes as a seed
            that specific Generator instance is returned
        
    Returns:
        python generator : length kfolds
            For each item k, the generator returns a tuple of 
            (train_indexes, test_indexes) for cross validation 
            fold k computed by get_fold_idx.
    """
    if kfolds <= 1:
        raise ValueError("kfolds must be greater than 1.")
    elif N <= 3*kfolds:
        raise ValueError(("Number of samples must be "
                          "greater than 3 times the number of "
                          "folds for cross validation."))
    
    idx = np.arange(N)

    if shuffle:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(idx)

    # size of j^th partition of data set
    di_floor = int(np.floor(N / kfolds))

    di_mod = N % kfolds
    if reverse_order:
        di_mod = kfolds - di_mod

    j_start = 0
    j_end = 0

    for k in range(kfolds):

        j_start = j_end
        j_end = j_start + di_floor

        if not reverse_order and k < di_mod:
            j_end += 1
        elif reverse_order and k >= di_mod:
            j_end += 1

        yield (np.hstack([idx[0:j_start], idx[j_end:]]),
                idx[j_start:j_end])
