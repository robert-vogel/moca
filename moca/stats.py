"""Statistics for the MOCA package

By: Robert Vogel
Current Affiliation: Scripps Research

TODO:
    * unit tests
        - moca_cov
        - delta
        - snr
        - roc
        - is_rank
    * confirm unit tests for:
        - rank_transform
"""
import warnings
import numbers
import numpy as np
from scipy.stats import rankdata

def mean_rank(N):
    """Compute the mean of N sample ranks on interval [1,N]"""

    if not isinstance(N, numbers.Number):
        raise ValueError
    if N < 2:
        raise ValueError("Mean requires at least two values, N >= 2")

    return (N + 1) / 2

def variance_rank(N):
    """Compute the variance of N sample ranks on interval [1, N]"""

    if not isinstance(N, numbers.Number):
        raise ValueError

    if N < 2:
        raise ValueError("Variance requires at least two values, N >= 2")

    return (N**2-1)/12


def third_central_moment(data):
    """Compute the third central moment

    Calculates the unbiased third central moment tensor 

    T_{ijk} = \frac{n}{(n-1)(n-2)}
                    \sum_{i=1}^n  (X_i - \mu_i) (X_j - \mu_j) (X_k - \mu_k)
 
    as published in Ref. 1.

    Args:
        data: ((M features, N samples) np.ndarray) 

    Returns:
        ((M,M,M) np.ndarray) 

    Refs:
        [1] https://mathworld.wolfram.com/SampleCentralMoment.html
    """
    
    if data.ndim == 1:
        n = data.size

        norm_constant = n / ((n-1) * (n-2))

        return np.sum(data - np.mean(data))**3 * norm_constant

    elif data.ndim != 2:
        raise ValueError("Only accepts np.ndarray with ndim == 1 or 2.")

    m, n = data.shape

    if m > n:
        warnings.warn("The number of dimensions exceeds number of samples.",
                        UserWarning)

    norm_constant = n / ((n-1) * (n-2))

    tmp = data - np.tile(np.mean(data, axis=1).reshape(m,1), (1, n))

    T = np.zeros(shape=(m,m,m))

    for k in range(m):
        for i in range(k, m):
            for j in range(i, m):

                t = np.sum(tmp[i, :]*tmp[j, :]*tmp[k, :]) * norm_constant

                # set all permutations accordingly
                T[i,j,k] = T[k,i,j] = T[j,k,i] = t
                T[j,i,k] = T[k,j,i] = T[i,k,j] = t

    return T 

def l2_vector_norm(x):
    """L2 vector norm, Euclidean norm.
    
    Args:
        x: (n,) np.ndarray, list, or tuple

    Returns:
        float
    """
    if isinstance(x, np.ndarray):
        return np.sqrt(np.sum(x * x))
    elif isinstance(x, list) or isinstance(x, tuple):
        s = 0
        for w in x:
            s += w**2
        return np.sqrt(s)
    else:
        raise ValueError

def delta(data, labels):
    """Difference of conditional means."""

    s = np.sum(labels)
    if s == 0 or s == labels.size:
        raise ValueError("Samples from only one class are observed.")

    if data.ndim == 1:
        return (np.mean(data[labels == 0]) - 
                np.mean(data[labels == 1]))
    elif data.ndim == 2:
        return (np.mean(data[:, labels == 0], axis=1) -
                np.mean(data[:, labels == 1], axis=1))
    else:
        raise ValueError

def moca_cov(data, labels):
    """Sum of class conditioned variances or cov matrices."""

    s = np.sum(labels)
    if s == 0 or s == labels.size:
        raise ValueError("Samples from only one class are observed.")

    if data.ndim == 1:
        return (np.var(data[labels == 0]) + 
                np.var(data[labels == 1]))
    elif data.ndim == 2:
        return (np.cov(data[:, labels == 0]) + 
                np.cov(data[:, labels == 1]))
    else:
        raise ValueError

def snr(data, labels):
    """Signal-to-noise ratio"""

    if data.ndim != 1:
        raise ValueError

    return delta(data, labels) / np.sqrt(moca_cov(data, labels))

def roc(scores, labels):
    """Receiver operating characteristic.
    
    Calculate the true (tpr) and false (fpr) positive rates, and
    the Area Under the receiver operating characteristic Curve, by
    the trapezoidal integration rule.

    Args:
        scores : ((N, ) np.ndarray) of scores, high correspond to
            postive class samples
        labels : ((N, ) np.ndarray) of labels where each element is in
            the set {0, 1} with 1 and 0 corresponding to positive and 
            negative class samples.
    Return:
        tuple ((N+1,) np.ndarray, (N+1, ) np.ndarray, float) of the 
            false positive rates (fpr), true postive rates (tpr), and auc.
    """

    s = np.sum(labels)
    if s == 0 or s == labels.size:
        raise ValueError("Samples from only one class are observed.")


    N = labels.size
    N1 = np.sum(labels)
    dT = 1 / N1
    dN = 1 / (N - N1)

    fpr = np.ones(N+1)
    tpr = np.ones(N+1)
    auc = 0
    tpr_integration = [1,1]

    sort_idx = np.argsort(scores)

    for i, idx in enumerate(sort_idx, start=1):
        
        j = N - i 

        if labels[idx] == 1:
            tpr[j] = tpr[j + 1] - dT
            fpr[j] = fpr[j + 1]
        elif labels[idx] == 0:
            tpr[j] = tpr[j+1]
            fpr[j] = fpr[j+1] - dN

            tpr_integration[1] = tpr_integration[0]
            tpr_integration[0] = tpr[j]

            auc += 0.5 * (tpr_integration[0] + tpr_integration[1]) * dN
        else:
            raise ValueError

    return fpr, tpr, auc
       

def rank_transform(data, ascending=False):
    '''Transform data to rank values in descending order.

    Here, ranks are in by default in descending order.  That is the sample 
    with the highest value has the lowest rank.

    Missing values stored as np.nan are assigned the median value over
    all samples that are not np.nan.  
    
    Ties are resolved by assigning rank according to 
    scipy.stats.rankdata keyword argument method=ordinal.

    Args:
        data : 1) 1-d array of sample scores ((N,) ndarray), or
               2) 2-d array of sample scores, independently compute rank 
                   values for each row ((M methods, N samples) ndarray)
        ascending: (bool) if False then samples are ranked in descending
            order, otherwise in ascending order

    Returns:
        Either 1) (N,) numpy array of sample ranks
               2) (M, N) numpy array of sample ranks
    '''
    if data.ndim == 1:
        return _compute_ranks(data, ascending)
    elif data.ndim == 2:
        rdata = np.zeros(shape=data.shape)
        for j in range(data.shape[0]):
            rdata[j, :] = _compute_ranks(data[j, :], ascending)
        return rdata
    else:
        raise ValueError("Input data must be either a 1-d or 2-d ndarray")

def _compute_ranks(data, ascending):
    '''Compute rank from data scores.
    
    Given a numpy array, convert data scores to sample rank.  Missing
    data are assigned the median (over available data) value.  Ties
    are handeled ordinally.
            
    Args:
        data: (N,) numpy array of sample scores
        ascending: (bool) whether sample ranks are in ascending
             order.  By default this is False and samples are
             ranked in descending order

    Returns:
        (N, ) numpy array of sample ranks
    '''
    # don't replace missing data of input data
    if any(np.isnan(data)):
        data = data.copy()
        nan_idxs = np.isnan(data)
        data[nan_idxs] = np.median(data[~nan_idxs])
        
    rdata = rankdata(data, method='ordinal')
    
    if ascending is False:
        rdata = rdata.size + 1 - rdata

    return rdata

def is_rank(X):
    """Test whether data are sample rank.

    Args:
        X : 1-d or 2-d ndarray of rank values.  If 2-d
            array, then we assume that the M samples are independently
            ranked across rows.  ((N, ) or (M, N) ndarray)
            
    Returns:
        bool
    """
    if X.ndim == 1:
        r_true = np.arange(1, X.size+1)
        return np.setdiff1d(r_true, X).size == 0
    elif X.ndim == 2:
        truth_val = True
        r_true = np.arange(1, X.shape[1]+1)

        for i in range(X.shape[0]):
            if np.setdiff1d(r_true, X[i, :]).size != 0:
                truth_val = False
                break
        return truth_val
    else:
        raise ValueError("Data must be a 1-d or 2-d ndarray.")


def is_auc(auc):
    """Is the float or iterable auc value(s).

    Returns:
        (bool)
    """
    if isinstance(auc, numbers.Number):
        return auc >= 0 and auc <= 1

    valid_auc = False

    for val in auc:
        if isinstance(val, numbers.Number):
            valid_auc = (val >= 0 and val <= 1)

        if not valid_auc:
            return False

    return True


def is_prevalence(prevalence):
    """Check prevalence value.

    Returns:
        (bool)
    """
    if isinstance(prevalence, numbers.Number):
        return prevalence < 1 and prevalence > 0

    return False
