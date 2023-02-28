"""Simulation tool for running experiments of the moca classifiers

By: Robert Vogel
Date: 2022-05-22

"""
import numbers

import numpy as np
from scipy.special import ndtri

from moca import stats


class EnsembleGaussianPredictions:
    """Model block diagonally correlated base classifier predictions.

    Fraction of classifiers assigned to each correlated group
    correlation per grup
    auc per classifier

    Assume that the number of classifiers is >= 2

    Args: 
        m_classifiers: None or int representing number of classifiers
        auc: float or ((auc_min, auc_max) array) or 
            ((auc_bc_1, auc_bc_2, ..., auc_bc_m_classifiers), array).  Each
            auc value is the area under receiver operating characteristic and
            consequently must be a value on the interval [0,1]
        group_sizes: ((G groups,) tuple, list, or np.ndarray) 
        group_corrs: (float or (G Pearson corr coefs,) tuple, list, or np.ndarray)

    Example:

        Generate 10 conditionally indepedent base classifiers with AUC of 0.7

        >>> EnsembleGaussianPredictions(auc = 0.7, m_classifiers = 10)

        Generate 15 conditionally independent base classifiers uniformally
        distributed on the interval [0.4, 0.75]

        >>> EnsembleGaussianPredictions(auc=(0.4, 0.75), m_classifiers = 15)

        Generate 10 base classifiers with AUC on the range [0.4, 0.75], such 
        that base classifiers 1-4, 5-6, and 7-10 exhibit class conditional 
        correlation of 0.6, 0.7, and 0, respectively.

        >>> EnsembleGaussianPredictions(auc=(0.4, 0.75),
                                        m_classifiers = 15,
                                        group_sizes=(4, 2, 4),
                                        group_corrs=(0.6, 0.7, 0))

        Generate 4 base classifiers with AUC [0.8, 0.63, 0.713], such 
        that base classifiers 1-2 and 3 exhibit class conditional 
        correlation of 0.6 and 0, respectively.

        >>> EnsembleRankPredictions(auc=(0.8, 0.63, 0.713),
                                    group_sizes=(2, 1),
                                    group_corrs=(0.6, 0))

        or equivalently,

        >>> EnsembleRankPredictions(auc=(0.8, 0.63, 0.713),
                                    group_sizes=(2,),
                                    group_corrs=(0.6,))
    """
    def __init__(self,
            m_classifiers=None,
            auc=None,
            group_corrs=None,
            group_sizes=None, 
            seed=None):

        self.m_classifiers, self.auc = self._parse_auc(auc, m_classifiers)

        self._cov = self._mk_covariance_matrix(group_corrs, group_sizes)
        self._delta = self._mk_delta_array(self.auc)

        self.rng = np.random.default_rng(seed=seed)

    def __len__(self):
        return self.m_classifiers

    def _parse_auc(self, auc, m_classifiers):
        # TypeError if m_classifiers is not int
        if isinstance(auc, float):
            auc = [auc for _ in range(m_classifiers)]

        # if given auc array represents each base classifier in the ensemble
        if m_classifiers is None:  
            m_classifiers = len(auc)

        # Verify AUC value is within AUC range
        for auc_i in auc:
            if auc_i < 0 or auc_i > 1: raise ValueError

        # if given m_classifiers was neither None or int, then error
        # thrown here. 
        if m_classifiers == len(auc): 
            return m_classifiers, auc
        elif len(auc) == 2 and auc[0] < auc[1]:
            return m_classifiers, np.linspace(auc[0], auc[1], m_classifiers)

        raise ValueError

    def _mk_delta_array(self, auc):
        """Compute delta array from auc values.
        
        Delta for each base classifer i is computed by

        \Delta_i = \sqrt{cov_i|positive_class + cov_i|negative_class} *
                        inv_standard_normal_cumulative(auc_i)

        as described in Marzben, C. "The ROC Curve and the Area under It
        as Performance Measures", Weather and Forecasting 2004.
        """
        delta = np.zeros(len(self))

        for i, auc_val in enumerate(auc):
            delta[i] = np.sqrt(2 * self._cov[i,i])  * ndtri(auc_val)

        return delta


    def _mk_covariance_matrix(self, group_corrs, group_sizes):
        """Make conditional covariance matrix.
        
        Parse input arguments and construct covariance matrix.  I have
        limited the covariance matrix to be the Pearson correlation matrix.
        Therefore, the diagonal elements will always be one, the matrix
        symmetric, and off-diagonal elements on the interval [-1, 1].
        """

        # either group_corrs and group_sizes are specified or they
        # are both not specified.  
        if group_corrs is None and group_sizes is None:
            return np.eye(len(self))
        elif group_corrs is None and group_sizes is not None:
            raise TypeError
        elif group_corrs is not None and group_sizes is None:
            raise TypeError
        elif np.sum(group_sizes) > self.m_classifiers:
            raise ValueError
        elif (not isinstance(group_corrs, numbers.Number) and 
                len(group_corrs) != len(group_sizes)):
            raise ValueError

        if isinstance(group_corrs, numbers.Number):  # if each group has same corr
            group_corrs = [group_corrs for g in group_sizes]

        cov = np.zeros(shape=(len(self), len(self)))

        g_start = 0
        for m, corr in zip(group_sizes, group_corrs):

            if corr > 1 or corr < -1:
                raise ValueError
            elif not isinstance(m, numbers.Number):
                raise TypeError

            g_final = g_start + m


            # loop over cov elements of group and set
            # the covariance values as specified
            for i in range(g_start, g_final):
                for j in range(g_start, g_final):

                    cov[i, j] = 1 if i == j else corr

            g_start = g_final

        # any remaining base classifiers are assumed 
        # conditionally independent,
        for i in range(g_final, len(self)):
            cov[i, i] = 1
        
        return cov

    @property
    def size(self):
        return self.m_classifiers

    def sample(self, label):
        """A single sample base classifier predictions."""
        s = self.rng.multivariate_normal(np.zeros(self._cov.shape[0]),
                        self._cov, size=1)

        if label ==1:
            s += self._delta

        return s


class EnsembleRankPredictions(EnsembleGaussianPredictions):
    def sample(self, n_samples, n_positive_class):
        scores = np.zeros(shape=(self.m_classifiers, n_samples))
        labels = np.zeros(n_samples)
        labels[:n_positive_class] = 1

        for j in range(n_samples):
            scores[:, j] = super().sample(labels[j])

        return stats.rank_transform(scores), labels

