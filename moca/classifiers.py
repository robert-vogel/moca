"""moca package classifiers

By: Robert Vogel

The classifiers of the moca packages are all subclasses
of the Moca abstract base class (MocaABC) which defines
the interface and default initialization of the implemented
classifiers.  The included classifiers are:
    * Smoca, implements the supervised moca algorithm,
    * Umoca, implements the unsupervised moca algorithm, and
    * Woc, implements the wisdom of crowds classifier.
Additional classes are define to support each of the
aforementioned classifiers.
"""
import warnings
import numpy as np

from . import stats
from . import cross_validate as cv

from summa.classifiers import Summa

class MocaABC:
    """Moca abstract base class."""
    def __init__(self):
        self.prevalence = None
        self.weights = None
        self.M = None

    @property
    def name(self):
        return self.__class__.__name__

    def fit(self, *_):
        raise NotImplementedError

    def get_scores(self, data):
        """Compute moca score for each sample.

        Args:
            data : ((M classifier, N sample) np.ndarray)
                rank predictions

        Returns:
            s : ((N sample,) np.ndarray) moca sample scores
        """
        # check that predictions by M methods are made. 
        # It is the user's responsiblity
        # that the order of methods is correct.
        if data.ndim != 2:
            raise ValueError(("Input data needs to be (M, N)"
                              " np.ndarray (ndim = 2), input data"
                              f" dim = {data.ndim}"))

        if not stats.is_rank(data):
            raise ValueError("Input data must be sample rank")

        if data.shape[0] != self.M:
            raise ValueError(("Input sample does not consist of"
                              f" predictions by {self.M}"
                              " methods"))

        s = np.zeros(data.shape[1])
        c = 0
        for j, w in enumerate(self.weights):
            s += w * data[j, :]
            c += w
        return stats.mean_rank(data.shape[1])*c - s

    def get_inference(self, data):
        """Estimate class labels from moca scores.
        
        The \hat{N1} samples with the highest moca scores are
        labeled class 1, where
        \hat{N1} = int(self.prevalence * N).

        Args:
            data : ((M classifiers, N samples) np.ndarray)
                rank predictions

        Returns:
            labels : ((N,) np.ndarray)
                Sample inferred class labels

        Raises:
            ValueError: if the predicted number of samples to 
                be of the positive class is either 0 or N. 
        """
        N = data.shape[1]
        labels = np.zeros(N)
        
        # Get the idx that would sort sample moca scores.
        # Note that negative is so that samples believed to be of
        # the positive class, high moca score, will be sorted to 
        # low values making subsequent indexing easier.
        idx = np.argsort(-self.get_scores(data))
        idx_thresh = np.int32(self.prevalence * N)
        
        if idx_thresh == 0:
            raise ValueError(("No samples predicted to be in"
                              " positive class"))
        elif idx_thresh >= N:
            raise ValueError(("All samples predicted to be in"
                              " positive class"))

        labels[idx[:idx_thresh]] = 1.
        return labels


class Woc(MocaABC):
    def fit(self, data, labels):
        self.M = data.shape[0]
        self.prevalence = np.mean(labels)
        self.weights = np.ones(data.shape[0])


class GreedySearchIdxManager:
    """Manage indexes for greedy search.
    
    Greedy search involves the sequential storage of found
    indexes.  As all indexes must be found only once, the
    set of candidate indexes is the complement set of the
    original list(range(m)) and the found list.  This class
    stores the found and complement sets as lists and 
    provides methods for updating, retrieving, and generating
    prospective sets.

    Args:
        m: (int)
            the number of entries for which an index is made
    """
    def __init__(self, m):
        if not isinstance(m, int):
            raise ValueError

        self.m = m

        self._found = list()
        self._complement = list(range(self.m))

    @property
    def found(self):
        return self._found.copy()

    @property
    def complement(self):
        return self._complement.copy()

    def update(self, i):
        # remember that remove, removes value from list by value
        # and not index
        self._complement.remove(i)
        self._found.append(i)

    def prospective(self, i):
        candidate_idx = self.found
        candidate_idx.append(i)
        return candidate_idx


class GreedySearchMocaStatsManager:
    """A manager of Moca stats and weights of subsets.
    
    The moca methodology requires the computation of
    class conditioned means and covariance matrices.
    When considering an ensemble method that consists of
    a subset of base classifiers, recalculation of the
    statistics is not required.  This manager provides
    a means to effectively subset statistics and compute
    the corresponding moca weights.
    
    Args:
        data: ((M, method, N sample) np.ndarray)
            sample ranks produced by each base classifier
        labels: ((N sample, ) np.ndarray)
            binary (0,1) sample class labels
    """
    def __init__(self, data, labels):
        self.delta = stats.delta(data, labels)
        self.cov_matrix = stats.moca_cov(data,labels)

    def _subset_cov(self, idx):
        if len(idx) == 0:
            raise ValueError

        c = np.zeros(shape = (len(idx), len(idx)))

        for i, idx_i in enumerate(idx):
            for j, idx_j in enumerate(idx):
                c[i, j] = self.cov_matrix[idx_i, idx_j]

        return c

    def delta_column_vector(self, idx=None):
        if idx is None:
            return self.delta.reshape(self.delta.size, 1)

        if len(idx) == 0:
            raise ValueError
        
        return self.delta[idx].reshape(len(idx), 1)

    def c_inv(self, idx):
        c = self._subset_cov(idx)

        # check if a unique solution to the inverse exists
        if np.linalg.matrix_rank(c) < len(idx):
            warnings.warn(("Class coniditioned covariance is"
                           " not full rank, using"
                           " pseudo-inverse"))

            return np.linalg.pinv(c)

        return np.linalg.inv(c)

    def sq_snr(self, idx):
        """Squared signal to noise ratio"""
        d = self.delta[idx]
        return d @ self.c_inv(idx) @ d

    def weights(self, idx):
        """Moca weights for a given subset of base classifiers.
        
        Args:
            idx: (i_1, i_2, ...,i_m) iterator object
                represents m base classifiers represented by
                their index in the correspoding
                data matrix, and consequently self.delta and
                self.cov_matrix.
        """
        if np.unique(idx).size != len(idx):
            raise ValueError

        weights = np.zeros(self.delta.size)

        subset_weights = self.c_inv(idx) @ self.delta[idx]

        for i, w in zip(idx, subset_weights):
            weights[i] = w

        return weights


class Smoca(MocaABC):
    """Supervised moca.
    
    Args:
        subset_select: ("greedy" or None)
            how to perform subset selection, default "greedy"
        subset_select_par: (None or int)
            if None determine optimal number of base classifiers
            by cross validation, otherwise use the specified
            number (default None).
    """
    _supported_subset_methods = ("greedy", None)

    def __init__(self, subset_select="greedy",
                 subset_select_par = None):
        super().__init__()

        if not subset_select in self._supported_subset_methods:
            raise ValueError(("Invalid subset_select"
                              "method string."))

        self.subset_select = subset_select
        self.subset_select_par = subset_select_par

    def _find_optimal_subset_number(self, data, labels, seed):
        """Find optimal number of base classifiers.

        The optimal number of base classifiers is the ensemble
        that maximizes the average ensemble signal-to-noise
        score by 10X cross validation
        """
        kfolds = 10
        nsamples = labels.size

        subset_sizes = list(range(1, data.shape[0] + 1))

        # subset SNR mean
        sub_snr_m = {w:0 for w in subset_sizes}

        # subset SNR standard error of mean
        sub_snr_s = {w:0 for w in subset_sizes}

        rng = np.random.default_rng(seed = seed)

        for m in subset_sizes:

            cv_generator = cv.stratified_kfold(data,
                                    labels,
                                    kfolds,
                                    ascending=True,
                                    seed=rng)

            for train, test in cv_generator: 

                cl = Smoca(subset_select = "greedy", 
                            subset_select_par = m)
                cl.fit(train["data"], train["labels"])

                s = cl.get_scores(test["data"])
                
                # note that scores switches convention, high score
                # correspondes to positive class sample, while
                # a low rank corresponds to a positive class
                # sample
                tmp_snr_stat = -stats.snr(s, test["labels"])
                sub_snr_m[m] += tmp_snr_stat
                sub_snr_s[m] += tmp_snr_stat**2

            # compute mean snr
            sub_snr_m[m] = sub_snr_m[m] / kfolds

            # compute sem: 1) computing unbiased variance,
            #               2) square root
            sub_snr_s[m] = ((sub_snr_s[m] - kfolds*sub_snr_m[m]**2)
                            / (kfolds-1))
            sub_snr_s[m] = np.sqrt(sub_snr_s[m] / kfolds)


        best_snr = 0 

        # note that search is from the smallest subset to largest,
        # therefore, if there is a tie for highest snr between two
        # subset sizes, the smallest sized subset will be selected
        # as optimal.

        for m in subset_sizes:
            if sub_snr_m[m] > best_snr:
                # self.subset_select_par = m
                opt_m_subset = m
                best_snr = sub_snr_m[m]

        # enforce parsimony as described in
        # Elements of Statistical Learning
        snr_threshold = best_snr - sub_snr_s[opt_m_subset]

        for m in subset_sizes:
            if sub_snr_m[m]+sub_snr_s[m] >= snr_threshold:
                break

        self.subset_select_par = m


    def _compute_greedy_subset_select(self, data, labels, seed):
        """Use a greedy search for subset selection."""

        if self.subset_select_par is None:

            self._find_optimal_subset_number(data, labels, seed)

        elif self.subset_select_par == data.shape[0]:
            
            return self._compute_default_weights(data, labels)

        elif (self.subset_select_par > data.shape[0]
              or self.subset_select_par < 1):

            raise IndexError

        stats_mgr = GreedySearchMocaStatsManager(data, labels)

        # Initialize by selecting b.c. with highest delta, which
        # in effect selects the method with the highest SNR.  To
        # see this recall that SNR is monotonic with AUC, and
        # that by Ahsen, Vogel, and Stolovitzky, JMLR 2019 that
        # AUC = delta / N + 1/2.

        best_idx, best_performance = None, 0

        for i, d in enumerate(stats_mgr.delta.squeeze()):
            if np.abs(d) > np.abs(best_performance):
                best_performance = d
                best_idx = i

        if self.subset_select_par == 1:
            weights = np.zeros(data.shape[0])
            weights[best_idx] = 1

            return weights

        idx_mgr = GreedySearchIdxManager(data.shape[0])

        idx_mgr.update(best_idx)

        for _ in range(1, self.subset_select_par):

            best_performance = 0
            best_idx = None

            for i in idx_mgr.complement:

                tmp_idx = idx_mgr.prospective(i)

                tmp_sq_snr = stats_mgr.sq_snr(tmp_idx)

                # ensemble snr = \sqrt{\Delta C^{-1} \Delta}, as
                # the \sqrt{x} monotonically increases with x,
                # the square root is not necessary for finding
                # the optimal snr

                if  tmp_sq_snr > best_performance:
                    best_performance = tmp_sq_snr
                    best_idx = i

            idx_mgr.update(best_idx)

        return stats_mgr.weights(idx_mgr.found)

    def _compute_default_weights(self, data, labels):
        m = data.shape[0]

        delta = stats.delta(data, labels).reshape(m, 1)
        c = stats.moca_cov(data, labels)

        if np.linalg.matrix_rank(c) < m:
            warnings.warn(("Class coniditioned covariance is"
                           " not full rank, consequently a"
                           " unique solution is not possible."
                           " Solve using pseudo-inverse."),
                           UserWarning)

            cinv = np.linalg.pinv(c)
        else:
            cinv = np.linalg.inv(c)

        return np.dot(cinv, delta).squeeze()

    @property
    def name(self):
        if self.subset_select is None:
            return self.__class__.__name__
        else:
            return (f"{self.__class__.__name__}"
                    f"-{self.subset_select}")

    def fit(self, data, labels, seed=None):
        self.M = data.shape[0]
        self.prevalence = np.mean(labels)

        if self.subset_select is None:

            self.weights = self._compute_default_weights(data,
                                                         labels)

        elif self.subset_select == "greedy":

            self.weights =self._compute_greedy_subset_select(data, 
                                labels, 
                                seed=seed)

        self.weights = (self.weights
                        / stats.l2_vector_norm(self.weights))


class Umoca(MocaABC):
    def __init__(self, prevalence=None, tol=1e-6, max_iter=5000):
        super().__init__()
        self.prevalence = prevalence
        self._tol = tol
        self._max_iter = max_iter

    @staticmethod
    def _infer_alpha(T,
                     cov_eig_val, cov_eig_vec,
                     tensor_singular_value):
        """Compute alpha by minimizing the sum squared errors.
        
        According to moca theory, the elements ijj (i\neq j) of
        the M x M x M third central moment tensor (T) of rank
        predictions by M conditionally independent binary
        classifiers is
        
        T_ijj = p (1-p) D_i d_j + p (1-p) (2p - 1) D_iD_j^2

        with:
            * p being the prevalence of the positive class
                samples, 
            * D_i the difference in the class conditioned
                average ranks 
                (D_i = E[R_i | Y=0] - E[R_i | Y=1])
                of the i^{th} base classifier [Ref. 1]
            * d_j is difference of the class conditioned
                variances,
                Var(R_j|Y=0) - Var(R_j|Y=1)
                of the j^{th} base classifier.

        From this formula, we are interested in inferring 

        \alpha_j = d_j / ||D||

        using the equation above and SUMMA [Ref. 1] inferred
        values
        
        tensor_singular_value = p (1-p) (2p-1) ||D||^3
        cov_eig_val = p (1-p) ||D||^2
        cov_eig_vec[i] = D_i / ||D||
        
        Consider the rearrangment of terms in the equation above
        for T_ijj

        T_ijj - p (1-p) (2p-1) D_iD_j^2 = p (1-p) D_i

        Remembering that for each j their are M-1 values of i
        that satisfy i \neq j.  We then call the LHS of the
        above equation the i^{th} value of M-1 length vector Y_j,

        Y_{ij} = T_{ijj} - 
                tensor_singular_value * cov_eig_vec[i]
                    * cov_eig_vec[j]**2

        and RHS the i^{th} value of M-1 length vector X

        X_i d_j/||D|| = cov_eig_val * cov_eig_vec[i] \alpha_j.
    
        From which it follows that 

        Y_j = X \alpha_j

        We infer \alpha_j for each j in {1,2,...,M} by minimizing
        the sum squared residuals

        \hat{\alpha_j} = argmin 
                \sum_{i\neq j} (Y_{ij} - X_i \alpha_j)^2

        resulting in

        \hat{\alpha_j} = \frac{
                \sum_{i \neq j} Y_{ij} X_i) 
            }{
                \sum_{i \neq j} X_i^2
            }
        
        Args:
            T: ((M, M, M) np.ndarray) third central moment tensor
            cov_eig_val: (float)
            cov_eig_vec: ((M,) np.ndarray)
            tensor_singular_value: (float)

        Returns:
            alpha: ((M,) np.ndarray)

        References:
            [1] Ahsen, Vogel, and Stolovitzky J. Mach.
                Learn. Res. 2019
        """
        M = cov_eig_vec.size

        idx = np.arange(M)

        alpha = np.zeros(M)

        for j in range(M):
            # get indexes not equal to j
            i_neq_j = idx != j

            # compute transformed x and y
            x = cov_eig_val * cov_eig_vec[i_neq_j]
            y = (T[i_neq_j, j, j] - 
                    tensor_singular_value *
                    cov_eig_vec[i_neq_j] * 
                    cov_eig_vec[j]**2)

            # infer alpha_j by minimizing the sum squared residuals
            alpha[j] = np.sum(x*y) / np.sum(x**2)

        return alpha

    @staticmethod
    def _infer_sum_class_conditional_variance(cov_eig_value,
                                            cov_eig_vector, 
                                            tensor_singular_value,
                                            alpha,
                                            n_samples):
        """Infer the sum of class conditional variances.
        
        Let the sum of class conditioned variances of rank
        predictions (R) by method j be denoted as C_jj.  Under
        an assumption of conditional independence

        C_jj = 2 Var(R) + (2p-1) \delta_j - 2p(1-p) \Delta_j^2

        where

        \Delta_j = E(R_j | Y=0) - E(R_j | Y=1),
        \delta_j = Var(R_j | Y=0) - Var(R_j | Y=1)

        Given the following variables and their definitions

           l_t = third central moment inferred tensor
                singular value
           l_c = covariance inferred eigenvalue
           v_i = i^th element of covariance inferred eigenvector
           s_j = (Var(R-E[R] | Y = 0) - Var(R-E[R] | Y=1)) / ||D||
           || \Delta || = Norm of the Delta vector

        the authors showed that, C_jj may be inferred without
        labeled data by
        
        C_jj = 2 Var(R) + l_t \alpha_j / l_c - 2 l_c v_j^2

        where 

        \alpha_j := \delta_j / || \Delta ||

        This method applies these function to infer C_jj

        Args:
            cov_eig_value: (float)
            cov_eig_vector: ((M,) np.ndarray)
            tensor_singular_value: (float)
            alpha: ((M,) np.ndarray)
            n_samples: (int) number of samples

        Returns:
            sum_cond_vars : ((M,) np.ndarray) the sum of class
                conditioned variances of each of the M base
                classifiers
        """
        rank_variance = stats.variance_rank(n_samples)

        # allocate array for conditional variance values
        sum_cond_vars = np.zeros(cov_eig_vector.size)

        # compute the sum of conditional variances by the
        # equation in the docstring
        j = 0
        for v, a in zip(cov_eig_vector, alpha):
            sum_cond_vars[j] = (2*rank_variance
                                + tensor_singular_value
                                * a / cov_eig_value
                                - 2*cov_eig_value
                                * v**2)
            j += 1

        return sum_cond_vars 

    def fit(self, data, *_):
        """Infer Umoca weights and store as class attribute.

        Args:
            data: ((M base classifier, N sample) sample rank
                predictions

        Returns:
            None
        """
        self.M = data.shape[0]

        scl = Summa()
        scl.fit(data, tol=self._tol, max_iter=self._max_iter)

        alpha = self._infer_alpha(stats.third_central_moment(data),
                                  scl._eig_val,
                                  scl._eig_vec,
                                  scl._tensor_sv)

        cond_vars = self._infer_sum_class_conditional_variance(
                scl._eig_val,
                scl._eig_vec,
                scl._tensor_sv,
                alpha,
                data.shape[1])

        self.prevalence = scl.prevalence
        self.weights = scl.delta / cond_vars

        self.weights = (self.weights
                        / stats.l2_vector_norm(self.weights))


class BestBC(Smoca):
    def __init__(self, seed=None):
        super().__init__(subset_select="greedy",
                         subset_select_par=1)


class BaseClassifier(MocaABC):
    """Helper class for interface consistency of ind. methods.

    Moca ensemble classifiers use an 
    (M base classifiers, N sample) data set 
    as input for fitting and computing scores.  This class
    provides the means to analyze invidiual base classifiers
    within the Moca framework.  This is achieved by storing
    the classifier-of-interest's index in the data set, and
    using this data at said index for subsequent calculations.

    Args:
        cl_index: (int)
            index of classifier that the class should use
    """
    def __init__(self, cl_index):

        super().__init__()

        if (isinstance(cl_index, float)
            and cl_index % int(cl_index) == 0):

            cl_index = int(cl_index)

        elif not isinstance(cl_index, int):
            raise TypeError("Input index must be an integer")

        self.idx = cl_index

    @property
    def name(self):
        return f"{self.__class__.__name__}-{self.idx}"

    def fit(self, data, labels):
        self.prevalence = np.mean(labels)
        self.M = data.shape[0]

    def get_scores(self, data):
        """Transform sample rank values to moca scores.

        Args:
            data: ((M, N) ndarray)
                M base classifier by N sample array of rank
                predictions 
                
        Returns:
            s: ((N,) np.ndarray)
                moca score for each of the N samples
        """
        if data.ndim != 2:
            raise ValueError(("Input data needs to be"
                              " (M, N) ndarray (ndim = 2),"
                              f" input data dim = {data.ndim}"))

        if data.shape[0] != self.M:
            raise ValueError(("Input sample does not consist of"
                            f" predictions by {self.M} methods"))

        if not stats.is_rank(data[self.idx, :]):
            raise ValueError("Data are not rank values")

        return stats.mean_rank(data.shape[1]) - data[self.idx, :]

