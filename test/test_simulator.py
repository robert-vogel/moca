
from unittest import TestCase, main

import numpy as np

from scipy.special import ndtr
from moca import simulate


def delta2auc(delta, *v):
    return ndtr(delta / np.sqrt(v[0] + v[1]))


class TestEnsembleGaussianPredictionsParseAuc(TestCase):
    def test_obvious_errors(self):
        with self.assertRaises(TypeError):
            simulate.EnsembleGaussianPredictions()

        with self.assertRaises(TypeError):
            simulate.EnsembleGaussianPredictions(
                    m_classifiers=10
                    )

        with self.assertRaises(ValueError):
            simulate.EnsembleGaussianPredictions(
                    auc = [-0.1, 0.2, 0.5],
                    )

        with self.assertRaises(ValueError):
            simulate.EnsembleGaussianPredictions(
                    auc = [0.1, 0.2, 1.5],
                    )

        with self.assertRaises(ValueError):
            simulate.EnsembleGaussianPredictions(
                    auc = 1.5,
                    m_classifiers = 10,
                    )

        with self.assertRaises(ValueError):
            simulate.EnsembleGaussianPredictions(
                    auc =-0.1, 
                    m_classifiers = 10,
                    )

        with self.assertRaises(ValueError):
            simulate.EnsembleGaussianPredictions(
                    auc =(-0.1, 0.8), 
                    m_classifiers = 10,
                    )

        with self.assertRaises(ValueError):
            simulate.EnsembleGaussianPredictions(
                    auc =(0.1, 1.8), 
                    m_classifiers = 10,
                    )

        with self.assertRaises(ValueError):
            simulate.EnsembleGaussianPredictions(
                    auc =(0.1, 1.8), 
                    m_classifiers = 2,
                    )

        with self.assertRaises(ValueError):
            simulate.EnsembleGaussianPredictions(
                    auc =(-0.1, 0.8), 
                    m_classifiers = 2,
                    )

        with self.assertRaises(ValueError):
            simulate.EnsembleGaussianPredictions(
                    auc =(0.9, 0.8), 
                    m_classifiers = 10,
                    )

        with self.assertRaises(ValueError):
            target_auc = [0.1, 0.2, 0.5]
            simulate.EnsembleGaussianPredictions(
                    auc = target_auc,
                    m_classifiers=10
                    )

    def validate_auc_input(self, true_auc, g):
        # assert correct number of classifiers
        self.assertEqual(len(true_auc), len(g))

        for i, (auc, delta) in enumerate(zip(true_auc, g._delta)):
            test_auc = delta2auc(delta, g._cov[i, i], g._cov[i, i])
            self.assertAlmostEqual(auc, test_auc)

    def test_auc_generation_from_range(self):
        # I want to generate 10 b.c.'s on the range [0.2, 0.5]
        target_auc = np.linspace(0.2, 0.5, 10)
        g = simulate.EnsembleGaussianPredictions(
                    auc = [0.2, 0.5],
                    m_classifiers=10
                    )
        self.validate_auc_input(target_auc, g)


        # I want these 15 auc values
        target_auc = np.linspace(0.4, 0.75, 15)
        g = simulate.EnsembleGaussianPredictions(auc=(0.4, 0.75), 
                            m_classifiers = 15)
        self.validate_auc_input(target_auc, g)

        # I want these 15 auc values when cov information is specified
        target_auc = np.linspace(0.4, 0.75, 15)
        g = simulate.EnsembleGaussianPredictions(auc=(0.4, 0.75),
                                        m_classifiers = 15,
                                        group_sizes=(4, 2, 4),
                                        group_corrs=(0.6, 0.7, 0))
        self.validate_auc_input(target_auc, g)

    def test_bc_specified_aucs(self):
        # the length of an array takes precedence over m_classifiers
        # in determining the length of an array
        target_auc = [0.1, 0.2, 0.5]
        g = simulate.EnsembleGaussianPredictions(
                    auc = target_auc,
                    )
        self.validate_auc_input(target_auc, g)

        # I want these 3 auc values when cov pars specified
        g = simulate.EnsembleGaussianPredictions(auc=(0.8, 0.63, 0.713),
                                    group_sizes=(2, 1),
                                    group_corrs=(0.6, 0))
        self.validate_auc_input((0.8, 0.63, 0.713), g)

    def test_auc_generation_from_float(self):
        # I want two base classifiers with auc 0.8
        m_classifiers = 2
        target_auc = 0.8
        g = simulate.EnsembleGaussianPredictions(
                auc = target_auc,
                m_classifiers = m_classifiers
                )
        self.validate_auc_input([target_auc for _ in range(m_classifiers)], g)

        # I want 38 base classifiers with AUC 0.6
        m_classifiers = 38
        target_auc = 0.6
        g = simulate.EnsembleGaussianPredictions(
                auc = target_auc,
                m_classifiers = m_classifiers
                )
        self.validate_auc_input([target_auc for _ in range(m_classifiers)], g)

        # docstring example
        g = simulate.EnsembleGaussianPredictions(auc = 0.7, m_classifiers = 10)

        self.validate_auc_input([0.7 for _ in range(10)], g)



class TestEnsembleGaussianPredictionsCov(TestCase):
    def validate_cov_vals(self, group_corrs, group_sizes, cov_test):
        m_classifiers = cov_test.shape[0]

        cov_true = self.mk_cov_matrix(group_corrs, 
                                group_sizes, 
                                m_classifiers)

        for i in range(m_classifiers):
            for j in range(m_classifiers):
                self.assertEqual(cov_true[i, j], cov_test[i, j])

    def mk_cov_matrix(self, group_corrs, group_sizes, m):
        if group_corrs is None:
            return np.eye(m)

        cov = np.zeros(shape=(m, m))

        # build index ranges
        idx_ranges = [(0, group_sizes[0])]
        for gs in group_sizes[1:]:
            idx_ranges.append((idx_ranges[-1][1], gs + idx_ranges[-1][1]))

        # build cov matrix
        for g, idx_range in enumerate(idx_ranges):
            for i in range(*idx_range):
                for j in range(*idx_range):
                    cov[i, j] = group_corrs[g]

        # set diagonal to 1
        for i in range(m):
            cov[i, i] = 1

        return cov

    def test_errors(self):
        with self.assertRaises(TypeError):
            simulate.EnsembleGaussianPredictions(auc=0.7, 
                                m_classifiers=10,
                                group_corrs = 0.7)

        with self.assertRaises(TypeError):
            simulate.EnsembleGaussianPredictions(auc=0.7, 
                                m_classifiers=10,
                                group_sizes = (0.7,))

        with self.assertRaises(TypeError):
            simulate.EnsembleGaussianPredictions(auc=0.7, 
                                m_classifiers=10,
                                group_sizes = (5,))

        with self.assertRaises(TypeError):
            simulate.EnsembleGaussianPredictions(auc=0.7, 
                                m_classifiers=10,
                                group_sizes = 5)

        with self.assertRaises(ValueError):
            simulate.EnsembleGaussianPredictions(auc=0.7, 
                                m_classifiers=10,
                                group_corrs = 5,
                                group_sizes = (3,))

        with self.assertRaises(TypeError):
            simulate.EnsembleGaussianPredictions(auc=0.7, 
                                m_classifiers=10,
                                group_corrs = 0.55,
                                group_sizes = (3,np.nan))

        with self.assertRaises(TypeError):
            simulate.EnsembleGaussianPredictions(auc=0.7, 
                                m_classifiers=10,
                                group_corrs = 0.5,
                                group_sizes = (3,None))


    def test_default(self):
        g = simulate.EnsembleGaussianPredictions(auc=0.7, m_classifiers=10)
        self.validate_cov_vals(None, None, g._cov)

    def test_float_corr_list_sizes(self):
        # I want all classifiers in distinct groups, but each group
        # to exhibit the same correlation between group members
        g = simulate.EnsembleGaussianPredictions(auc=(0.4, 0.75),
                                        m_classifiers = 15,
                                        group_sizes=(4, 2, 4),
                                        group_corrs=0.7)
        self.validate_cov_vals((0.7, 0.7, 0.7), (4, 2, 4), g._cov)

        # I want a subset of classifiers to be conditionally dependent, 
        # such that the correlation between group members are the same.
        # the remaining base classifiers are independent

        g = simulate.EnsembleGaussianPredictions(auc=(0.4, 0.75),
                                        m_classifiers = 15,
                                        group_sizes=(4, ),
                                        group_corrs=0.7)
        self.validate_cov_vals((0.7, 0), (4, 11), g._cov)

    def test_standard_par_inputs(self):
        g = simulate.EnsembleGaussianPredictions(auc=(0.4, 0.75),
                                        m_classifiers = 15,
                                        group_sizes=(4, 2, 4),
                                        group_corrs=(0.6, 0.7, 0))
        self.validate_cov_vals((0.6, 0.7, 0), (4, 2, 4), g._cov)

        g = simulate.EnsembleRankPredictions(auc=(0.8, 0.63, 0.713),
                                    group_sizes=(2, 1),
                                    group_corrs=(0.6, 0))

        self.validate_cov_vals((0.6, 0), (2, 1), g._cov)

        g = simulate.EnsembleRankPredictions(auc=(0.8, 0.63, 0.713),
                                    group_sizes=(2,),
                                    group_corrs=(0.6,))

        self.validate_cov_vals((0.6,), (2,), g._cov)


class TestEnsembleRankPredictions(TestCase):
    def test_sample_ranks(self):
        rsim = simulate.EnsembleRankPredictions(auc=(0.55, 0.74),
                                m_classifiers=10,
                                group_sizes=(4,),
                                group_corrs=(0.7,))

        n_samples = 100
        n1_samples = 30
        r, y = rsim.sample(n_samples, n1_samples)

        self.assertEqual(r.shape[0], 10)
        self.assertEqual(r.shape[1], n_samples)
        self.assertEqual(y.size, n_samples)

        rank_set = np.arange(1, n_samples+1)
        label_set = np.array([0,1])

        self.assertEqual(len(np.setdiff1d(label_set, y)), 0)
        self.assertEqual(len(np.setdiff1d(y, label_set)), 0)

        self.assertAlmostEqual(np.sum(y), n1_samples)

        for r_i in r:
            self.assertEqual(len(np.setdiff1d(rank_set, r_i)), 0)
            self.assertEqual(len(np.setdiff1d(r_i, rank_set)), 0)


if __name__ == "__main__": 
    main()
