
from unittest import TestCase, main

import numpy as np

from moca import simulate


class TestSampleNetwork(TestCase):
    seed = 42

    def test_reproducibility(self):

        for n in (2, 5, 10, 20, 50):
            network = simulate._sample_network(n, seed=self.seed)

            self.assertTrue((
                network
                == simulate._sample_network(n,seed=self.seed)
                ).all())

    def test_diagonal(self):
        rng = np.random.default_rng(seed=self.seed)

        for n in (2, 5, 10, 20, 50):

            network = simulate._sample_network(n, seed=rng)

            for i in range(n):

                summation = 0

                for j in range(n):

                    if i == j:
                        self.assertTrue(network[i, i] < 0)
                        continue
                        
                    self.assertTrue(network[i,j] > 0)
                    summation += network[i, j]

                self.assertTrue(summation < np.abs(network[i,i]))


class TestCorrMatrix(TestCase):
    seed = 42

    def test_identity(self):

        for n in (3, 4, 8, 16, 32, 64):
            corr = simulate.make_corr_matrix(n, independent=True)

            self.assertTrue((corr == np.eye(n)).all())

    def test_reproducibility(self):

        for n in (3, 5, 10, 20, 50):
            corr = simulate.make_corr_matrix(n, seed=self.seed)

            self.assertTrue((
                corr
                == simulate.make_corr_matrix(n, seed=self.seed)
                ).all())


    def test_input_filters(self):
        with self.assertRaises(ValueError):
            simulate.make_corr_matrix(2)

        with self.assertRaises(ValueError):
            simulate.make_corr_matrix(-5)

        with self.assertRaises(TypeError):
            simulate.make_corr_matrix("goat")
    
    def test_diagonal(self):
        for n in (3, 5, 10, 20, 50):
            corr = simulate.make_corr_matrix(n)

            self.assertEqual(n, np.linalg.matrix_rank(corr))

            for i in range(n):
                self.assertEqual(corr[i, i], 1)

                for j in range(n):
                    self.assertTrue(corr[i, j] >= -1)
                    self.assertTrue(corr[i, j] <= 1)

                    self.assertAlmostEqual(corr[i,j],
                                           corr[j,i],
                                           places=8)


class TestGaussianScores(TestCase):
    n_samples = 100
    n_positive = 25
    auc = np.array([0.1, 0.6, 0.7])
    m = auc.size
    corr_matrix = np.eye(m)
    seed=42

    def test_filter(self):
        with self.assertRaises(ValueError):
            simulate.gaussian_scores(self.n_samples,
                                     self.n_samples + 1,
                                     self.auc,
                                     self.corr_matrix)

        with self.assertRaises(ValueError):
            simulate.gaussian_scores(-self.n_samples,
                                     self.n_positive,
                                     self.auc,
                                     self.corr_matrix)

        with self.assertRaises(ValueError):
            simulate.gaussian_scores(-self.n_samples,
                                     -self.n_positive,
                                     self.auc,
                                     self.corr_matrix)

        with self.assertRaises(ValueError):
            simulate.gaussian_scores(self.n_samples,
                                     -self.n_positive,
                                     self.auc,
                                     self.corr_matrix)

        with self.assertRaises(ValueError):
            simulate.gaussian_scores(self.n_samples,
                                     np.array([self.n_positive,
                                               self.n_positive]),
                                     self.auc,
                                     self.corr_matrix)

        with self.assertRaises(ValueError):
            auc = self.auc.copy()
            auc[0] = -0.3
            simulate.gaussian_scores(self.n_samples,
                                     self.n_positive,
                                     auc,
                                     self.corr_matrix)


        with self.assertRaises(ValueError):
            auc = self.auc.copy()
            auc[0] = 1.3
            simulate.gaussian_scores(self.n_samples,
                                     self.n_positive,
                                     auc,
                                     self.corr_matrix)

        with self.assertRaises(AttributeError):
            auc = self.auc.tolist()
            simulate.gaussian_scores(self.n_samples,
                                     self.n_positive,
                                     auc,
                                     self.corr_matrix)

        with self.assertRaises(ValueError):
            rng = np.random.default_rng()
            auc = rng.uniform(size=self.m+1)
            simulate.gaussian_scores(self.n_samples,
                                     self.n_positive,
                                     auc,
                                     self.corr_matrix)

    def test_reproducibility(self):
        scores, labels = simulate.gaussian_scores(self.n_samples,
                                              self.n_positive,
                                              self.auc,
                                              self.corr_matrix,
                                              seed=self.seed)

        s_, l_ = simulate.gaussian_scores(self.n_samples,
                                              self.n_positive,
                                              self.auc,
                                              self.corr_matrix,
                                              seed=self.seed)

        self.assertTrue((scores == s_).all())
        self.assertTrue((labels == l_).all())


    def test_sampling_differences(self):
        scores, labels = simulate.gaussian_scores(self.n_samples,
                                              self.n_positive,
                                              self.auc,
                                              self.corr_matrix,
                                              seed=self.seed)

        s_, l_ = simulate.gaussian_scores(self.n_samples,
                                              self.n_positive,
                                              self.auc,
                                              self.corr_matrix,
                                              seed=self.seed+1)

        self.assertFalse((scores == s_).all())
        self.assertFalse((labels == l_).all())

    def test_labels(self):
        scores, labels = simulate.gaussian_scores(self.n_samples,
                                              self.n_positive,
                                              self.auc,
                                              self.corr_matrix)

        l_ = np.array([0,1])
        self.assertEqual(np.setdiff1d(labels, l_).size, 0)
        self.assertEqual(np.setdiff1d(l_, labels).size, 0)
    

class TestRankScores(TestCase):
    n_samples = 100
    n_positive = 25
    auc = np.array([0.1, 0.6, 0.7])
    m = auc.size
    corr_matrix = np.eye(m)
    seed=42

    def test_reproducibility(self):
        ranks, labels = simulate.rank_scores(self.n_samples,
                                             self.n_positive,
                                             self.auc,
                                             self.corr_matrix,
                                             seed=self.seed)

        r_, l_ = simulate.rank_scores(self.n_samples,
                                      self.n_positive,
                                      self.auc,
                                      self.corr_matrix,
                                      seed=self.seed)

        self.assertTrue((ranks == r_).all())
        self.assertTrue((labels == l_).all())


    def test_ranks(self):
        ranks, labels = simulate.rank_scores(self.n_samples,
                                             self.n_positive,
                                             self.auc,
                                             self.corr_matrix,
                                             seed=self.seed)

        l_ = np.array([0,1])
        self.assertEqual(np.setdiff1d(labels, l_).size, 0)
        self.assertEqual(np.setdiff1d(l_, labels).size, 0)

        r_ = np.arange(1, self.n_samples+1)
        for j in range(self.m):
            self.assertEqual(np.setdiff1d(ranks,r_).size, 0)
            self.assertEqual(np.setdiff1d(r_, ranks).size, 0)




if __name__ == "__main__": 
    main()
