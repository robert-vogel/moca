from unittest import TestCase, main

import numpy as np

from moca import classifiers as cls
from moca import simulate as sim


class TestMocaAbc(TestCase):
    m_classifiers = 8
    n_samples = 100
    n_pos = 25
    rng = np.random.default_rng()
    auc = rng.uniform(size=m_classifiers)
    corr_matrix = np.eye(m_classifiers)

    def test_init(self):
        m = cls.MocaABC()
        self.assertIsNone(m.prevalence)
        self.assertIsNone(m.weights)
        self.assertIsNone(m.M)

    def test_fit(self):
        m = cls.MocaABC()

        with self.assertRaises(NotImplementedError):
            m.fit()
        with self.assertRaises(NotImplementedError):
            m.fit(1, 2, 3, 4)
            
    def test_get_scores_filter(self):
        m = cls.MocaABC()
        data, _ = sim.rank_scores(self.n_samples,
                                  self.n_pos,
                                  self.auc,
                                  self.corr_matrix,
                                  seed=self.rng)
        d = data[0,:]
        with self.assertRaises(ValueError):
            m.get_scores(d)

        data[0,1] = self.n_samples + 1
        with self.assertRaises(ValueError):
            m.get_scores(data)

        m.M = data.shape[0]
        m.weights = self.auc

        with self.assertRaises(ValueError):
            m.get_scores(data[:-1, :])

    def test_get_inference(self):
        data, _ = sim.rank_scores(self.n_samples,
                                  self.n_pos,
                                  self.auc,
                                  self.corr_matrix,
                                  seed=self.rng)
        m = cls.MocaABC()
        m.M = data.shape[0]
        m.weights = self.auc
        m.prevalence = self.n_pos / self.n_samples

        l = m.get_inference(data)
        l_ = np.array([0,1])
        self.assertEqual(np.setdiff1d(l,l_).size, 0)
        self.assertEqual(np.setdiff1d(l_,l).size, 0)

    def test_get_inference_filter(self):
        data, _ = sim.rank_scores(self.n_samples,
                                  self.n_pos,
                                  self.auc,
                                  self.corr_matrix,
                                  seed=self.rng)
        m = cls.MocaABC()
        m.M = data.shape[0]
        m.weights = self.auc

        m.prevalence = 0
        with self.assertRaises(ValueError):
            m.get_inference(data)

        m.prevalence = 1
        with self.assertRaises(ValueError):
            m.get_inference(data)



if __name__ == "__main__":
    main()
