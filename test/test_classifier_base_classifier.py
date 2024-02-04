from unittest import TestCase, main

import numpy as np

from moca import simulate as sim
from moca import classifiers as cls


class TestBaseClassifier(TestCase):
    m_classifiers = 8
    n_samples = 1000
    n_pos = 250
    rng = np.random.default_rng()
    auc = rng.uniform(low=0.6, high=0.99, size=m_classifiers)
    corr_matrix = sim.make_corr_matrix(m_classifiers,
                                       independent=True)
    def test_inheritance(self):
        cl = cls.BaseClassifier(3)
        self.assertTrue(isinstance(cl, cls.MocaABC))

    def test_init_filter(self):

        with self.assertRaises(TypeError):
            cls.BaseClassifier("the")

        with self.assertRaises(TypeError):
            cls.BaseClassifier([0,3])

        with self.assertRaises(TypeError):
            cls.BaseClassifier(np.array([0]))

        with self.assertRaises(TypeError):
            cls.BaseClassifier(2.1)


    def test_init_correct(self):
        cl = cls.BaseClassifier(3)

        self.assertEqual(cl.idx, 3)

    def test_fit(self):
        cl = cls.BaseClassifier(3)

        data, labels = sim.rank_scores(self.n_samples,
                                       self.n_pos,
                                       self.auc,
                                       self.corr_matrix,
                                       seed=self.rng)
        cl.fit(data, labels)

        self.assertEqual(cl.prevalence, self.n_pos/self.n_samples)
        self.assertEqual(cl.M, self.m_classifiers)

    def test_get_scores(self):
        cl_idx = 3
        cl = cls.BaseClassifier(cl_idx)

        data, labels = sim.rank_scores(self.n_samples,
                                       self.n_pos,
                                       self.auc,
                                       self.corr_matrix,
                                       seed=self.rng)
        cl.fit(data, labels)
        
        c = np.corrcoef(data[cl_idx,:],
                        cl.get_scores(data))

        self.assertAlmostEqual(c[0,1], -1, 8)

    def test_get_score_filters(self):
        cl_idx = 3
        cl = cls.BaseClassifier(cl_idx)

        data, labels = sim.rank_scores(self.n_samples,
                                       self.n_pos,
                                       self.auc,
                                       self.corr_matrix,
                                       seed=self.rng)

        cl.fit(data, labels)

        with self.assertRaises(ValueError):
            cl.get_scores(data[0, :])

        with self.assertRaises(ValueError):
            cl.get_scores(data[:self.m_classifiers-1,:])

        with self.assertRaises(ValueError):
            data[cl_idx, 3] = self.n_samples + 1
            cl.get_scores(data)

if __name__ == "__main__":
    main()
