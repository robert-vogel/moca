from unittest import TestCase, main

import numpy as np

from moca import classifiers as cls
from moca import simulate as sim


class TestWoc(TestCase):
    m_classifiers = 8
    n_samples = 100
    n_pos = 25
    rng = np.random.default_rng()
    auc = rng.uniform(size=m_classifiers)
    corr_matrix = np.eye(m_classifiers)

    def test_fit(self):
        data, labels = sim.rank_scores(self.n_samples,
                                       self.n_pos,
                                       self.auc,
                                       self.corr_matrix,
                                       seed=self.rng)
        cl = cls.Woc()

        self.assertIsNone(cl.M)
        self.assertIsNone(cl.weights)
        self.assertIsNone(cl.prevalence)

        cl.fit(data, labels)

        self.assertEqual(cl.M, self.m_classifiers)
        self.assertEqual(cl.prevalence, self.n_pos/self.n_samples)
        
        for _, w in enumerate(cl.weights):
            self.assertEqual(w, 1)
        

if __name__ == "__main__":
    main()
