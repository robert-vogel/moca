"""Umoca test

Umoca is hard to test because parameters estimates are the
results of complicated equations applied to data.  It would
be beneficial to figure out how to make an automated test
for the accuracy.

For now, I will focus on testing whether Umoca runs, and
produces results that are consistent with the properties
of the expected results.
"""
from unittest import TestCase, main

import numpy as np

from moca import classifiers as cls
from moca import simulate as sim


class TestUmoca(TestCase):
    m_classifiers = 8
    n_samples = 1000
    n_pos = 250
    rng = np.random.default_rng()
    auc = rng.uniform(low=0.6, high=0.99, size=m_classifiers)
    corr_matrix = sim.make_corr_matrix(m_classifiers,
                                       independent=True)

    def test_inheritance(self):
        cl = cls.Umoca()

        self.assertTrue(isinstance(cl, cls.MocaABC))

    def test_fit_value_properties(self):
        data, labels = sim.rank_scores(self.n_samples,
                                       self.n_pos,
                                       self.auc,
                                       self.corr_matrix,
                                       seed=self.rng)

        cl = cls.Umoca()
        cl.fit(data)

        self.assertEqual(cl.M, self.m_classifiers)

        self.assertTrue(cl.prevalence >= 0)
        self.assertTrue(cl.prevalence <= 1)

        for j, w in enumerate(cl.weights):
            if self.auc[j] < 0.5:
                self.assertTrue(w < 0)
                continue

            self.assertTrue(w > 0)





if __name__ == "__main__":
    main()
