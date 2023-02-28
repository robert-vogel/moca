
from unittest import TestCase, main

from moca import classifiers as cls

import numpy as np

class TestIndividualClassifier(TestCase):
    def setUp(self):
        self.m_bc = 4
        self.n_samples = 8
        self.data = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                            [1,2,3,4,5,6,7,0],
                            [2,3,4,5,6,7,0,1],
                            [3,4,5,6,7,0,1,2]])
        self.data += 1
        self.labels = np.array([0,1,0,0,1,0,0,0])

    def test_init(self):
        # test that initialization works
        for bc_i in range(self.m_bc):
            cl = cls.BaseClassifier(bc_i)
            self.assertEqual(bc_i, cl.idx)

        # test floating point conversion
        cl = cls.BaseClassifier(3.0)
        self.assertEqual(3, cl.idx)

        with self.assertRaises(TypeError):
            cl = cls.BaseClassifier(3.2)

        with self.assertRaises(TypeError):
            cl = cls.BaseClassifier([1])

        with self.assertRaises(TypeError):
            cl = cls.BaseClassifier((1,))

    def test_fit(self):
        for bc_i in range(self.m_bc):
            cl = cls.BaseClassifier(bc_i)

            cl.fit(self.data, self.labels)

            self.assertEqual(cl.prevalence, 
                    np.sum(self.labels) / self.n_samples)
            self.assertEqual(cl.M, self.m_bc)

    def test_scores(self):
        mean_rank = (self.n_samples + 1) / 2

        for bc_i in range(self.m_bc):
            cl = cls.BaseClassifier(bc_i)

            cl.fit(self.data, self.labels)

            for k, s in enumerate(cl.get_scores(self.data)):
                self.assertEqual(s, mean_rank - self.data[bc_i, k])


if __name__ == "__main__":
    main()
