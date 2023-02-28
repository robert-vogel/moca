
from unittest import TestCase, main

import numpy as np

# from pySUMMA.simulate import Rank
from moca import Smoca
from moca import moca


class TestSmoca(TestCase):
    def setUp(self):
        self.Nsamples = 1000
        self.Npositive = 300
        self.Mmodels = 10

        sim = Rank(self.Mmodels,
                    self.Nsamples,
                    self.Npositive)
        sim.sim()

        self.data, self.labels = sim.data, sim.labels

    def test_default_init(self):
        smoca = Smoca()

        self.assertIsNone(smoca.subset_select)
        self.assertIsNone(smoca.subset_select_par)
        self.assertEqual(smoca._compute_weights, smoca._compute_default_weights)

        self.assertEqual(smoca.name, "Smoca")
        self.assertIsNone(smoca.prevalence)
        self.assertIsNone(smoca.weights)

    def test_subset_init(self):
        smoca = Smoca(subset_select = "greedy")
        
        self.assertEqual(smoca.subset_select, "greedy")
        self.assertIsNone(smoca.subset_select_par)

        self.assertEqual(smoca._compute_weights, smoca._compute_greedy_subset_select)

    def test_wrong_fit_method(self):
        with self.assertRaises(ValueError):
            Smoca(subset_select = "l1")

    def test_no_subset_select(self):
        pass

    def test_compute_greedy_subset_selection(self):
        pass

    def test_find_optimal_subset(self):
        pass

    def test_fit(self):
        # TODO need to make this a real test

        sim = Rank(12, 1000, 300)
        sim.sim()

        cl = Smoca(subset_select="greedy", subset_select_par=5)
        cl.fit(sim.data, sim.labels)

        cl = Smoca(subset_select="greedy")
        cl.fit(sim.data, sim.labels)


class TestGreedySearchIdxManager(TestCase):
    def test_init(self):
        m = 10
        gs_idx = moca.GreedySearchIdxManager(m)

        self.assertEqual(gs_idx.m, m)
        self.assertEqual(gs_idx.found, [])
        self.assertEqual(gs_idx.complement, list(range(m)))

    def test_wrong_init_input(self):
        m = 10

        with self.assertRaises(ValueError):
            moca.GreedySearchIdxManager(float(m))

        with self.assertRaises(ValueError):
            moca.GreedySearchIdxManager(str(m))

        with self.assertRaises(ValueError):
            moca.GreedySearchIdxManager((m, ))

        with self.assertRaises(ValueError):
            moca.GreedySearchIdxManager([m])
 
        with self.assertRaises(ValueError):
            moca.GreedySearchIdxManager(np.array([m]))

        with self.assertRaises(ValueError):
            moca.GreedySearchIdxManager(set((m,)))

    def test_ens_properties(self):
        m = 10
        n = 4

        gs_idx = moca.GreedySearchIdxManager(m)

        rng = np.random.default_rng()
        ens_idx = list()
        for i in rng.choice(range(m), replace=False, size=n):

            # object creation by property
            self.assertNotEqual(id(gs_idx.found), id(gs_idx._found))
            self.assertNotEqual(id(gs_idx.complement), id(gs_idx._complement))

            # test created object values
            self.assertEqual(gs_idx.found, gs_idx._found)
            self.assertEqual(gs_idx.complement, gs_idx._complement)


            # test returned prosepective ensemble indexes
            ens_idx.append(i)
            self.assertEqual(ens_idx, gs_idx.prospective(i))
            gs_idx.update(i)

    def test_update(self):
        m = 10
        gs_idx = moca.GreedySearchIdxManager(m)

        for i in range(m):

            # test that a found is never in complement
            # after the i^th update call
            for j in gs_idx.complement:
                self.assertTrue(j not in gs_idx.found)

            # test that a complement idx is never in found
            # after the i^th update call
            for j in gs_idx.found:
                self.assertTrue(j not in gs_idx.complement)

            gs_idx.update(i)


class TestGreedyStatsManager(TestCase):
    def setUp(self):
        self.m, self.n, self.n1 = 10, 1000, 300
        self.auc = [1 for _ in range(self.m)]
        sim = Rank(self.m, self.n, self.n1, auc=self.auc)
        sim.sim()
        self.data, self.labels = sim.data, sim.labels

    def test_init(self):
        mgr = moca.GreedySearchMocaStatsManager(self.data, self.labels)

        # test delta
        for delta in mgr.delta:
            self.assertEqual(delta, self.n/2)

        # cov matrix tests
        # 1) symmetry
        self.assertTrue((mgr.cov_matrix == mgr.cov_matrix.T).all())
        
        # 2) ensembles of perfect classifiers should have equal
        # diagonal elements

        for i in range(self.m):
            for j in range(self.m):
                self.assertEqual(mgr.cov_matrix[i,i], mgr.cov_matrix[j,j])

        # 3) positive definite
        l, v = np.linalg.eigh(mgr.cov_matrix)

        self.assertEqual(np.sum(np.abs(l)), np.sum(l))
        self.assertTrue(np.prod(l) > 0)

    def test_delta_column(self):
        mgr = moca.GreedySearchMocaStatsManager(self.data, self.labels)

        # check indeed a column vector with expected entries
        # when no subset specified
        delta_column = mgr.delta_column_vector()
        self.assertEqual(delta_column.shape, (self.m, 1))

        for i, d in enumerate(mgr.delta):
            self.assertEqual(delta_column[i, 0], d)


        # test for random idxs of length int(self.m * 0.7)
        m_subsets = [int(self.m * w) for w in (0.3, 0.7, 1)]

        rng = np.random.default_rng()

        for m_subset in m_subsets:
            for _ in range(20):
                idx = rng.choice(range(self.m), replace=False, size=m_subset)
    
                delta_column = mgr.delta_column_vector(idx)
                
                # ensure column vector
                self.assertEqual(delta_column.shape, (m_subset, 1))
    
                # ensure equal values
                for i, j in enumerate(idx):
                    self.assertEqual(delta_column[i, 0], mgr.delta[j])

    def test_delta_column_vector_wrong_input(self):
        mgr = moca.GreedySearchMocaStatsManager(self.data, self.labels)

        with self.assertRaises(ValueError):
            mgr.delta_column_vector(list())



if __name__ == "__main__":
    main()
