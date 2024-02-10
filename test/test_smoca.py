
from unittest import TestCase, main

import numpy as np

from moca import classifiers as cls
from moca import simulate as sim
from moca import stats


class TestSmoca(TestCase):
    rng = np.random.default_rng()

    def setUp(self):
        self.n_samples = 1000
        self.n_pos = 300
        self.m_classifiers = 10
        self.auc = self.rng.uniform(size=self.m_classifiers)
        self.corr_matrix = sim.make_corr_matrix(self.m_classifiers,
                                                independent=True,
                                                seed=self.rng)

        self.data, self.labels = sim.rank_scores(self.n_samples,
                                        self.n_pos,
                                        self.auc,
                                        self.corr_matrix,
                                        seed=self.rng)

    def test_inheritance(self):
        cl = cls.Smoca()
        self.assertTrue(isinstance(cl, cls.MocaABC))

    def test_default_init(self):
        cl = cls.Smoca()

        self.assertEqual("greedy", cl.subset_select)
        self.assertIsNone(cl.subset_select_par)
        self.assertEqual(cl.name, "Smoca-greedy")
        self.assertIsNone(cl.prevalence)
        self.assertIsNone(cl.weights)

    def test_wrong_fit_method(self):
        with self.assertRaises(ValueError):
            cls.Smoca(subset_select = "l1")

    def test_init(self):
        cl = cls.Smoca(subset_select=None)

        self.assertIsNone(cl.subset_select)
        self.assertIsNone(cl.subset_select_par)

        cl = cls.Smoca(subset_select=None)


    def test_no_subset_select(self):
        cl = cls.Smoca(subset_select=None)

        self.assertIsNone(cl.subset_select)
        self.assertIsNone(cl.subset_select_par)

        cl.fit(self.data, self.labels)

        self.assertEqual(cl.prevalence, self.n_pos/self.n_samples)
        self.assertEqual(cl.M, self.m_classifiers)

        #TODO need test for inferred weights

    def test_low_rank_warning(self):
        tmpdata = np.vstack([self.data, self.data[0,:]])

        cl =cls.Smoca(subset_select=None)

        with self.assertWarns(UserWarning):
            cl.fit(tmpdata, self.labels)


    def test_greedy_subset_selection_filter(self):
        cl =cls.Smoca(subset_select_par=0)

        with self.assertRaises(IndexError):
            cl.fit(self.data, self.labels)

    def test_greedy_subset_one(self):
        cl = cls.Smoca(subset_select_par=1)
        cl.fit(self.data, self.labels)
        
        delta_abs = np.abs(stats.delta(self.data, self.labels))
        best_idx = np.where(delta_abs == np.max(delta_abs))[0]
        
        for i in range(cl.M):
            if i != best_idx:
                self.assertEqual(cl.weights[i], 0)
                continue

            self.assertEqual(i, best_idx)
            self.assertTrue(cl.weights[i] != 0)

    def test_greedy_subset_size(self):

        for i in range(1, self.m_classifiers-1):
            cl = cls.Smoca(subset_select_par=i)
            cl.fit(self.data, self.labels)

            self.assertEqual(np.sum(cl.weights != 0), i)


    def test_find_optimal_subset(self):
        pass

    def test_fit(self):
        pass



class TestGreedySearchIdxManager(TestCase):
    def test_init(self):
        m = 10
        gs_idx = cls.GreedySearchIdxManager(m)

        self.assertEqual(gs_idx.m, m)
        self.assertEqual(gs_idx.found, [])
        self.assertEqual(gs_idx.complement, list(range(m)))

    def test_wrong_init_input(self):
        m = 10

        with self.assertRaises(ValueError):
            cls.GreedySearchIdxManager(float(m))

        with self.assertRaises(ValueError):
            cls.GreedySearchIdxManager(str(m))

        with self.assertRaises(ValueError):
            cls.GreedySearchIdxManager((m, ))

        with self.assertRaises(ValueError):
            cls.GreedySearchIdxManager([m])
 
        with self.assertRaises(ValueError):
            cls.GreedySearchIdxManager(np.array([m]))

        with self.assertRaises(ValueError):
            cls.GreedySearchIdxManager(set((m,)))

    def test_ens_properties(self):
        m = 10
        n = 4

        gs_idx = cls.GreedySearchIdxManager(m)

        rng = np.random.default_rng()
        ens_idx = list()
        for i in rng.choice(range(m), replace=False, size=n):

            # object creation by property
            self.assertNotEqual(id(gs_idx.found),
                                id(gs_idx._found))
            self.assertNotEqual(id(gs_idx.complement),
                                id(gs_idx._complement))

            # test created object values
            self.assertEqual(gs_idx.found, gs_idx._found)
            self.assertEqual(gs_idx.complement,
                             gs_idx._complement)


            # test returned prosepective ensemble indexes
            ens_idx.append(i)
            self.assertEqual(ens_idx, gs_idx.prospective(i))
            gs_idx.update(i)

    def test_update(self):
        m = 10
        gs_idx = cls.GreedySearchIdxManager(m)

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


# TODO test low rank warning
class TestGreedyStatsManager(TestCase):
    rng = np.random.default_rng()

    def setUp(self):
        self.m, self.n, self.n1 = 10, 1000, 300
        self.auc = np.array([1 for _ in range(self.m)])
        self.corr_matrix = sim.make_corr_matrix(self.m,
                                                independent=True)

        self.data, self.labels = sim.rank_scores(self.n,
                                                 self.n1,
                                                 self.auc,
                                                 self.corr_matrix,
                                                 seed=self.rng)


    def test_init(self):
        mgr = cls.GreedySearchMocaStatsManager(self.data,
                                               self.labels)

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
                self.assertEqual(mgr.cov_matrix[i,i],
                                 mgr.cov_matrix[j,j])

        # 3) positive definite
        l, v = np.linalg.eigh(mgr.cov_matrix)

        self.assertEqual(np.sum(np.abs(l)), np.sum(l))
        self.assertTrue(np.prod(l) > 0)

    def test_delta_column(self):
        mgr = cls.GreedySearchMocaStatsManager(self.data,
                                               self.labels)

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
                idx = rng.choice(range(self.m),
                                 replace=False, size=m_subset)
    
                delta_column = mgr.delta_column_vector(idx)
                
                # ensure column vector
                self.assertEqual(delta_column.shape,
                                 (m_subset, 1))
    
                # ensure equal values
                for i, j in enumerate(idx):
                    self.assertEqual(delta_column[i, 0],
                                     mgr.delta[j])

    def test_delta_column_vector_wrong_input(self):
        mgr = cls.GreedySearchMocaStatsManager(self.data,
                                               self.labels)

        with self.assertRaises(ValueError):
            mgr.delta_column_vector(list())



if __name__ == "__main__":
    main()
