
from unittest import TestCase, main
import numpy as np


from moca import stats
from moca import cross_validate as cv

                       

class TestSampleCV(TestCase):
    def test_index_sampling(self):
        """Verify index sampling.  

        No index in the training set should be in the test set,
        and the no index in the test set should be in the training
        set.  The set of test set indexes should be 
        equal to all possible indexes, and the number of indexes
        should be the number of samples.
        """

        n_samples=675

        idx = np.arange(n_samples)

        for kfolds in range(2, 10):

            cumulate_test = []

            for train, test in cv.sample_idx(n_samples, kfolds):
                # np.setdiff1d(x,y) finds the values in x not
                # in y
                self.assertEqual(np.setdiff1d(test, train).size, 
                        test.size)
                self.assertEqual(np.setdiff1d(train, test).size,
                        train.size)

                cumulate_test.append(test)

            cumulate_test = np.hstack(cumulate_test)
            self.assertEqual(np.setdiff1d(cumulate_test, idx).size,
                    0)
            self.assertEqual(np.setdiff1d(idx, cumulate_test).size,
                    0)

            self.assertEqual(cumulate_test.size, n_samples)

    def test_minimal_size_requirement(self):
        """Verify requirement that nsamples > 3*kfolds"""
        kfolds = 10

        for nsamples in range(1, kfolds*3 + 1):
            with self.assertRaises(ValueError):
                next(cv.sample_idx(nsamples, kfolds))

    def test_no_shuffle(self):

        nsamples = 100
        rng = np.random.default_rng()

        for kfolds in range(2, 11):

            seed= rng.choice(1000000000)

            g1 = cv.sample_idx(nsamples, kfolds, 
                    reverse_order=False, shuffle=False, seed=seed)
            g2 = cv.sample_idx(nsamples, kfolds, 
                    reverse_order=False, shuffle=False, seed=seed)
    
            for g1_idx, g2_idx in zip(g1, g2):
                # compare train set indexes
                for a,b in zip(g1_idx[0], g2_idx[0]):
                    self.assertEqual(a,b)
    
                # compare test set indexes
                for a, b in zip(g1_idx[1], g2_idx[1]):
                    self.assertEqual(a,b)

    def test_shuffle(self):
        nsamples = 100
        rng = np.random.default_rng()

        for kfolds in range(2, 11):

            test_set_size = int(np.floor(nsamples / kfolds))
            test_set_mod = nsamples % kfolds

            seed= rng.choice(1000000000)

            g1 = cv.sample_idx(nsamples, kfolds, 
                    reverse_order=False, shuffle=True, seed=seed)
            g2 = cv.sample_idx(nsamples, kfolds, 
                    reverse_order=False, shuffle=True, seed=seed)

            k = 0
            for g1_idx, g2_idx in zip(g1, g2):
                # compare train set indexes
                for a,b in zip(g1_idx[0], g2_idx[0]):
                    self.assertEqual(a,b)

                # compare test set indexes
                for a, b in zip(g1_idx[1], g2_idx[1]):
                    self.assertEqual(a,b)

                if k < test_set_mod:
                    self.assertEqual(test_set_size + 1, len(g1_idx[1]))
                else:
                    self.assertEqual(test_set_size, len(g1_idx[1]))
                k += 1

    def test_shuffle_reverse_order(self):
        nsamples = 100
        rng = np.random.default_rng()

        for kfolds in range(2, 11):

            test_set_size = int(np.floor(nsamples / kfolds))
            test_set_mod_reverse = kfolds - nsamples % kfolds

            seed= rng.choice(1000000000)

            g1 = cv.sample_idx(nsamples, kfolds, 
                    reverse_order=True, shuffle=True, seed=seed)
            g2 = cv.sample_idx(nsamples, kfolds, 
                    reverse_order=True, shuffle=True, seed=seed)
    
            k = 0
            for g1_idx, g2_idx in zip(g1, g2):
                # compare train set indexes
                for a,b in zip(g1_idx[0], g2_idx[0]):
                    self.assertEqual(a,b)
    
                # compare test set indexes
                for a, b in zip(g1_idx[1], g2_idx[1]):
                    self.assertEqual(a,b)

                if k >= test_set_mod_reverse:
                    self.assertEqual(test_set_size + 1, len(g1_idx[1]))
                else:
                    self.assertEqual(test_set_size, len(g1_idx[1]))
                k += 1



if __name__ == "__main__":
    main()
