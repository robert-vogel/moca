from unittest import TestCase, main

import numpy as np
from moca import stats

class TestMeanRank(TestCase):
    def input_filter(self):
        with self.assertRaises(ValueError):
            stats.mean_rank([1,2,3])

        with self.assertRaises(ValueError):
            stats.mean_rank((1,2,3))

        with self.assertRaises(ValueError):
            stats.mean_rank(np.array([1,2,3]))

        with self.assertRaises(ValueError):
            stats.mean_rank({"a":1,"b":2,"c":3})

        with self.assertRaises(ValueError):
            stats.mean_rank(1)

        with self.assertRaises(ValueError):
            stats.mean_rank(-10)

    def test_value(self):
        for n in range(2, 100):
            self.assertEqual((n+1)/2, stats.mean_rank(n))


class TestVarianceRank(TestCase):
    def input_filter(self):
        with self.assertRaises(ValueError):
            stats.variance_rank([1,2,3])

        with self.assertRaises(ValueError):
            stats.variance_rank((1,2,3))

        with self.assertRaises(ValueError):
            stats.variance_rank(np.array([1,2,3]))

        with self.assertRaises(ValueError):
            stats.variance_rank({"a":1,"b":2,"c":3})

        with self.assertRaises(ValueError):
            stats.mean_rank(1)

        with self.assertRaises(ValueError):
            stats.mean_rank(-10)

    def test_value(self):
        for n in range(2, 100):
            self.assertEqual((n**2-1)/12, stats.variance_rank(n))

class TestThirdMoment(TestCase):
    def setUp(self):
        self.m, self.n = 10, 2500
        ranks = np.arange(1, self.n+1)

        rng = np.random.default_rng()
        self.data = np.zeros(shape=(self.m, self.n))
        self.norm_constant = self.n / ((self.n-1) * (self.n-2))

        for i in range(self.m):
            self.data[i, :] = rng.choice(ranks, size=self.n, replace=False)


    def test_inputs(self):
        with self.assertRaises(AttributeError):
            stats.third_central_moment(1)

        with self.assertRaises(AttributeError):
            stats.third_central_moment(list(range(100)))

        with self.assertRaises(ValueError):
            rng = np.random.default_rng()
            stats.third_central_moment(rng.random(size=(10,10,10)))

    def test_warning(self):
        """Warning when the number of samples is less than features."""
        with self.assertWarns(UserWarning):
            stats.third_central_moment(self.data[:, :self.m-2])
           
    def test_1d_moment(self):
        idx = 0
        mean = np.mean(self.data[idx, :])
        T_true = self.norm_constant * np.sum((self.data[idx, :] - mean)**3)

        self.assertEqual(T_true, stats.third_central_moment(self.data[idx]))

    def test_2d_elements(self):
        # compute third moment
        T = stats.third_central_moment(self.data)

        means = np.mean(self.data, axis=1)

        for i in range(self.m):

            for j in range(self.m):

                for k in range(self.m):

                    T_element_ijk = (self.norm_constant * 
                                        np.sum((self.data[i,:] - means[i]) *
                                                (self.data[j,:] - means[j]) *
                                                (self.data[k,:] - means[k])))

                    self.assertEqual(T_element_ijk, T[i,j,k])

class TestNorm(TestCase):
    def setUp(self):
        rng = np.random.default_rng()
        self.data = np.arange(10)
        rng.shuffle(self.data)

    def test_not_numpy_list_tuple(self):
        with self.assertRaises(ValueError):
            stats.l2_vector_norm(10)

        # test if it would work on iterables
        with self.assertRaises(ValueError):
            stats.l2_vector_norm(range(10))

        with self.assertRaises(ValueError):
            tmp = {}
            for i, val in enumerate(self.data):
                tmp[i] = val

            stats.l2_vector_norm(tmp)

    def test_norm_value(self):
        # as data is the first 9 integers
        # the norm should be the square root
        # of the square pyramidal number
        n = self.data.size -1
        Pn = n**3 / 3 + n**2 / 2 + n / 6
        norm_true = np.sqrt(Pn)

        self.assertEqual(stats.l2_vector_norm(self.data),
                        norm_true)


        self.assertEqual(stats.l2_vector_norm(self.data.tolist()),
                        norm_true)

        self.assertEqual(stats.l2_vector_norm(tuple(self.data.tolist())),
                        norm_true)

class TestDelta(TestCase):
    pass

class TestMocaCov(TestCase):
    pass

class TestSnr(TestCase):
    pass

class TestRank(TestCase):
    def setUp(self):
        self.scores = np.array([[0.1, 0.8, 0.5, 0.3, 0.4],
                                [0.11, 0.7, 0.9, 0.2, 0.14]])
        self.rank_descending = np.array([[5, 1, 2, 4, 3],
                                         [5, 2, 1, 3, 4]])
        self.rank_ascending = self.rank_descending.copy()
        self.rank_ascending = self.rank_descending.max() - self.rank_ascending + 1

    def test_rank_transform_ascending(self):
        test_rank = stats.rank_transform(self.scores, ascending=True)

        for i in range(self.scores.shape[0]):
            self.assertTrue(all(test_rank[i, :] == self.rank_ascending[i, :]))

    def test_rank_transform_descending(self):
        # test simple 2-d case
        test_rank = stats.rank_transform(self.scores)

        for i in range(self.scores.shape[0]):
            self.assertTrue(all(test_rank[i, :] == self.rank_descending[i, :]))

    def test_rank_descend_nans(self):
        # test nan
        self.scores[0, 2] = np.nan
        # once the np.nan element is udpated to the median the effective
        # score[0, :] is [0.1, 0.8, 0.35, 0.3, 0.4], and consequently
        # true_rank[0, :] = [5,1,3,4,2]
        self.rank_descending[0, [2, 4]] = 3, 2
        test_rank = stats.rank_transform(self.scores)
        for i in range(self.scores.shape[0]):
            self.assertTrue(all(test_rank[i, :] == self.rank_descending[i, :]))

    def test_rank_descend_1d(self):
        # test simple 1-d case
        scores = np.array([0.1, 0.8, 0.5, 0.3, 0.4])
        true_rank = np.array([5, 1, 2, 4, 3])

        self.assertTrue(all(stats.rank_transform(scores) == true_rank))

    def test_rank_ascend_1d(self):
        # test simple 1-d case
        scores = np.array([0.1, 0.8, 0.5, 0.3, 0.4])
        true_rank = np.array([1, 5, 4, 2, 3])

        self.assertTrue(all(stats.rank_transform(scores, ascending=True) == true_rank))


class TestIsRank(TestCase):
    pass

class TestIsPrevalence(TestCase):
    def test_correct(self):
        self.assertTrue(stats.is_prevalence(0.1))

    def test_error(self):
        self.assertFalse(stats.is_prevalence(-0.1))
        self.assertFalse(stats.is_prevalence(1.1))
        self.assertFalse(stats.is_prevalence("a"))
        self.assertFalse(stats.is_prevalence([0.1]))


class TestIsAuc(TestCase):
    def test_correct(self):
        self.assertTrue(stats.is_auc(0.2))
        self.assertTrue(stats.is_auc([0.1, 0.2, 0.8]))
        self.assertTrue(stats.is_auc(np.array([0.1, 0.9])))

    def test_false(self):
        self.assertFalse(stats.is_auc(-0.2))
        self.assertFalse(stats.is_auc("a"))
        self.assertFalse(stats.is_auc("ac"))
        self.assertFalse(stats.is_auc([1.1, 0.2, 0.8]))
        self.assertFalse(stats.is_auc(np.array([0.1, -0.9])))


if __name__ == "__main__":
    main()
