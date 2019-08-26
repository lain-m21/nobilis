from nobilis.feature_pool import FeaturePool
import numpy as np


def test_feature_pool():
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    query_dict = {1: [0, 1], 2: [2, 3]}
    pool = FeaturePool(query_dict, data)
    expected = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(pool.get_query_features(1), expected)
    expected = np.array([[5, 6], [7, 8]])
    np.testing.assert_array_equal(pool.get_query_features(2), expected)
