from nobilis.feature_pool import FeaturePool
import numpy as np


def test_feature_pool():
    data = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    query_dict = {1: [0, 1], 2: [2, 3]}
    pool = FeaturePool(query_dict, data)
    assert (pool.get_query_features(1) == np.array([[1, 2], [3, 4]])).all()
    assert (pool.get_query_features(2) == np.array([[1, 2], [3, 4]])).all()