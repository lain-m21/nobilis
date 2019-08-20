from __future__ import annotations

import numpy as np
from typing import Dict, Sequence, Any


class FeaturePool:
    """
    Contains features for items

    Attributes
    ----------
    query_dict: Dict[Any, Sequence[int]]
        map query id to items
    feature_table: np.ndarray
        2d numpy table, whose i-th row corresponds to the feature of i-th item
    """

    def __init__(self,
                 query_dict: Dict[Any, Sequence[int]],
                 feature_table: np.ndarray,
                 ) -> None:
        assert feature_table.ndim == 2
        self.feature_table = feature_table
        self.query_dict = query_dict

    def save(self, filepath: str) -> None:
        raise NotImplementedError

    @staticmethod
    def load(self, filepath: str) -> FeaturePool:
        raise NotImplementedError

    def get_query_features(self, query_id) -> np.ndarray:
        return self.feature_table[self.query_dict[query_id]]
