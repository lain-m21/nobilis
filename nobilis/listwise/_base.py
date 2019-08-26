from __future__ import annotations

from typing import Dict, Sequence, Hashable, Tuple
from nobilis.feature_pool import FeaturePool


class BaseListwiseRanker:
    """
    Base class for listwise ranker
    """

    def __init__(self):
        pass

    def fit(self,
            train_query: Sequence[Hashable],
            target: Dict[Hashable, Tuple[int, ...]],
            pool: FeaturePool) -> None:
        """
        function for fitting. 

        Parameters
        ----------
        train_query: Sequence[Any]
            Contains queries for train. Each elements must be the valid key of target and query dict of pool.
        target: Dict[Any, Tuple[int, ...]]
            Map to query id to listwise relations. Tuple contains the rank of items in each query.
        pool: FeaturePool
            Contains features of each item
        """
        raise NotImplementedError

    