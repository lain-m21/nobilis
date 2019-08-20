from __future__ import annotations

from typing import Dict, Sequence, Any, Tuple
from nobilis.feature_pool import FeaturePool


class BasePairwiseRanker:
    """
    Base class for pairwise ranker
    """

    def __init__(self):
        pass

    def fit(self,
            train_query: Sequence[Any],
            target: Dict[Any, Sequence[Tuple[int, int]]],
            pool: FeaturePool) -> None:
        """
        function for fitting. 

        Parameters
        ----------
        train_query: Sequence[Any]
            Contains queries for train. Each elements must be the valid key of target and query dict of pool.
        target: Dict[Any, Sequence[Tuple[int, int]]]
            Map to query id to pairwise relations. The first element in each tuple should be rank higher than the second element.
        pool: FeaturePool
            Contains features of each item
        """
        raise NotImplementedError

    