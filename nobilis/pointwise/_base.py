from __future__ import annotations

from typing import Dict, Sequence, Any
import numpy as np
from nobilis.feature_pool import FeaturePool


class BasePointwiseRanker:
    """
    Base class for pointwise ranker
    """

    def __init__(self):
        pass

    def fit(self,
            train_query: Sequence[Any],
            target: Dict[Any, np.ndarray],
            pool: FeaturePool) -> None:
        """
        function for fitting. 

        Parameters
        ----------
        train_query: Sequence[Any]
            Contains queries for train. Each elements must be the valid key of target and query dict of pool.
        target: Dict[Any, np.ndarray]
            Map to query id to target values. 
        pool: FeaturePool
            Contains features of each item
        """
        raise NotImplementedError
