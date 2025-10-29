from typing import Protocol, List, Dict, Any
import numpy as np

class Sampler(Protocol):
    def select(self, pool_rows: List[Dict[str, Any]], K: int,
               embeddings: np.ndarray, uncertainties: np.ndarray,
               state: Dict[str, Any]) -> List[str]: ...
