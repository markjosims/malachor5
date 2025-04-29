from pomegranate.distributions._distribution import Distribution
from pomegranate._utils import _cast_as_tensor
import torch
from typing import *

class EmbeddingSimilarity(Distribution):
    """
    Defines a 'distribution' whose probability is defined as the cosine similarity
    between `self.state_embed` and `X`, the observed embedding input to `self.log_probability`.
    Enables using embedding similarity as a observation probabilities with an HMM.
    """
    def __init__(
            self,
            state_embed: Sequence[float],
            inertia: float=0.0,
            frozen: bool=False,
            check_data: bool=True
        ):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = "Embedding similarity"
        self.state_embed = _cast_as_tensor(state_embed)

    def log_probability(self, X):
        X = _cast_as_tensor(X)
        return torch.nn.functional.cosine_similarity(self.state_embed, X).log()
    
class KeySimilarityMatrix(Distribution):
    """
    Defines a 'distribution' whose probability is determined by the value of a certain column,
    indicates by `self.col_i`, in a similarity matrix with `self.max_i` total columns.
    Allows using embedding similarity as an observation probability with an HMM iterating
    over a matrix of embedding similarities pre-calculated for a given audio sequence. 
    """
    def __init__(
            self,
            col_i: int,
            max_i: int,
            inertia: float=0.0,
            frozen: bool=False,
            check_data: bool=True
        ):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.col_i = col_i
        self.d = max_i
        self.name = "KeySimilarityMatrix"
    
    def log_probability(self, X):
        X = _cast_as_tensor(X)
        return X[:,self.col_i].log()
    
    def _reset_cache(self):
        return