from pomegranate.distributions._distribution import Distribution
from pomegranate._utils import _cast_as_tensor
import torch

class EmbeddingSimilarity(Distribution):
    def __init__(
            self,
            state_embed,
            inertia=0.0,
            frozen=False,
            check_data=True
        ):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = "Embedding similarity"
        self.state_embed = _cast_as_tensor(state_embed)

    def log_probability(self, X):
        X = _cast_as_tensor(X)
        return torch.nn.functional.cosine_similarity(self.state_embed, X).log()
    
class KeySimilarityMatrix(Distribution):
    def __init__(
            self,
            col_i: int,
            max_i: int,
            inertia=0.0,
            frozen=False,
            check_data=True
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