from typing import Union, List
import torch

def embed_speech(audio: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
    ...

def embed_text(text: Union[str, List[str]]) -> torch.Tensor:
    ...