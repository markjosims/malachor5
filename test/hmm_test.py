import sys
sys.path.append('scripts')
from hmm_utils import EmbeddingSimilarity, KeySimilarityMatrix
from kws import get_similarity_matrix
import torch

from test_utils import NYEN_PATH, ALBRRIZO_PATH, XDDERE_PATH
from test_utils import NYEN_IPA, ALBRRIZO_IPA, XDDERE_IPA
from test_utils import SAMPLE_BILING_PATH, SAMPLE_BILING_TG_PATH, ZAVELEZE_IPA, NGINE_IPA

def test_embedding_similarity_distribution():
    state_embed = [1,   0, 1]
    close_embed = [0.9, 0, 1]
    orth_embed =  [0,   1, 0]
    embedsim = EmbeddingSimilarity(state_embed=state_embed)
    stateprobs=embedsim.log_probability([state_embed, close_embed, orth_embed])
    assert stateprobs.argmax(0)==0
    assert stateprobs[1]<stateprobs[0]
    assert stateprobs[1]>stateprobs[2]
    assert stateprobs.argmin(0)==2


def test_key_similarity_distribution():
    keysim1 = KeySimilarityMatrix(0, 2)
    keysim2 = KeySimilarityMatrix(1, 2)
    state1_embed = [0.5, 1,   0]
    state2_embed = [1,   0.5, 0]
    close1_embed = [0.5, 0.9,   0]
    close2_embed = [0.9, 0.5, 0]
    orth_embed =   [0,   0,   1]

    simmat = get_similarity_matrix(
        row_embeds=torch.tensor([close1_embed, close2_embed, orth_embed]),
        col_embeds=torch.tensor([state1_embed, state2_embed]),
    )
    keysim1_probs=keysim1.log_probability(simmat)
    assert keysim1_probs.argmax(0)==0
    assert keysim1_probs[1]<keysim1_probs[0]
    assert keysim1_probs[1]>keysim1_probs[2]
    assert keysim1_probs.argmin(0)==2


    keysim2_probs=keysim2.log_probability(simmat)
    assert keysim2_probs[0]<keysim2_probs[1]
    assert keysim2_probs[0]>keysim2_probs[2]
    assert keysim2_probs.argmax(0)==1
    assert keysim2_probs.argmin(0)==2