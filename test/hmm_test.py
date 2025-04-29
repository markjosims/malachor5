import sys
sys.path.append('scripts')
from hmm_utils import EmbeddingSimilarity, KeySimilarityMatrix, calculate_transition_probs, tag_sentence
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

def test_tag_tokens():
    keyphrase_list = [
        "foo foo foo foo foo",
        "foo bar bar baz",
        "bar baz baz bar"
    ]
    expected_tagged_keyphrases = [
        "foo_S foo_1 foo_2 foo_3 foo_F",
        "foo_S bar_1 bar_2 baz_F",
        "bar_S baz_1 baz_2 bar_F",
    ]
    tagged_keyphrases = [tag_sentence(sentence) for sentence in keyphrase_list]
    assert expected_tagged_keyphrases == tagged_keyphrases

def test_calculate_transition_probs():
    keyphrase_list = [
        "foo foo foo",
        "foo bar",
        "bar baz"
    ]
    entr_prob = 0.1
    skip_prob = 0.01
    exit_prob = 0.01
    self_prob = 0.5
    cont_prob = 1-(exit_prob+self_prob)
    transition_probs = calculate_transition_probs(
        keyphrase_list,
        enter_prob=entr_prob,
        self_trans_prob=self_prob,
        early_exit_prob=exit_prob,
        late_enter_prob=skip_prob,
    )
    expected_transitions = [
        ("SIL",   "SIL",   (1/2)*(1-(entr_prob+skip_prob))),
        ("SIL",   "SPCH",  (1/2)*(1-(entr_prob+skip_prob))),
        ("SPCH",  "SPCH",  (1/2)*(1-(entr_prob+skip_prob))),
        ("SPCH",  "SIL",   (1/2)*(1-(entr_prob+skip_prob))),

        ("SIL",   "foo_S", (2/3)*entr_prob),
        ("SPCH",  "foo_S", (2/3)*entr_prob),
        ("foo_S", "foo_S", self_prob),
        ("foo_S", "foo_1", (1/2)*cont_prob),
        ("foo_S", "bar_F", (1/2)*cont_prob),
        ("foo_S", "SIL",   (1/2)*exit_prob),
        ("foo_S", "SPCH",  (1/2)*exit_prob),

        ("SIL",   "foo_1", (1/4)*skip_prob),
        ("SPCH",  "foo_1", (1/4)*skip_prob),
        ("foo_1", "foo_1", self_prob),
        ("foo_1", "foo_F", (1/1)*cont_prob),
        ("foo_1", "SIL",   (1/2)*exit_prob),
        ("foo_1", "SPCH",  (1/2)*exit_prob),
        
        ("SIL",   "foo_F", (1/4)*skip_prob),
        ("SPCH",  "foo_F", (1/4)*skip_prob),
        ("foo_F", "foo_F", self_prob),
        ("foo_F", "SIL",   (1/2)*cont_prob),
        ("foo_F", "SPCH",  (1/2)*cont_prob),

        ("SIL",   "bar_F", (1/4)*skip_prob),
        ("SPCH",  "bar_F", (1/4)*skip_prob),
        ("bar_F", "bar_F", self_prob),
        ("bar_F", "SIL",   (1/2)*cont_prob),
        ("bar_F", "SPCH",  (1/2)*cont_prob),

        ("SIL",   "bar_S", (1/3)*entr_prob),
        ("SPCH",  "bar_S", (1/3)*entr_prob),
        ("bar_S", "bar_S", self_prob),
        ("bar_S", "baz_F", (1/1)*cont_prob),
        ("bar_S", "SIL",   (1/2)*exit_prob),
        ("bar_S", "SPCH",  (1/2)*exit_prob),

        ("SIL",   "baz_F", (1/4)*skip_prob),
        ("SPCH",  "baz_F", (1/4)*skip_prob),
        ("baz_F", "baz_F", self_prob),
        ("baz_F", "SIL",   (1/2)*cont_prob),
        ("baz_F", "SPCH",  (1/2)*cont_prob),
    ]

    bigrams = [t[:2] for t in transition_probs]
    expected_bigrams = [t[:2] for t in expected_transitions]
    missing_bigrams = [t for t in expected_bigrams if t not in bigrams]
    extra_bigrams = [t for t in bigrams if t not in expected_bigrams]

    # uncomment for debugging
    # duplicate_bigrams = [b for b in bigrams if bigrams.count(b)>1]
    # duplicate_transitions = [t for t in transition_probs if t[:2] in duplicate_bigrams]
    # extra_transitions = [t for t in transition_probs if t not in expected_transitions]

    assert len(missing_bigrams)==0
    assert len(extra_bigrams)==0


    assert len(transition_probs) == len(expected_transitions)
    for expected_transition in expected_transitions:
        bigram = expected_transition[:2]
        transition = [t for t in transition_probs if t[:2]==bigram]
        assert len(transition)==1
        transition_weight = transition[0][-1]
        expected_weight = expected_transition[-1]
        assert transition_weight == expected_weight