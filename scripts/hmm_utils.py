from pomegranate.distributions._distribution import Distribution
from pomegranate._utils import _cast_as_tensor
from pomegranate.hmm import SparseHMM
import torch
from typing import *
from nltk import ngrams, FreqDist

BOS = '<s>'
EOS = '</s>'

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
    
def init_keyword_hmm(
        keyphrase_list: List[str],
        transprob_kwargs: Dict[str, float] = dict(),
        dist_type: Literal['embed_sim', 'sim_mat'] = 'sim_mat',
        embeddings: Optional[torch.Tensor] = None,
) -> Tuple[SparseHMM, ]:
    transitions, states = calculate_transition_probs(keyphrase_list, **transprob_kwargs)
    distribution_dict = {}
    if dist_type == 'sim_mat':
        for i, state in enumerate(states):
            distribution_dict[state]=KeySimilarityMatrix(col_i=i, max_i=len(states))
    else: # dist_type == 'embed_sim
        for i, state in enumerate(states):
            distribution_dict[state]=EmbeddingSimilarity(embeddings[i])
    hmm = SparseHMM(distributions=list(distribution_dict.values()))
    for instate, outstate, prob in transitions:
        hmm.add_edge(
            distribution_dict[instate],
            distribution_dict[outstate],
            prob
        )
    breakpoint()
    return hmm
    
def tag_sentence(sentence: str) -> str:
    """
    Given a string of a sentence, tag the first word with '_S' for 'start',
    the last word with '_F' for 'final', and all non-terminal words with '_1'
    unless the word occurs after itself, in which case tag the first instance with '_1',
    the second with '_2' and so on.
    If a sentence only has one word, tag word with '_O' for 'only'.
    """
    words = sentence.split()
    if len(words)==1:
        return words[0]+'_O'
    words[0]+='_S'
    words[-1]+='_F'
    for i in range(1, len(words)-1):
        word = words[i]
        prev_word, prev_tag = words[i-1].split('_')
        if (word == prev_word) and (prev_tag != 'S'):
            prev_tag_val = int(prev_tag)
            words[i] = f"{word}_{prev_tag_val+1}"
        else:
            words[i] = word+'_1'
    return ' '.join(words)
    
def is_non_terminal(word: str) -> bool:
    if word in [BOS, EOS]:
        return False
    if word.split('_')[-1] in ['S', 'F']:
        return False
    return True

def calculate_transition_probs(
        keyphrase_list: List[str],
        enter_prob: float = 0.1,
        self_trans_prob: float = 0.5,
        early_exit_prob: float = 0.01,
        late_enter_prob: float = 0.01,
        non_keyword_states: List[str] = ['SIL', 'SPCH']
    ) -> Tuple[
        List[Tuple[str, str, float]],
        Set[str]
    ]:
    """
    Calculates transition probabilities between words given a list of keyphrases.
    Returns a list of tuples of shape ('$start_word', '$end_word', $weight)
    and a set of strs corresponding to each unique state.
    `non_keyword_states` indicates the names of all non-keyword states, by default
    'SIL' for silence/non-speech and 'SPCH' for (non-keyword) speech.
    """
    transitions = []
    continue_prob = 1-(self_trans_prob+early_exit_prob)

    keyphrases_tagged = [tag_sentence(keyphrase) for keyphrase in keyphrase_list]
    bigrams = []
    unigrams = []
    for keyphrase in keyphrases_tagged:
        bigrams.extend(
            ngrams(
                keyphrase.split(),
                2,
                pad_left=True,
                pad_right=True,
                left_pad_symbol=BOS,
                right_pad_symbol=EOS
            )
        )
        unigrams.append(BOS)
        unigrams.extend(keyphrase.split())
        unigrams.append(EOS)
    bigram_cts = FreqDist(bigrams)
    unigram_cts = FreqDist(unigrams)
    # add bigram transitions
    for bigram, bigram_ct in bigram_cts.items():
        start_word, end_word = bigram
        start_word_ct = unigram_cts[start_word]
        trans_prob = bigram_ct/start_word_ct # Prob(end_word|start_word)

        if start_word == BOS:
            # transition into keyphrase
            transition_weight = trans_prob*enter_prob
            for non_keyword_state in non_keyword_states:
                transitions.append((non_keyword_state, end_word, transition_weight))
        elif end_word == EOS:
            # transition out of keyphrase
            # ignore `trans_prob`, instead scale by number of non-keyword states
            transition_weight = continue_prob*(1/len(non_keyword_states))
            for non_keyword_state in non_keyword_states:
                transitions.append((start_word, non_keyword_state, transition_weight))
        else:
            # transition within keyphrase
            transition_weight = trans_prob*continue_prob
            transitions.append((start_word, end_word, transition_weight))
    
    # add self-transition, late entry and early exit probs
    non_initial_words = [word for word in unigrams if is_non_terminal(word) or word.endswith('_F')]
    non_initial_ct = len(non_initial_words)
    for unigram, unigram_ct in unigram_cts.items():
        # Prob(unigram|unigram)
        # `self_trans_prob` only applies to words
        # non-keyword states transition to self or each other with equal probability
        if unigram == BOS:
            non_enter_prob = 1-(enter_prob+late_enter_prob)
            self_trans_weight = non_enter_prob/len(non_keyword_states)
            for start_state in non_keyword_states:
                for end_state in non_keyword_states:
                    transitions.append((start_state, end_state, self_trans_weight))
        elif unigram == EOS:
            # only do self-transitions of BOS to avoid duplicate transitions
            # since non-keyword states are same before and after sentence
            pass
        else:
            transitions.append((unigram, unigram, self_trans_prob))

        # late entry applies to non-initial words
        if is_non_terminal(unigram) or unigram.endswith('_F'):
            # Prob(unigram|non_keyword_state)
            late_enter_weight = late_enter_prob * unigram_ct/non_initial_ct
            for non_keyword_state in non_keyword_states:
                transitions.append((non_keyword_state, unigram, late_enter_weight))
        # early exit only apply to non-final words
        if is_non_terminal(unigram) or unigram.endswith('_S'):
            # Prob(non_keyword_state|unigram)
            early_exit_weight = early_exit_prob / len(non_keyword_states)
            for non_keyword_state in non_keyword_states:
                transitions.append((unigram, non_keyword_state, early_exit_weight))

    states = set(unigrams)
    states.remove(BOS)
    states.remove(EOS)
    for state in non_keyword_states:
        states.add(state)
    return transitions, states
    
    