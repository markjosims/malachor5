import json
from transformers import WhisperTokenizer
import sys
sys.path.append('scripts')
import os
from lid_utils import get_word_language
from string_norm import remove_punct, unicode_normalize

PYARROW_DIR=os.environ.get('MALACHOR5_PYARROW_DIR', 'data/pyarrow-datasets')
DATA_DIR=os.environ.get('MALACHOR5_DATA_DIR', 'data/')
TRANSCRIBE_TOKEN_ID=50359
BOS_TOKEN_ID=50258
EOS_TOKEN_ID=50257
NOTIMESTAMPS_ID=50363
TIRA_ASR_DS = os.path.join(PYARROW_DIR, 'tira-asr')
TIRA_DRZ = os.path.join(PYARROW_DIR, 'tira-drz')
TIRA_LONGFORM = os.path.join(DATA_DIR, 'longform-drz')
TIRA_SLI = os.path.join(PYARROW_DIR, 'tira-sli')
FLEURS = os.path.join(PYARROW_DIR, 'fl_en')
TIRA_BILING = os.path.join(PYARROW_DIR, 'HH20210913')
with open('meta/whisper_special_tokens.json') as f:
    SPECIAL_TOKENS = json.load(f)
LANG_TOKENS = SPECIAL_TOKENS['lang']
LANG_TOKEN_IDS = [lang_obj['id'] for lang_obj in LANG_TOKENS.values()]
FUNCTIONAL_TOKENS = SPECIAL_TOKENS['functional']
SPECIAL_TOKENS_FLAT = dict(**LANG_TOKENS, **FUNCTIONAL_TOKENS)
SB_VOXLINGUA = 'speechbrain/lang-id-voxlingua107-ecapa'
TOKENIZER = WhisperTokenizer.from_pretrained('openai/whisper-medium')

with open('meta/language_codes.json') as f:
    LANGUAGE_CODES = json.load(f)

def get_forced_decoder_ids(tokenizer, language=None, ids_only=False):
    """
    Get task and language prompt tokens for languages specified by `language` arg
    and task 'transcribe'. By default returns a list of tuples, [(i, token_id), ...].
    If `ids_only`, pass a list of token ids sorted by `i`.
    """
    forced_decoder_ids=set()
    language_list = language if type(language) is list else [language]
    for language in language_list or [None]:
        forced_decoder_ids.update(
                tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
        )
    forced_decoder_ids=list(forced_decoder_ids)
    forced_decoder_ids.sort(key=lambda t:t[0])
    if ids_only:
        forced_decoder_ids=[t[1] for t in forced_decoder_ids]
    return forced_decoder_ids


def normalize_eng_words_only(s: str, tokenizer: WhisperTokenizer=None) -> str:
    """
    Normalize all non-Tira words in string, leaving Tira unchanged
    """
    if tokenizer is None:
        tokenizer = TOKENIZER
    norm_words = []
    for word in s.split():
        if get_word_language(word)!='eng':
            norm_word = unicode_normalize(word)
            norm_word = remove_punct(word)
            norm_words.append(norm_word)
        else:
            norm_words.append(tokenizer.normalize(word))
    return ' '.join(norm_words)

    