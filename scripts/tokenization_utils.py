import json

TRANSCRIBE_TOKEN_ID=50359
BOS_TOKEN_ID=50258
EOS_TOKEN_ID=50257
NOTIMESTAMPS_ID=50363
TIRA_ASR_DS = 'data/pyarrow-datasets/tira-clean-split'
TIRA_DRZ = 'data/pyarrow-datasets/tira-drz'
TIRA_SLI = 'data/pyarrow-datasets/tira-sli'
FLEURS = 'data/pyarrow-datasets/fl_en'
TIRA_BILING = 'data/pyarrow-datasets/HH20210913'
with open('meta/whisper_special_tokens.json') as f:
    SPECIAL_TOKENS = json.load(f)
LANG_TOKENS = SPECIAL_TOKENS['lang']
LANG_TOKEN_IDS = [lang_obj['id'] for lang_obj in LANG_TOKENS.values()]
FUNCTIONAL_TOKENS = SPECIAL_TOKENS['functional']
SPECIAL_TOKENS_FLAT = dict(**LANG_TOKENS, **FUNCTIONAL_TOKENS)
SB_VOXLINGUA = 'speechbrain/lang-id-voxlingua107-ecapa'

with open('meta/language_codes.json') as f:
    LANGUAGE_CODES = json.load(f)

def get_forced_decoder_ids(tokenizer, language=None, ids_only=False):
    """
    Get task and language prompt tokens for languages specified by `.language`
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
    if ids_only:
        forced_decoder_ids.sort(key=lambda t:t[0])
        forced_decoder_ids=[t[1] for t in forced_decoder_ids]
    return forced_decoder_ids
