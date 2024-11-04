from argparse import Namespace
from test_utils import assert_tokens_in_row

import sys
sys.path.append('scripts')
from dataset_utils import load_and_prepare_dataset, DATASET_ARGS

TIRA_ASR_DS = 'data/pyarrow-datasets/tira-asr-hf'
SPECIAL_TOKENS = {
    'en':           {'token': '<|en|>',                 'id': 50259},
    'sw':           {'token': '<|sw|>',                 'id': 50318},
    'bos':          {'token': '<|startoftranscript|>',  'id': 50258},
    'eos':          {'token': '<|endoftext|>',          'id': 50257},
    'notimestamps': {'token': '<|notimestamps|>',       'id': 50363},
    'transcribe':   {'token': '<|transcribe|>',         'id': 50359},
}

def test_dataset_language():
    args = Namespace(
        dataset=TIRA_ASR_DS,
        language='en',
        model='openai/whisper-tiny',
        num_records=50,
    )
    for arg in DATASET_ARGS:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    ds, _ = load_and_prepare_dataset(args)
    ds.map(
        lambda row: assert_tokens_in_row(row, languages=['en'], special_tokens=SPECIAL_TOKENS),
        batched=False,
    )
    