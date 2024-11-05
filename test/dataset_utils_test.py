from argparse import Namespace
from test_utils import assert_tokens_in_row
from datasets import Dataset

import sys
sys.path.append('scripts')
from dataset_utils import load_and_prepare_dataset, load_eval_datasets, DATASET_ARGS

TIRA_ASR_DS = 'data/pyarrow-datasets/tira-clean-split'
FLEURS = 'data/pyarrow-datasets/fl_en'
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
        language=['en'],
        model='openai/whisper-tiny',
        num_records=50,
    )
    for arg in DATASET_ARGS:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    ds, _ = load_and_prepare_dataset(args)
    ds.map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['en', 'transcribe', 'bos', 'eos', 'notimestamps'],
            special_tokens=SPECIAL_TOKENS
        ),
        batched=False,
    )

def test_dataset_multi_language():
    args = Namespace(
        dataset=TIRA_ASR_DS,
        language=['en', 'sw'],
        model='openai/whisper-tiny',
        num_records=50,
    )
    for arg in DATASET_ARGS:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    ds, _ = load_and_prepare_dataset(args)
    ds.map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['en', 'sw', 'transcribe', 'bos', 'eos', 'notimestamps'],
            special_tokens=SPECIAL_TOKENS
        ),
        batched=False,
    )

def test_eval_datasets():
    args = Namespace(
        dataset=TIRA_ASR_DS,
        language=['sw'],
        model='openai/whisper-tiny',
        num_records=50,
        eval_datasets=[TIRA_ASR_DS, FLEURS],
        eval_dataset_languages=['sw', 'en']
    )
    for arg in DATASET_ARGS:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    eval_datasets = load_eval_datasets(args)
    assert type(eval_datasets) is dict

    assert 'fl_en' in eval_datasets
    assert type(eval_datasets['fl_en']) is Dataset
    eval_datasets['fl_en'].map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['en', 'transcribe', 'bos', 'eos', 'notimestamps'],
            special_tokens=SPECIAL_TOKENS
        ),
        batched=False,
    )

    assert 'tira-clean-split' in eval_datasets
    assert type(eval_datasets['tira-clean-split']) is Dataset
    eval_datasets['tira-clean-split'].map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['sw', 'transcribe', 'bos', 'eos', 'notimestamps'],
            special_tokens=SPECIAL_TOKENS
        ),
        batched=False,
    )