from argparse import Namespace
from test_utils import assert_tokens_in_row
from datasets import Dataset

import sys
sys.path.append('scripts')
from dataset_utils import load_and_prepare_dataset, DATASET_ARGS, TIRA_ASR_DS, FLEURS, TIRA_BILING, SPECIAL_TOKENS

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
        eval_datasets=[FLEURS, TIRA_BILING],
        eval_dataset_languages=['en', 'sw+en']
    )
    for arg in DATASET_ARGS:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    ds, _ = load_and_prepare_dataset(args)
    eval_datasets=ds['validation']
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

    assert 'HH20210913' in eval_datasets
    assert type(eval_datasets['HH20210913']) is Dataset
    eval_datasets['HH20210913'].map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['sw', 'en', 'transcribe', 'bos', 'eos', 'notimestamps'],
            special_tokens=SPECIAL_TOKENS
        ),
        batched=False,
    )

def test_decoder_input_added():
    args = Namespace(
        dataset=TIRA_ASR_DS,
        language=['sw'],
        model='openai/whisper-tiny',
        num_records=50,
    )
    for arg in DATASET_ARGS:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    ds, _ = load_and_prepare_dataset(args)
    assert 'decoder_input_ids' in ds['validation'][0]
    ds['validation'].map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['sw', 'transcribe', 'bos', 'notimestamps'],
            special_tokens=SPECIAL_TOKENS,
            col='decoder_input_ids'
        )
    )