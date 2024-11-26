from argparse import Namespace
from test_utils import assert_tokens_in_row
from datasets import Dataset
import math

import sys
sys.path.append('scripts')
from dataset_utils import load_and_prepare_dataset, DATASET_ARGS, TIRA_ASR_DS, FLEURS, TIRA_BILING

def test_dataset_language():
    args = Namespace(
        dataset=TIRA_ASR_DS,
        language=['en'],
        model='openai/whisper-tiny',
        num_records=50,
        action='evaluate',
    )
    for arg in DATASET_ARGS:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    ds, _ = load_and_prepare_dataset(args)
    ds.map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['en', 'transcribe', 'startoftranscript', 'eos', 'notimestamps'],
        ),
        batched=False,
    )

def test_dataset_multi_language():
    args = Namespace(
        dataset=TIRA_ASR_DS,
        language=['en', 'sw'],
        model='openai/whisper-tiny',
        num_records=50,
        action='evaluate',
    )
    for arg in DATASET_ARGS:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    ds, _ = load_and_prepare_dataset(args)
    ds.map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['en', 'sw', 'transcribe', 'startoftranscript', 'eos', 'notimestamps'],
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
        eval_dataset_languages=['en', 'sw+en'],
        action='evaluate',
    )
    for arg in DATASET_ARGS:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    ds, _ = load_and_prepare_dataset(args)
    eval_datasets=ds['validation']
    assert type(eval_datasets) is dict

    assert 'fl_en-en' in eval_datasets
    assert type(eval_datasets['fl_en-en']) is Dataset
    eval_datasets['fl_en-en'].map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['en', 'transcribe', 'startoftranscript', 'eos', 'notimestamps'],
        ),
        batched=False,
    )

    assert 'tira-clean-split-sw' in eval_datasets
    assert type(eval_datasets['tira-clean-split-sw']) is Dataset
    eval_datasets['tira-clean-split-sw'].map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['sw', 'transcribe', 'startoftranscript', 'eos', 'notimestamps'],
        ),
        batched=False,
    )

    assert 'HH20210913-sw+en' in eval_datasets
    assert type(eval_datasets['HH20210913-sw+en']) is Dataset
    eval_datasets['HH20210913-sw+en'].map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['sw', 'en', 'transcribe', 'startoftranscript', 'eos', 'notimestamps'],
        ),
        batched=False,
    )

def test_decoder_input_added():
    args = Namespace(
        dataset=TIRA_ASR_DS,
        language=['sw'],
        model='openai/whisper-tiny',
        num_records=50,
        action='evaluate',
    )
    for arg in DATASET_ARGS:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    ds, _ = load_and_prepare_dataset(args)
    assert 'forced_decoder_ids' in ds['validation'][0]
    ds['validation'].map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['sw', 'transcribe'],
            col='forced_decoder_ids'
        )
    )
    ds['validation'].map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['sw', 'transcribe', 'startoftranscript', 'notimestamps'],
        )
    )

def test_label_prefix_added():
    args = Namespace(
        dataset=TIRA_ASR_DS,
        language=['sw'],
        model='openai/whisper-tiny',
        num_records=50,
        action='evaluate',
    )
    for arg in DATASET_ARGS:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    ds, _ = load_and_prepare_dataset(args)