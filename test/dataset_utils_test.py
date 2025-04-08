from argparse import Namespace
from test_utils import assert_tokens_in_row, assert_labels_begin_with, assert_labels_end_with, assert_tokens_appear_once, assert_tokens_not_in_row
from datasets import Dataset
import json

import sys
sys.path.append('scripts')
from dataset_utils import load_and_prepare_dataset, DATASET_ARG_NAMES, TIRA_ASR_DS, FLEURS, TIRA_BILING
from train_whisper import init_parser

def test_dataset_language():
    args = init_parser().parse_args([])
    args.dataset=TIRA_ASR_DS
    args.language=['en']
    args.model='openai/whisper-tiny'
    args.num_records=50
    args.action='evaluate'

    ds, _ = load_and_prepare_dataset(args)
    ds.map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['en', 'transcribe', 'startoftranscript', 'eos', 'notimestamps'],
        ),
        batched=False,
    )

def test_dataset_multi_language():
    args = init_parser().parse_args([])
    args.dataset=TIRA_ASR_DS
    args.language=['en', 'sw']
    args.model='openai/whisper-tiny'
    args.num_records=50
    args.action='evaluate'

    ds, _ = load_and_prepare_dataset(args)
    ds.map(
        lambda row: assert_tokens_in_row(
            row,
            token_names=['en', 'sw', 'transcribe', 'startoftranscript', 'eos', 'notimestamps'],
        ),
        batched=False,
    )

def test_eval_datasets():
    args = init_parser().parse_args([])
    args.dataset=TIRA_ASR_DS
    args.language=['sw']
    args.model='openai/whisper-tiny'
    args.num_records=50
    args.eval_datasets=[FLEURS, TIRA_BILING]
    args.eval_dataset_languages=['en', 'sw+en']
    args.action='evaluate'

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

    assert 'tira-asr-sw' in eval_datasets
    assert type(eval_datasets['tira-asr-sw']) is Dataset
    eval_datasets['tira-asr-sw'].map(
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
    args = init_parser().parse_args([])
    args.dataset=TIRA_ASR_DS
    args.language=['sw']
    args.model='openai/whisper-tiny'
    args.num_records=50
    args.action='evaluate'
    
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

def test_prompt_input_ids_added(tmpdir):
    args = init_parser().parse_args([])
    args.dataset=TIRA_ASR_DS
    args.language=['sw']
    args.model='openai/whisper-tiny'
    args.num_records=5
    prompt_file = str(tmpdir/'prompts.txt')
    args.prompt_file=prompt_file

    prompts = {
        'validation': [
            'This is a prompt',
            'This is also a prompt.',
            'This is a prompt?!?',
            'Promptly prompting proper promptable prompters.',
            'Foo bar baz.'
        ],
        'test': [
            'Testing the prompt.',
            'Prompting the test.',
            'Prompting the test.',
            'Foo bar bazzing the test.',
            'Baz barring the foo.',
        ]
    }
    with open(prompt_file, 'w') as f:
        json.dump(prompts, f)
    ds, processor = load_and_prepare_dataset(args)
    validation_prompts = processor.batch_decode(
        ds['validation']['prompt_ids'], skip_special_tokens=True
    )
    test_prompts = processor.batch_decode(
        ds['test']['prompt_ids'], skip_special_tokens=True
    )
    assert validation_prompts == prompts['validation']
    assert test_prompts == prompts['test']
    assert 'prompt_ids' not in ds['train'].column_names
        

def test_label_prefix_added():
    args = init_parser().parse_args([])
    args.dataset=TIRA_ASR_DS
    args.language=['sw']
    args.model='openai/whisper-tiny'
    args.num_records=50
    args.action='evaluate'

    ds, _ = load_and_prepare_dataset(args)
    ds['validation'].map(
        lambda row: assert_labels_begin_with(
            row,
            token_names=['startoftranscript', 'sw', 'transcribe', 'notimestamps'],
        )
    )
    ds['validation'].map(
        lambda row: assert_labels_end_with(
            row,
            token_names=['eos'],
        )
    )
    ds['validation'].map(
        lambda row: assert_tokens_appear_once(
            row,
            token_names=['startoftranscript', 'sw', 'transcribe', 'notimestamps', 'eos'],
        )
    )

def test_skip_recordings():
    args = init_parser().parse_args([])
    args.dataset=TIRA_ASR_DS
    args.language=['sw']
    args.model='openai/whisper-tiny'
    args.action='train'
    args.skip_recordings=['HH20210312']

    ds, _ = load_and_prepare_dataset(args)
    # 16384 records in the base train dataset
    # minus 230 records from recording HH20210312
    assert len(ds['train'])==16154

def test_train_datasets():
    args = init_parser().parse_args([])
    args.dataset=TIRA_ASR_DS
    args.language=['sw']
    args.train_datasets=[FLEURS, FLEURS]
    args.train_dataset_languages=['en', 'en+sw']
    args.model='openai/whisper-tiny'
    args.action='train'
    args.num_records=10
    
    for arg in DATASET_ARG_NAMES:
        if not hasattr(args, arg):
            setattr(args, arg, None)
    ds, _ = load_and_prepare_dataset(args)
    fleurs = ds['train'].filter(lambda row: row['dataset']=='fl_en-en')
    fleurs.map(assert_tokens_in_row, fn_kwargs={'token_names':['en', 'transcribe', 'startoftranscript', 'eos', 'notimestamps']})
    fleurs.map(assert_tokens_not_in_row, fn_kwargs={'token_names':['sw']})
    tira_biling = ds['train'].filter(lambda row: row['dataset']=='fl_en-en+sw')
    tira_biling.map(assert_tokens_in_row, fn_kwargs={'token_names':['en', 'sw', 'transcribe', 'startoftranscript', 'eos', 'notimestamps']})
    tira_mono = ds['train'].filter(lambda row: row['dataset']=='tira-asr-sw')
    tira_mono.map(assert_tokens_in_row, fn_kwargs={'token_names':['sw', 'transcribe', 'startoftranscript', 'eos', 'notimestamps']})
    tira_mono.map(assert_tokens_not_in_row, fn_kwargs={'token_names':['en']})