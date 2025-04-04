from math import ceil
from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk
import datasets
from speechbrain.dataio.batch import PaddedBatch
import torch
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Union, Tuple, Literal
from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from string_norm import get_epitran
from transformers import WhisperProcessor
import os
from tokenization_utils import *
from argparse_utils import make_arggroup_from_argdict
import numpy as np
from copy import copy
from tqdm import tqdm

TRAIN_DS_ARGS = {
    'train_datasets': {'nargs': '+', 'help': 'Extra datasets for training'},
    'train_data_pct': {'type': float, 'help': 'Portion of train data to use (from 0 to 1)'},
    'train_dataset_languages': {'nargs': '+', 'help': 'Language for each extra train set'},
}
EVAL_DS_ARGS = {
    'eval_datasets': {'nargs': '+', 'help': 'Extra datasets for validation'},
    'eval_dataset_languages': {'nargs': '+', 'help': 'Language for each extra validation set'},
}
DATASET_ARGS = {
    'dataset': {'type': str},
    'num_records': {'abbreviation': 'n', 'type': int},
    'stream': {'action': 'store_true'},
    'fleurs_lang': {'type': str},
    'skip_idcs': {'nargs': '+', 'type': lambda x: [int(i) for i in x]},
    'skip_recordings': {'nargs': '+'},
    'make_split': {'action': 'store_true'},
    'g2p': {'action': 'store_true'},
    'transcription_ids': {'action': 'store_true', 'help': "Instead of tokenizing str in `transcription` column, load token ids directly from `transcription_ids` column"},
    'label_key': {'default': 'transcription'},
    'language': {'abbreviation': 'l', 'nargs': '+'},
    'load_ds_cache': {'abbreviation': 'c', 'action': 'store_true'},
}
DATASET_ARGS.update(**TRAIN_DS_ARGS, **EVAL_DS_ARGS)
DATASET_ARG_NAMES = list(DATASET_ARGS.keys())

# ------------- #
# data collator #
# ------------- #

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        if 'forced_decoder_ids' in features[0]:
            batch['forced_decoder_ids']=torch.stack(
                [torch.tensor(feature['forced_decoder_ids']) for feature in features]
            )

        batch["labels"] = labels

        return batch


def load_data_collator(model, processor):
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    return data_collator

# --------------------- #
# dataset preprocessing #
# --------------------- #

def prepare_dataset(
        row,
        processor,
        transcription_ids=False,
        g2p=None,
        label_key='transcription',
        decoder_prompt_ids=None,
    ):
    wav=row["audio"]["array"]
    sr=row["audio"]["sampling_rate"]
    label = row[label_key or 'transcription']
    row["input_features"] = processor(wav, sampling_rate=sr, return_tensors='np').input_features[0]
    row["input_length"] = ceil(len(wav)/sr)
    if transcription_ids:
        transcription_ids=row["transcription_ids"]
        if type(transcription_ids) is str:
            transcription_ids=eval(transcription_ids)
        label_ids=transcription_ids
    elif g2p:
        label=g2p.transliterate(label)
        label_ids = processor.tokenizer(label, return_tensors='np').input_ids[0]
    elif decoder_prompt_ids:
        label_ids = processor.tokenizer(label, return_tensors='np', add_special_tokens=False).input_ids[0]
        label_ids = np.concatenate([[BOS_TOKEN_ID], decoder_prompt_ids, label_ids, [EOS_TOKEN_ID]])
        row['forced_decoder_ids']=decoder_prompt_ids
    else:
        label_ids = processor.tokenizer(label, return_tensors='np').input_ids[0]
    row["labels"]=label_ids
    return row

def add_decoder_input_ids(
        row,
        processor: WhisperProcessor,
) -> Dataset:
    special_ids=processor.tokenizer.all_special_ids
    label_ids=row['labels']
    prefix=[
        tok_id for tok_id in label_ids if
        (tok_id in special_ids) and
        (tok_id!=EOS_TOKEN_ID)
    ]
    row['decoder_input_ids']=prefix
    return row

def load_dataset_safe(args) -> Union[Dataset, DatasetDict]:
    """
    If dataset points to a path on disk, load using `load_from_disk`,
    otherwise use `load_dataset` (to load from HF hub or local cache).
    """
    if hasattr(args, 'dataset'):
        dataset_path=args.dataset
    else:
        dataset_path=args.input
    split=getattr(args, 'split', None)
    make_split=getattr(args, 'make_split', False)
    if os.path.exists(dataset_path):
        if os.path.exists(
            os.path.join(dataset_path, 'dataset_dict.json')
        ):
            dataset=load_from_disk(dataset_path)
        else:
            dataset=load_dataset('audiofolder', data_dir=dataset_path)
        if split and args.num_records:
            return dataset[split].select(range(args.num_records))
        if split:
            return dataset[split]
        if make_split:
            dataset=make_ds_split(dataset)
        if args.num_records:
            for split in dataset:
                dataset[split]=dataset[split].select(range(args.num_records))
        return dataset

    if ('fleurs' in dataset_path) or ('common_voice' in dataset_path):
        return load_dataset(dataset_path, args.fleurs_lang, split=split, streaming=args.stream)
    dataset = load_dataset(dataset_path, split=split)
    if (args.num_records) and (not args.stream) and (split):
        dataset = dataset.select(range(args.num_records))
    return dataset

def load_sli_dataset(args) -> Tuple[Union[Dataset, DatasetDict], Namespace]:
    ds = load_dataset_safe(args)
    sli_map_path = getattr(
        args,
        'sli_map_path',
        os.path.join(args.dataset, 'sli_map.json')
    )
    with open(sli_map_path) as fh:
        sli_map = json.load(fh)
    args.sli_map = sli_map
    args.sli_label2id={}
    args.sli_id2label={}
    for language in sli_map:
        args.sli_label2id[language['label']]=language['id']
        args.sli_id2label[language['id']]=language['label']
    return ds, args

def load_and_prepare_dataset(args):
    ds = load_dataset_safe(args)
    processor = WhisperProcessor.from_pretrained(args.processor or args.model, task="transcribe")
    # set language prefix tokens
    # use native method if only one decoding one language
    decoder_prompt_ids=None
    # if args.language and len(args.language)==1:
    #     processor.tokenizer.set_prefix_tokens(
    #         language=args.language[0],
    #         task='transcribe',
    #     )
    if args.language:
        decoder_prompt_ids=get_forced_decoder_ids(processor.tokenizer, args.language, ids_only=True)
    # get a random split name dynamically since we don't know what splits are saved in dataset
    split_key=list(ds.keys())[0]
    if ds[split_key][0]["audio"]["sampling_rate"]!=16_000:
        print("Resampling to 16kHz...")
        ds=ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds_cache_files={}
    if args.action=='evaluate':
        ds=DatasetDict({'validation': ds['validation']})
        colnames=ds['validation'].column_names
    elif args.action=='test':
        ds=DatasetDict({'test': ds['test']})
        colnames=ds['test'].column_names
    else:
        colnames=ds['train'].column_names
    for split in ds:
        ds_cache_files[split]=os.path.join(args.dataset, split+'-cache.arrow')
    if args.skip_idcs:
        skip_range = list(
            i for i in range(len(ds['train']))
            if i not in args.skip_idcs
        )
        ds['train']=ds['train'].select(skip_range)
    if args.skip_recordings:
        ds['train'] = ds['train'].filter((lambda x: x['filestem'] not in args.skip_recordings))
    if args.train_data_pct and 'train' in ds:
        num_train = int(len(ds['train'])*(args.train_data_pct))
        ds['train'] = ds['train'].shuffle(seed=42).select(range(num_train))
    epitran=get_epitran(
        args.fleurs_lang,
        lang_key='fleurs' if 'fleurs' in args.dataset
        else 'commonvoice_code' if 'common_voice' in args.dataset
        else 'whisper'
    ) if args.g2p else None
    ds = ds.map(
        lambda b: prepare_dataset(
            b,
            processor,
            transcription_ids=args.transcription_ids,
            g2p=epitran,
            label_key=args.label_key,
            decoder_prompt_ids=decoder_prompt_ids,
        ),
        num_proc=1,
        remove_columns=colnames,
        cache_file_names=ds_cache_files,
        load_from_cache_file=bool(args.load_ds_cache),
    )
    # forced decoder ids not passed during training
    if 'train' in ds:
        ds['train']=ds['train'].remove_columns('forced_decoder_ids')

    if args.eval_datasets:
        print("Loading additional eval datasets...")
        eval_dataset_dict = load_extra_datasets(args, 'eval')
        if 'validation' in ds:
            dataset_stem = args.dataset.removesuffix('/').split('/')[-1]
            if args.eval_dataset_languages:
                dataset_stem+='-'+'+'.join(args.language or ['LID'])
            eval_dataset_dict[dataset_stem]=ds['validation']
        ds['validation']=eval_dataset_dict
    if args.train_datasets:
        print("Loading additional train datasets...")
        train_dataset_dict = load_extra_datasets(args, 'train')
        if 'train' in ds:
            dataset_stem = args.dataset.removesuffix('/').split('/')[-1]
            if args.train_dataset_languages:
                dataset_stem+='-'+'+'.join(args.language or ['LID'])
            ds['train']=ds['train'].add_column('dataset', [dataset_stem]*len(ds['train']))
            train_dataset_dict[dataset_stem]=ds['train']
        ds_concat = datasets.concatenate_datasets(list(train_dataset_dict.values()))
        ds_concat = ds_concat.shuffle(seed=42)
        ds['train']=ds_concat

    # if 'validation' in ds:
    #     ds['validation'] = ds['validation'].map(
    #         lambda row: add_decoder_input_ids(row, processor),
    #         batched=False,
    #         num_proc=4,
    #     )
    # if 'test' in ds:
    #     ds['test'] = ds['test'].map(
    #         lambda row: add_decoder_input_ids(row, processor),
    #         batched=False,
    #         num_proc=4,
    #     )
    return ds, processor

def load_extra_datasets(args, split: Literal['train', 'eval', 'test']) -> Dict[str, Dataset]:
    # if args.eval_dataset_languages:
    dataset_languages = getattr(args, f'{split}_dataset_languages', None)
    datasets = getattr(args, f'{split}_datasets', None)
    if dataset_languages:
        dataset_languages = [lang.split('+') for lang in dataset_languages]
    else:
        dataset_languages = [
            args.language for _ in datasets
        ]
    dataset_dict = {}

    for dataset, lang in tqdm(list(zip(datasets, dataset_languages)), desc=f'Extra {split} datasets'):
        tqdm.write(f'Preparing {dataset}...')
        dataset_args = copy(args)
        # avoid infinite recursion
        setattr(dataset_args, split+'_datasets', None)
        dataset_args.language=lang if lang!=['None'] else None
        # assuming that skip_recordings is only used for main dataset
        dataset_args.skip_recordings=None
        if 'fleurs' in dataset or 'commonvoice' in dataset:
            dataset_args.fleurs_lang = iso2_to_fleurs(lang[0])
        dataset_args.dataset=dataset
        dataset_args.action='evaluate' if split=='eval' else split
        dataset_obj, _ = load_and_prepare_dataset(dataset_args)
        
        dataset_name=dataset.removesuffix('/').split('/')[-1]
        if getattr(args, f'{split}_dataset_languages', None):
            # when specifying language for each dataset, include language in dataset name
            dataset_name+='-'+'+'.join(lang)
            dataset_name=dataset_name.replace('None', 'LID')
        split_key = 'validation' if split=='eval' else split
        dataset_obj[split_key]=dataset_obj[split_key].add_column('dataset', [dataset_name]*len(dataset_obj[split_key]))

        dataset_dict[dataset_name]=dataset_obj[split_key]
    return dataset_dict

def make_ds_split(dataset: DatasetDict, percent_val: float=0.2) -> DatasetDict:
    """
    Make an ad-hoc train-val split.
    Assume dataset only has `train`.
    Select the first `percent_val` records to go into the validation split.
    """
    num_records = len(dataset['train'])
    num_val=int(percent_val*num_records)
    dataset['validation']=dataset['train'].select(range(num_val))
    dataset['train']=dataset['train'].select(range(num_val, num_records))
    return dataset


def dataset_generator(dataset: Dataset) -> Generator:
    """
    For progress bars to work with the HuggingFace pipeline,
    the dataset must be wrapped in an iterable class,
    with the Pipeline object handling batching.
    """
    for row in dataset:
        yield row['audio']


def collate_dataset(batch):
    return PaddedBatch([{'wav':row['audio']['array']} for row in batch]).wav.data

def collate_chunks(batch):
    wavs=[{'wav':row['wav'].squeeze()} for row in batch]
    return PaddedBatch(wavs).wav.data

def build_sb_dataloader(dataset, batch_size, dataset_type: Literal['hf_dataset', 'chunk_list']='hf_dataset'):
    # create a dataloader that returns batches of wav objs
    if dataset_type == 'hf_dataset':
        collate_fn=collate_dataset
    else:
        collate_fn=collate_chunks
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    return dataloader

# --------------------- #
# Language code helpers #
# --------------------- #

def iso2_to_fleurs(iso2, raise_error=False):
    """
    Attempts to find "fleurs" code for a given language based on its provided iso2 code.
    If not found, defaults to returning the iso2 code.
    Set `raise_error=True` to override this and raise an error if no key is found. 
    """
    for language in LANGUAGE_CODES:
        if language.get('iso2', None)==iso2:
            fleurs=language.get('fleurs', None)
            if fleurs is None and raise_error:
                raise KeyError
            return fleurs or iso2

    if raise_error:
        raise KeyError
    return iso2

# ---------------- #
# Argparse methods #
# ---------------- #

def add_dataset_args(parser: ArgumentParser) -> ArgumentParser:
    make_arggroup_from_argdict(DATASET_ARGS, parser, 'dataset_args')
    return parser