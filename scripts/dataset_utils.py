from math import ceil
from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk
from speechbrain.dataio.batch import PaddedBatch
import torch
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Union
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from string_norm import get_epitran
from transformers import WhisperProcessor
import os
from model_utils import get_forced_decoder_ids
import numpy as np

DATASET_ARGS = [
    'dataset',
    'num_records',
    'stream',
    'fleurs_lang',
    'skip_idcs',
    'make_split',
    'processor',
    'model',
    'action',
    'g2p',
    'load_ds_cache',
    'transcription_ids',
    'label_key',
]
TRANSCRIBE_TOKEN_ID=50359


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
        labels=transcription_ids
    elif g2p:
        label=g2p.transliterate(label)
        labels = processor.tokenizer(label, return_tensors='np').input_ids[0]
    else:
        labels = processor.tokenizer(label, return_tensors='np').input_ids[0]
    if decoder_prompt_ids:
        task_tok_idx = np.argwhere(labels==TRANSCRIBE_TOKEN_ID).item()
        labels = np.concatenate([labels[:task_tok_idx],decoder_prompt_ids,labels[task_tok_idx+1:]])
    row["labels"]=labels
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
            os.path.join(dataset_path, 'metadata.csv')
        ):
            dataset=load_dataset('audiofolder', data_dir=dataset_path)
        else:
            dataset=load_from_disk(dataset_path)
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


def load_and_prepare_dataset(args):
    ds = load_dataset_safe(args)
    processor = WhisperProcessor.from_pretrained(args.processor or args.model, task="transcribe")
    # set language prefix tokens
    # use native method if only one decoding one language
    decoder_prompt_ids=None
    if args.language and len(args.language)==1:
        processor.tokenizer.set_prefix_tokens(
            language=args.language[0],
            task='transcribe',
        )
    elif args.language:
        decoder_prompt_ids=get_forced_decoder_ids(args, processor.tokenizer, ids_only=True)
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
        num_proc=4,
        remove_columns=colnames,
        cache_file_names=ds_cache_files,
        load_from_cache_file=bool(args.load_ds_cache),
    )
    return ds, processor


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


def collate_sb(batch):
    return PaddedBatch([{'wav':row['audio']['array']} for row in batch]).wav.data


def build_dataloader(dataset, batch_size):
    # create a dataloader that returns batches of wav objs
    # dataset = dataset.map(lambda row: {'wav': row['audio']['array']})
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_sb
    )

    return dataloader

# ---------------- #
# Argparse methods #
# ---------------- #

def add_dataset_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--num_records', '-n', type=int)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--fleurs_lang')
    parser.add_argument('--skip_idcs', nargs='+', type=int)
    parser.add_argument('--make_split', action='store_true')
    return parser