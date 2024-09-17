# Finetune Whisper model on a new dataset using HF
# Large part of code taken from https://huggingface.co/blog/fine-tune-whisper on Sep 17 2024

from argparse import ArgumentParser
from typing import Sequence, Optional, Dict, Any
from asr_dataset import load_dataset_safe, DEVICE, device_type
from transformers import WhisperProcessor
from datasets import Audio

DEFAULT_HYPERPARAMS = {
    'group_by_length': True,
    'per_device_train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'evaluation_strategy': "epoch",
    'save_strategy': "epoch",
    'num_train_epochs': 4,
    'gradient_checkpointing': True,
    'fp16': False,
    'save_steps': 5000,
    'eval_steps': 5000,
    'logging_steps': 1000,
    'learning_rate': 3e-4,
    'warmup_steps': 500,
    'report_to': 'tensorboard',
    'debug': 'underflow_overflow',
}
HYPERPARAM_ABBREVIATIONS = {
    'per_device_train_batch_size': 'b',
    'num_train_epochs': 'e',
}

# ---------------- #
# Argparse methods #
# ---------------- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--model', '-m')
    parser.add_argument('--num_rows', '-n', type=int)
    parser.add_argument('--split', '-s', default='train')
    parser.add_argument('--device', '-D', default=DEVICE, type=device_type)
    parser.add_argument('--language', '-l')
    parser = add_hyperparameter_args(parser)
    return parser

def add_hyperparameter_args(parser: ArgumentParser) -> None:
    hyper_args = parser.add_argument_group(
        'hyperparameters',
        description='Hyperparameter values for training'
    )
    for k, v in DEFAULT_HYPERPARAMS.items():
        flags=['--'+k,]
        if k in HYPERPARAM_ABBREVIATIONS:
            flags.append('-'+HYPERPARAM_ABBREVIATIONS[k])
        if type(v) is bool:
            hyper_args.add_argument('--'+k, default=v, action='store_true')
        else:
            hyper_args.add_argument('--'+k, type=type(v), default=v)
    return parser

# --------------------- #
# dataset preprocessing #
# --------------------- #

def load_and_prepare_dataset(args):
    ds = load_dataset_safe(args)
    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task="transcribe")
    if ds[0]["audio"]["sampling_rate"]!=16_000:
        ds=ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.map(
        lambda b: prepare_dataset(b, processor),
        batched=True,
        remove_columns=ds['train'].column_names
    )
    return ds

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

# ---- #
# main #
# ---- #

def main(argv: Sequence[Optional[str]]=None) -> int:
    return 0

if __name__ == '__main__':
    main()