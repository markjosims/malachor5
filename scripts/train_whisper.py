# Finetune Whisper model on a new dataset using HF
# Large part of code taken from https://huggingface.co/blog/fine-tune-whisper on Sep 17 2024

from argparse import ArgumentParser
from typing import Sequence, Optional, Dict, Any, List, Union
from asr_dataset import load_dataset_safe, DEVICE, device_type
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Audio
import torch
from dataclasses import dataclass
import evaluate
import os

wer = evaluate.load("wer")

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
    # 'debug': 'underflow_overflow', 
}
HYPERPARAM_ABBREVIATIONS = {
    'per_device_train_batch_size': 'b',
    'num_train_epochs': 'e',
    'gradient_accumulation_steps': 'g',
}

# ---------------- #
# Argparse methods #
# ---------------- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--output', '-o')
    parser.add_argument('--model', '-m')
    parser.add_argument('--num_records', '-n', type=int)
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
            hyper_args.add_argument(*flags, default=v, action='store_true')
        else:
            hyper_args.add_argument(*flags, type=type(v), default=v)
    return parser

# --------------------- #
# dataset preprocessing #
# --------------------- #

def load_and_prepare_dataset(args):
    ds = load_dataset_safe(args)
    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task="transcribe")
    if ds['train'][0]["audio"]["sampling_rate"]!=16_000:
        ds=ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds_cache_files={}
    for split in ds:
        ds_cache_files[split]=os.path.join(args.dataset, split+'-cache.arrow')
    ds = ds.map(
        lambda b: prepare_dataset(b, processor),
        num_proc=20,
        remove_columns=ds['train'].column_names,
        cache_file_names=ds_cache_files,
    )
    return ds, processor

def prepare_dataset(row, processor):
    wav=row["audio"]["array"]
    sr=row["audio"]["sampling_rate"]
    label = row["transcription"]
    row["input_features"] = processor(wav, sampling_rate=sr).input_features
    row["labels"] = processor.tokenizer(label).input_ids
    return row

# ------------- #
# data collator #
# ------------- #

def load_data_collator(model, processor):
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    return data_collator

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

# ------------------ #
# evaluation methods #
# ------------------ #

def compute_wer(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# ----------------- #
# model preparation #
# ----------------- #

def load_whisper_model(args) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return model

# ------------- #
# training args #
# ------------- #

def get_training_args(args):
    arg_dict={k: getattr(args, k) for k in DEFAULT_HYPERPARAMS.keys()}
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        **arg_dict,
    )
    return training_args

# ---- #
# main #
# ---- #

def main(argv: Sequence[Optional[str]]=None) -> int:
    parser=init_parser()
    args=parser.parse_args(argv)

    print("Preparing dataset...")
    ds, processor = load_and_prepare_dataset(args)
    print("Loading model...")
    model = load_whisper_model(args)
    print("Making data collator...")
    data_collator = load_data_collator(model, processor)
    print("Defining training args...")
    training_args = get_training_args(args)
    print("Defining metrics...")
    compute_metrics = lambda pred: compute_wer(pred, processor.tokenizer)

    print("Training!")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()

    return 0

if __name__ == '__main__':
    main()