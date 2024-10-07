# Finetune Whisper model on a new dataset using HF
# Large part of code taken from https://huggingface.co/blog/fine-tune-whisper on Sep 17 2024

from argparse import ArgumentParser
from typing import Sequence, Optional, Dict, Any, List, Union
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutomaticSpeechRecognitionPipeline
from datasets import Audio, load_dataset, load_from_disk, Dataset, DatasetDict
import torch
from dataclasses import dataclass
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from jiwer import wer, cer
from math import ceil
import pandas as pd

import os

DEFAULT_HYPERPARAMS = {
    'group_by_length': True,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_accumulation_steps': 16,
    'eval_strategy': "epoch",
    'save_strategy': "epoch",
    'num_train_epochs': 4,
    'gradient_checkpointing': False,
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
    'per_device_eval_batch_size': 'B',
    'num_train_epochs': 'e',
    'gradient_accumulation_steps': 'g',
}

DEVICE = 0 if torch.cuda.is_available() else 'cpu'
device_type = lambda s: int(s) if s!='cpu' else s

# ---------------- #
# Argparse methods #
# ---------------- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--make_split', action='store_true')
    parser.add_argument('--output', '-o')
    parser.add_argument('--model', '-m')
    parser.add_argument('--num_records', '-n', type=int)
    parser.add_argument('--device', '-D', default=DEVICE, type=device_type)
    parser.add_argument('--language', '-l')
    parser.add_argument('--peft_type', choices=['LoRA'])
    parser.add_argument('--load_ds_cache', '-c', action='store_true')
    parser.add_argument('--resume_from_checkpoint' ,action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--action', choices=['train', 'evaluate', 'test'], default=['train'])
    parser.add_argument('--eval_output')
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

    if 'fleurs' in dataset_path:
        return load_dataset(dataset_path, args.fleurs_lang, split=split, streaming=args.stream)
    dataset = load_dataset(dataset_path, split=split)
    if (args.num_records) and (not args.stream) and (split):
        dataset = dataset.select(range(args.num_records))
    return dataset

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
        num_proc=4,
        remove_columns=ds['train'].column_names,
        cache_file_names=ds_cache_files,
        load_from_cache_file=bool(args.load_ds_cache),
    )
    return ds, processor

def prepare_dataset(row, processor):
    wav=row["audio"]["array"]
    sr=row["audio"]["sampling_rate"]
    label = row["transcription"]
    row["input_features"] = processor(wav, sampling_rate=sr, return_tensors='np').input_features[0]
    row["input_length"] = ceil(len(wav)/sr)
    row["labels"] = processor.tokenizer(label, return_tensors='np').input_ids[0]
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

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    Taken from https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
    on 25 Sep 2024
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

def compute_wer_cer(pred, tokenizer):
    predictions = pred.predictions
    # if type(predictions) is tuple:
    #     # got logits instead of ids, decode greedily
    #     pred_ids = np.argmax(predictions[0], axis=-1)
    # else:
    #     # assume pred.predictions is the ids otherwise
    pred_ids = predictions[0]
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    batch_wer = wer(label_str, pred_str)
    batch_cer = cer(label_str, pred_str)

    return {"wer": batch_wer, "cer": batch_cer}


# ----------------- #
# model preparation #
# ----------------- #

def load_whisper_model_for_training(args) -> WhisperForConditionalGeneration:
    if args.peft_type and args.checkpoint:
        model = load_whisper_peft(args)
        model = set_generation_config(args, model)
        return model
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model = set_generation_config(args, model)
    if args.peft_type == 'LoRA':
        print("Wrapping model with LoRA...")
        # TODO add LoRA args to CLI
        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    return model

def set_generation_config(args, model):
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return model

def load_whisper_peft(args) -> WhisperForConditionalGeneration:
    model_path = args.checkpoint or args.model
    peft_config = PeftConfig.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
    )
    model = PeftModel.from_pretrained(model, model_path)
    return model

def load_whisper_pipeline(args) -> AutomaticSpeechRecognitionPipeline:
    if args.peft:
        model = load_whisper_peft(args)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model)
    tokenizer = WhisperTokenizer.from_pretrained(args.model)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        batch_size=args.batch_size,
        device=args.device,
    )
    return pipe

# ------------- #
# training args #
# ------------- #

def get_training_args(args):
    arg_dict={k: getattr(args, k) for k in DEFAULT_HYPERPARAMS.keys()}
    for k, v in arg_dict.items():
        # -1 passed to CLI indicates arg value should be None
        if v==-1:
            arg_dict[k]=None
    if args.peft_type=='LoRA':
        arg_dict['remove_unused_columns']=False
        arg_dict['label_names']=["labels"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        **arg_dict,
    )
    return training_args

# -------- #
# evaluate #
# -------- #

def evaluate_dataset(args, ds_split, processor, trainer):
    predictions=trainer.predict(ds_split)
    labels_decoded=processor.tokenizer.batch_decode(
                predictions.predictions[0]
    )
    output_decoded=processor.tokenizer.batch_decode(
                predictions.predictions[1]
    )
    torch.save(predictions, args.eval_output+'.pt' or os.path.join(args.output, 'predictions.pt'))
    df=pd.DataFrame({'labels_decoded': labels_decoded, 'output_decoded': output_decoded})
    df.to_csv(args.output+'.csv' or os.path.join(args.output, 'predictions.csv'))
    print(predictions.metrics)

# ---- #
# main #
# ---- #

def main(argv: Sequence[Optional[str]]=None) -> int:
    parser=init_parser()
    args=parser.parse_args(argv)

    print("Preparing dataset...")
    ds, processor = load_and_prepare_dataset(args)
    print("Loading model...")
    model = load_whisper_model_for_training(args)
    print("Making data collator...")
    data_collator = load_data_collator(model, processor)
    print("Defining training args...")
    training_args = get_training_args(args)
    print("Defining metrics...")
    compute_metrics = lambda pred: compute_wer_cer(pred, processor.tokenizer)

    print("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    if args.action=='train':
        trainer.train(resume_from_checkpoint=args.checkpoint or args.resume_from_checkpoint)
        save_dir=os.path.join(args.output, 'pretrained')
        trainer.save_model(save_dir)
        processor.save_pretrained(save_dir)

        if args.eval_output:
            evaluate_dataset(args, ds['validation'], processor, trainer)
    elif args.action=='evaluate':
        evaluate_dataset(args, ds['validation'], processor, trainer)

    else:
        # args.action == 'test'
        evaluate_dataset(args, ds['test'], processor, trainer)


    return 0

if __name__ == '__main__':
    main()