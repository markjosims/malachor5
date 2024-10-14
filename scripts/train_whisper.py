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
from glob import glob
from tqdm import tqdm
from string_norm import get_remove_oov_char_funct, condense_tones

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
    'predict_with_generate': False,
    'generation_num_beams': 1,
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
    parser.add_argument('--processor')
    parser.add_argument('--num_records', '-n', type=int)
    parser.add_argument('--transcription_ids', action='store_true')
    parser.add_argument('--device', '-D', default=DEVICE, type=device_type)
    parser.add_argument('--language', '-l')
    parser.add_argument('--peft_type', choices=['LoRA'])
    parser.add_argument('--ft_peft_model', action='store_true')
    parser.add_argument('--load_ds_cache', '-c', action='store_true')
    parser.add_argument('--resume_from_checkpoint' ,action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--action', choices=['train', 'evaluate', 'test'], default='train')
    parser.add_argument('--all_chkpnts', action='store_true')
    parser.add_argument('--eval_output')
    parser.add_argument('--char_vocab')
    parser.add_argument('--condense_tones', action='store_true')
    parser.add_argument('--skip_idcs', nargs='+', type=int)
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
    processor = WhisperProcessor.from_pretrained(args.processor or args.model, language=args.language, task="transcribe")
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
    ds = ds.map(
        lambda b: prepare_dataset(b, processor, transcription_ids=args.transcription_ids),
        num_proc=4,
        remove_columns=colnames,
        cache_file_names=ds_cache_files,
        load_from_cache_file=bool(args.load_ds_cache),
    )
    return ds, processor

def prepare_dataset(row, processor, transcription_ids=False):
    wav=row["audio"]["array"]
    sr=row["audio"]["sampling_rate"]
    label = row["transcription"]
    row["input_features"] = processor(wav, sampling_rate=sr, return_tensors='np').input_features[0]
    row["input_length"] = ceil(len(wav)/sr)
    if transcription_ids:
        transcription_ids=row["transcription_ids"]
        if type(transcription_ids) is str:
            transcription_ids=eval(transcription_ids)
        row["labels"]=transcription_ids
    else:
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

def get_str_process_pipe(args):
    str_process_pipe=[]
    if args.condense_tones:
        str_process_pipe.append(condense_tones)
    if args.char_vocab:
        remove_oov=get_remove_oov_char_funct(args.char_vocab)
        str_process_pipe.append(remove_oov)

    if not str_process_pipe:
        return
    
    def do_str_process_pipe(s_list):
        for f in str_process_pipe:
            s_list=[f(s) for s in s_list]
        return s_list

    return do_str_process_pipe

def get_metrics(args, processor, return_decoded=False):
    str_process_pipe=get_str_process_pipe(args)
    compute_metrics = lambda pred: compute_wer_cer(
        pred, processor.tokenizer,
        output_process_f=str_process_pipe,
        return_decoded=return_decoded,
    )
    return compute_metrics

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    Taken from https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
    on 25 Sep 2024
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

def compute_wer_cer(pred, tokenizer, output_process_f=None, return_decoded=False):
    pred_ids = pred.predictions
    if type(pred_ids) is tuple:
        pred_ids = pred_ids[0]
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    batch_wer = wer(label_str, pred_str)
    batch_cer = cer(label_str, pred_str)
    batch_metrics={
        "wer": batch_wer, 
        "cer": batch_cer,

    }
    if return_decoded:
        batch_metrics["labels"]=label_str
        batch_metrics["preds"]=pred_str

    if output_process_f:
        pred_str_processed=output_process_f(pred_str)
        batch_metrics['cer_processed']=cer(label_str, pred_str_processed)
        batch_metrics['wer_processed']=wer(label_str, pred_str_processed)
        if return_decoded:
            batch_metrics['preds_processed']=pred_str_processed

    return batch_metrics

def evaluate_dataset(args, ds_split, trainer, processor):
    metric_key_prefix = 'test' if args.action=='test' else 'eval'
    # change metrics to return labels
    trainer.compute_metrics=get_metrics(args, processor=processor, return_decoded=True)
    predictions=trainer.predict(ds_split, metric_key_prefix=metric_key_prefix, use_cache=False)
    # breakpoint()
    labels=predictions.metrics[f'{metric_key_prefix}_labels']
    preds=predictions.metrics[f'{metric_key_prefix}_preds']
    preds_processed=predictions.metrics.get(f'{metric_key_prefix}_preds_processed', None)
    df=pd.DataFrame({'labels': labels, 'preds': preds})
    if not (preds_processed is None):
        df['preds_processed']=preds_processed
    df.to_csv(
        args.eval_output+'.csv' if args.eval_output
        else os.path.join(args.output, f'{args.action}-predictions.csv')
    )
    predictions.metrics.pop(f'{metric_key_prefix}_labels')
    predictions.metrics.pop(f'{metric_key_prefix}_preds')
    predictions.metrics.pop(f'{metric_key_prefix}_preds_processed', None)

    torch.save(
        predictions,
        args.eval_output+'.pt' if args.eval_output
        else os.path.join(args.output, f'{args.action}-predictions.pt')
    )
    print(predictions.metrics)
    return predictions

def evaluate_all_checkpoints(args, ds, processor, training_args, compute_metrics):
    chkpnts=glob(
                os.path.join(args.output, 'checkpoint-*')
            )
    eval_output_stem=args.eval_output or args.output
    metrics=[]
    for chkpnt in tqdm(chkpnts, desc='Evaluating checkpoints'):
        chkpnt=chkpnt.removesuffix('/')
        args.checkpoint=chkpnt
        chkpnt_basename=os.path.basename(chkpnt)
        args.eval_output=os.path.join(eval_output_stem, chkpnt_basename)
        tqdm.write(f"Loading {chkpnt}...")
        chkpnt_model = load_whisper_model_for_training_or_eval(args)
        chkpnt_model = set_generation_config(args, chkpnt_model, processor.tokenizer)
        data_collator=load_data_collator(chkpnt_model, processor)
        trainer = Seq2SeqTrainer(
                    args=training_args,
                    model=chkpnt_model,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    tokenizer=processor.feature_extractor,
                    preprocess_logits_for_metrics=preprocess_logits_for_metrics if not args.predict_with_generate else None,
                )
        predictions=evaluate_dataset(args, ds['validation'], trainer, processor)
        metrics.append(predictions.metrics)
        metrics[-1]['checkpoint']=chkpnt
    csv_path=os.path.join(eval_output_stem, 'checkpoints-eval.csv')
    df=pd.DataFrame(data=metrics)
    df.to_csv(csv_path, index=False)

# ----------------- #
# model preparation #
# ----------------- #

def load_whisper_model_for_training_or_eval(args) -> WhisperForConditionalGeneration:
    if args.ft_peft_model:
        model = load_peft_model_for_finetuning(args)
    elif args.action in ('evaluate', 'test') and args.peft_type:
        return load_whisper_peft(args)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model)
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

def set_generation_config(args, model, tokenizer):
    forced_decoder_ids=get_forced_decoder_ids(args, tokenizer)
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    return model

def get_forced_decoder_ids(args, tokenizer):
    forced_decoder_ids=set()
    for language in args.language or [None]:
        forced_decoder_ids.update(
                tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
            )
    forced_decoder_ids=list(forced_decoder_ids)
    return forced_decoder_ids

def load_peft_model_for_finetuning(args):
    model = load_whisper_peft(args)
    print("Merging PEFT model for further finetuning...")
    model = model.merge_and_unload()
    return model

def load_whisper_peft(args) -> WhisperForConditionalGeneration:
    model_path = args.checkpoint or args.model
    peft_config = PeftConfig.from_pretrained(model_path)
    model_basename = peft_config.base_model_name_or_path
    print(f"Loading adapters from {model_path} for model {model_basename}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_basename,
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
        chunk_length_s=getattr(args, 'chunk_length_s', None),
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

# ---- #
# main #
# ---- #

def main(argv: Sequence[Optional[str]]=None) -> int:
    parser=init_parser()
    args=parser.parse_args(argv)

    print("Preparing dataset...")
    ds, processor = load_and_prepare_dataset(args)
    print("Defining metrics...")
    compute_metrics = get_metrics(args, processor)
    print("Defining training args...")
    training_args = get_training_args(args)

    if not args.all_chkpnts:
        print("Loading model...")
        model = load_whisper_model_for_training_or_eval(args)
        print("Setting model generation config...")
        model = set_generation_config(args, model, processor.tokenizer)
        print("Making data collator...")
        data_collator = load_data_collator(model, processor)
        print("Initializing trainer...")
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if not args.predict_with_generate else None,
        )
    if args.action=='train':
        trainer.train_dataset=ds['train']
        trainer.eval_dataset=ds['validation']
        print("Training!")
        trainer.train(resume_from_checkpoint=args.checkpoint or args.resume_from_checkpoint)
        save_dir=os.path.join(args.output, 'pretrained')
        trainer.save_model(save_dir)
        processor.save_pretrained(save_dir)

        if args.eval_output:
            evaluate_dataset(args, ds['validation'], trainer, processor)
    elif args.action=='evaluate':
        if args.all_chkpnts:
            evaluate_all_checkpoints(args, ds, processor, training_args, compute_metrics)
        else:
            evaluate_dataset(args, ds['validation'], trainer, processor)

    else:
        # args.action == 'test'
        evaluate_dataset(args, ds['test'], trainer, processor)


    return 0



if __name__ == '__main__':
    main()