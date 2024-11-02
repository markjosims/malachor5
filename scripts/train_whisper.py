# Finetune Whisper model on a new dataset using HF
# Large part of code taken from https://huggingface.co/blog/fine-tune-whisper on Sep 17 2024

from argparse import ArgumentParser
from typing import Sequence, Optional
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from jiwer import wer, cer
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
from dataset_utils import load_and_prepare_dataset, load_data_collator, add_dataset_args
from model_utils import load_whisper_model_for_training_or_eval, set_generation_config, add_processor_args, add_whisper_model_args, device_type, DEVICE
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

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--output', '-o')
    parser.add_argument('--ft_peft_model', action='store_true')
    parser.add_argument('--resume_from_checkpoint', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--action', choices=['train', 'evaluate', 'test'], default='train')
    parser.add_argument('--all_chkpnts', action='store_true')
    parser.add_argument('--num_chkpnts', type=int, help='useful for debugging `--all_chkpnts`')
    parser.add_argument('--chkpnts', nargs='+')
    parser.add_argument('--eval_output')
    parser = add_processor_args(parser)
    parser = add_whisper_model_args(parser)
    parser = add_dataset_args(parser)
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
    predictions=trainer.predict(ds_split, metric_key_prefix=metric_key_prefix)
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
                os.path.join(args.output, 'checkpoint-*/')
    )
    chkpnts.sort(key=lambda s:int(s.removesuffix('/').split(sep='-')[-1]))
    if args.num_chkpnts:
        chkpnts=chkpnts[:args.num_chkpnts]
    elif args.chkpnts:
        chkpnts=[
            chkpnt for chkpnt in chkpnts
            if chkpnt.removesuffix('/').split(sep='-')[-1] in args.chkpnts
        ]
    eval_output_stem=args.eval_output or args.output
    os.makedirs(eval_output_stem, exist_ok=True)
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

        del chkpnt_model
        del data_collator
        del trainer
        del predictions
    csv_path=os.path.join(eval_output_stem, 'checkpoints-eval.csv')
    df=pd.DataFrame(data=metrics)
    df.to_csv(csv_path, index=False)

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