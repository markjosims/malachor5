# Finetune Whisper model on a new dataset using HF
# Large part of code taken from https://huggingface.co/blog/fine-tune-whisper on Sep 17 2024

from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional, Dict, Any
from transformers import Seq2SeqTrainingArguments
from tbparse import SummaryReader
import torch
import numpy as np
from jiwer import wer, cer
import pandas as pd
import os
import json
from glob import glob
from tqdm import tqdm
from dataset_utils import load_and_prepare_dataset, load_data_collator, DATASET_ARGS, TRAIN_DS_ARGS, EVAL_DS_ARGS
from tokenization_utils import LANG_TOKENS, LANG_TOKEN_IDS, normalize_multiling
from model_utils import WhisperTrainer, load_whisper_model_for_training_or_eval, set_generation_config, PROCESSOR_ARGS, MODEL_ARGS, prepare_trainer_for_peft
from argparse_utils import make_arggroup_from_argdict
from string_norm import get_remove_oov_char_funct, condense_tones
from datetime import datetime
from uuid import uuid4
from eval import get_metrics_by_language
from copy import deepcopy
import re
from logging_utils import make_experiment_json, get_checkpoint_num
from training_arg_utils import (
    TRAIN_PROG_ARGS,
    DEFAULT_TRAINER_HYPERPARAMS,
    get_hyperparam_argdict,
    LM_ARGS,
    EXTRA_OUTPUT_ARGS,
    LOSS_REGULARIZATION_HYPERPARAMS,
    PROMPT_TUNING_HYPERPARAMS,
    EVAL_ARGS,
)

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    argdicts = [
        (TRAIN_PROG_ARGS, 'prog_args'),
        (DATASET_ARGS, 'dataset_args'),
        (MODEL_ARGS, 'model_args'),
        (PROCESSOR_ARGS, 'processor_args'),
        (get_hyperparam_argdict(), 'trainer_hyperparams'),
        (LM_ARGS, 'lm_decoding'),
        (EXTRA_OUTPUT_ARGS, 'extra_output_args'),
        (LOSS_REGULARIZATION_HYPERPARAMS, 'loss_regularization'),
        (PROMPT_TUNING_HYPERPARAMS, 'prompt_tuning'),
        (EVAL_ARGS, 'eval'),
    ]
    for argdict, name in argdicts:
        make_arggroup_from_argdict(argdict, parser, name)
    return parser

# -------------------- #
# Misc trainer actions #
# -------------------- #

def calculate_fisher_matrix(args, trainer, model):
    fisher_matrix = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    dataloader = trainer.get_train_dataloader()
    for batch in tqdm(dataloader, total=len(dataloader), desc='Calculating gradient for each batch in dataset'):
        # normally popped during `WhisperTrainer.training_step()`
        # need to do manually since we're not using the `training_step()` function
        batch.pop('forced_decoder_ids', None)
        inputs = trainer._prepare_inputs(batch)
        with trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(model, inputs)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_matrix[name] += param.grad.pow(2).detach()
        model.zero_grad()
    fisher_matrix = {name: fisher / len(dataloader) for name, fisher in fisher_matrix.items()}
    fisher_matrix_path = getattr(
        args,
        'fisher_matrix_path',
        None,
    ) or os.path.join(args.output, args.dataset.split('/')[-1]+'_fisher.pt')
    torch.save(fisher_matrix, fisher_matrix_path)
    return fisher_matrix_path

def get_lid_probs(args, trainer, model):
    lid_probs = {
        lang: [] for lang in LANG_TOKENS
    }
    dataloader = trainer.get_train_dataloader()
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc='Calculating LID logits for each batch in dataset'):
            # normally popped during `WhisperTrainer.training_step()`
            # need to do manually since we're not using the `training_step()` function
            batch.pop('forced_decoder_ids', None)
            inputs = trainer._prepare_inputs(batch)
            lid_logits = trainer.get_lid_logits(input_features=inputs.input_features)
            batch_probs=lid_logits.softmax(dim=1)
            for lang, lang_obj in LANG_TOKENS.items():
                lid_probs[lang].extend(batch_probs[:,lang_obj['id']].detach().tolist())
            del lid_logits, batch_probs
    for lang in lid_probs:
        lid_probs[lang]=torch.tensor(lid_probs[lang])
    lid_logits_path = getattr(
        args,
        'lid_logits_path',
        None,
    ) or os.path.join(args.output, args.dataset.split('/')[-1]+'_lid_logits.pt')
    torch.save(lid_probs, lid_logits_path)
    return lid_logits_path



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
    if args.whisper_normalize:
        str_process_pipe.append(normalize_multiling)

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
        str_process_f=str_process_pipe,
        return_decoded=return_decoded,
        langs=args.langs_for_metrics,
    )
    return compute_metrics

def argmax_logits(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    Taken from https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
    on 25 Sep 2024
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

def compute_wer_cer(
        pred,
        tokenizer,
        str_process_f=None,
        return_decoded=False,
        langs=None,
    ):
    pred_ids = pred.predictions
    if type(pred_ids) is tuple:
        pred_ids = pred_ids[0]
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if langs:
        batch_wer = get_metrics_by_language(label_str, pred_str, 'wer', langs=langs, average=True)
        batch_cer = get_metrics_by_language(label_str, pred_str, 'cer', langs=langs, average=True)
        batch_metrics = {**batch_wer, **batch_cer}
    else:
        batch_wer = wer(label_str, pred_str)
        batch_cer = cer(label_str, pred_str)
        batch_metrics={
            "wer": batch_wer, 
            "cer": batch_cer,
        }
    if return_decoded:
        batch_metrics["labels"]=label_str
        batch_metrics["preds"]=pred_str

    if str_process_f:
        pred_str_processed=str_process_f(pred_str)
        label_str_processed=str_process_f(label_str)
        if langs:
            batch_wer_processed = get_metrics_by_language(label_str_processed, pred_str_processed, 'wer', langs=langs, average=True)
            batch_cer_processed = get_metrics_by_language(label_str_processed, pred_str_processed, 'cer', langs=langs, average=True)
            add_processed_suffix = lambda d:{k+'_processed':v for k,v in d.items()}
            batch_wer_processed = add_processed_suffix(batch_wer_processed)
            batch_cer_processed = add_processed_suffix(batch_cer_processed)
            batch_metrics.update(**batch_wer_processed)
            batch_metrics.update(**batch_cer_processed)
        else:
            batch_metrics['cer_processed']=cer(label_str_processed, pred_str_processed)
            batch_metrics['wer_processed']=wer(label_str_processed, pred_str_processed)
        if return_decoded:
            batch_metrics['preds_processed']=pred_str_processed

    return batch_metrics

def evaluate_dataset(args, ds_split, trainer, processor, save_results_to_disk=True):
    if type(ds_split) is dict:
        # evaluate on each dataset in dict
        predictions_dict = {}
        for ds_name, ds in tqdm(ds_split.items(), total=len(ds_split), desc='Evaluating datasets'):
            tqdm.write(f'Evaluating dataset {ds_name}...')
            ds_args = deepcopy(args)
            if args.eval_output:
                ds_args.eval_output = args.eval_output + '-' + ds_name
            else:
                ds_args.eval_output = os.path.join(
                    args.output, f'{ds_name}-{args.action}-predictions'
                )
            predictions_dict[ds_name]=evaluate_dataset(
                args=ds_args,
                ds_split=ds,
                trainer=trainer,
                processor=processor,
                save_results_to_disk=save_results_to_disk,
            )
        return predictions_dict
    metric_key_prefix = 'test' if args.action=='test' else 'eval'
    # change metrics to return labels
    trainer.compute_metrics=get_metrics(args, processor=processor, return_decoded=True)
    predictions=trainer.predict(ds_split, metric_key_prefix=metric_key_prefix)
    if not save_results_to_disk:
        return predictions
    labels=predictions.metrics[f'{metric_key_prefix}_labels']
    preds=predictions.metrics[f'{metric_key_prefix}_preds']
    preds_processed=predictions.metrics.get(f'{metric_key_prefix}_preds_processed', None)
    df=pd.DataFrame({'labels': labels, 'preds': preds})
    if not (preds_processed is None):
        df['preds_processed']=preds_processed
    df.to_csv(
        args.eval_output+'.csv' if args.eval_output
        else os.path.join(args.checkpoint or args.output, f'{args.action}-predictions.csv')
    )
    # predictions.metrics.pop(f'{metric_key_prefix}_labels')
    # predictions.metrics.pop(f'{metric_key_prefix}_preds')
    # predictions.metrics.pop(f'{metric_key_prefix}_preds_processed', None)

    torch.save(
        predictions,
        args.eval_output+'.pt' if args.eval_output
        else os.path.join(args.checkpoint or args.output, f'{args.action}-predictions.pt')
    )
    print(predictions.metrics)
    return predictions

def evaluate_all_checkpoints(args, ds, processor, training_args, compute_metrics):
    chkpnts=glob(
        os.path.join(args.output, 'checkpoint-*')
    )
    chkpnts.sort(key=lambda s:get_checkpoint_num(s))
    if args.eval_checkpoints != ['all']:
        chkpnts=[
            chkpnt for chkpnt in chkpnts
            if chkpnt.removesuffix('/').split(sep='-')[-1] in args.eval_checkpoints
        ]
    eval_output_stem=args.eval_output or args.output
    os.makedirs(eval_output_stem, exist_ok=True)
    metrics=[]

    # add execution metadata before looping through checkpoints
    # as for each checkpoint we call `make_experiment_json` to add the events
    # from that evaluation
    add_exec_json(args)

    for chkpnt in tqdm(chkpnts, desc='Evaluating checkpoints'):
        chkpnt=chkpnt.removesuffix('/')
        args.checkpoint=chkpnt
        chkpnt_basename=os.path.basename(chkpnt)
        args.eval_output=os.path.join(eval_output_stem, chkpnt_basename)
        tqdm.write(f"Loading {chkpnt}...")
        chkpnt_model = load_whisper_model_for_training_or_eval(args)
        chkpnt_model = set_generation_config(args, chkpnt_model, processor.tokenizer)
        data_collator=load_data_collator(chkpnt_model, processor)
        trainer = init_trainer(args, processor, training_args, compute_metrics, chkpnt_model, ds, data_collator)
        predictions=evaluate_dataset(args, ds['validation'], trainer, processor)
        make_experiment_json(args, predictions=predictions, add_execution_object=False)
        if type(predictions) is dict:
            # multiple datasets
            for ds_name, ds_preds in predictions.items():
                metrics.append(ds_preds.metrics)
                metrics[-1]['checkpoint']=chkpnt
                metrics[-1]['dataset']=ds_name
        else:
            # single dataset
            metrics.append(predictions.metrics)
            metrics[-1]['checkpoint']=chkpnt

        del chkpnt_model
        del data_collator
        del trainer
        del predictions
    df=pd.DataFrame(data=metrics)
    df=df.melt(id_vars='checkpoint', var_name='tag')
    df['step']=df['checkpoint'].apply(lambda s: re.match(r'.*checkpoint-([0-9]+)/?', s).groups()[0]).astype(int)
    csv_path=os.path.join(eval_output_stem, f"checkpoints-{'eval' if args.action=='evaluate' else 'test'}.csv")
    df.to_csv(csv_path, index=False)

def init_trainer(args, processor, training_args, compute_metrics, model, ds, data_collator):
    trainer = WhisperTrainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=processor.feature_extractor,
            preprocess_logits_for_metrics=argmax_logits if not args.predict_with_generate else None,
            mean_embed_path=args.mean_embed_path,
            embed_dist_lambda=args.embed_dist_lambda,
            embed_dist_type=args.embed_dist_type,
            lid_loss_alpha=args.lid_loss_alpha,
            fisher_matrix_path=args.fisher_matrix_path if args.action=='train' else None,
            lm_path=args.lm,
            lm_alpha=args.lm_alpha,
            lm_input=args.lm_input,
            lm_betas=args.lm_betas,
            tokenizer=processor.tokenizer,
            train_dataset=ds.get('train', None),
            eval_dataset=ds.get('validation', None),
        )
    
    return trainer



# ------------- #
# training args #
# ------------- #

def get_training_args(args):
    arg_dict={k: getattr(args, k) for k in DEFAULT_TRAINER_HYPERPARAMS.keys()}
    for k, v in arg_dict.items():
        # -1 passed to CLI indicates arg value should be None
        if v==-1:
            arg_dict[k]=None
    if args.peft_type=='LoRA':
        arg_dict['remove_unused_columns']=False
        arg_dict['label_names']=["labels"]
    if args.action!='train':
        arg_dict['evaluation_strategy']='no'

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        **arg_dict,
    )
    return training_args

# ---- #
# main #
# ---- #

def train(args: Namespace) -> int:
    # set environment variable for UUID and starting datetime
    timestr = str(datetime.now())
    uuid = str(uuid4())
    os.environ['STARTTIME']=timestr
    os.environ['UUID']=uuid

    print("Preparing dataset...")
    ds, processor = load_and_prepare_dataset(args)
    print("Defining metrics...")
    compute_metrics = get_metrics(args, processor)
    print("Defining training args...")
    training_args = get_training_args(args)

    if not args.eval_checkpoints:
        print("Loading model...")
        model = load_whisper_model_for_training_or_eval(args)
        print("Setting model generation config...")
        model = set_generation_config(args, model, processor.tokenizer)
        print("Building dataloader...")
        data_collator = load_data_collator(model, processor)
        print("Initializing trainer...")
        trainer = init_trainer(args, processor, training_args, compute_metrics, model, ds, data_collator)
        if args.peft_type:
            trainer = prepare_trainer_for_peft(args, trainer, processor)
    if args.action=='train':
        print("Training!")
        trainer.train(resume_from_checkpoint=args.checkpoint or args.resume_from_checkpoint)

        trainer.save_model(args.output)
        processor.save_pretrained(args.output)
        make_experiment_json(args, training_args=training_args)

        if args.eval_output:
            predictions = evaluate_dataset(args, ds['validation'], trainer, processor)
            make_experiment_json(args, predictions=predictions)
        
    elif args.action=='evaluate':
        if args.eval_checkpoints:
            evaluate_all_checkpoints(args, ds, processor, training_args, compute_metrics)
        else:
            predictions = evaluate_dataset(args, ds['validation'], trainer, processor)
            make_experiment_json(args, predictions=predictions)
    elif args.action=='calculate_fisher':
        calculate_fisher_matrix(args, trainer, model)
    elif args.action=='get_lid_probs':
        get_lid_probs(args, trainer, model)
    else:
        # args.action == 'test'
        predictions = evaluate_dataset(args, ds['test'], trainer, processor)
        make_experiment_json(args, predictions=predictions)
    return 0

def main(argv: Sequence[Optional[str]]=None) -> int:
    parser=init_parser()
    args=parser.parse_args(argv)
    return train(args)


if __name__ == '__main__':
    main()
