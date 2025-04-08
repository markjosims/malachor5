import os
import json
from glob import glob
from tbparse import SummaryReader
import pandas as pd
import sys
from typing import Dict, Optional, Any
from training_arg_utils import DEFAULT_TRAINER_HYPERPARAMS, LM_ARGS, GENERATE_ARGS

def get_checkpoint_num(checkpoint_dir):
    return int(checkpoint_dir.removesuffix('/').split(sep='-')[-1])

def get_step_of_checkpoint(checkpoint_dir) -> int:
    with open(os.path.join(checkpoint_dir, 'trainer_state.json')) as f:
        trainer_state = json.load(f)
    return trainer_state['global_step']

def get_latest_checkpoint(model_dir):
    checkpoints = glob(os.path.join(model_dir, 'checkpoint-*'))
    checkpoints.sort(key=get_checkpoint_num)
    return checkpoints[-1] if checkpoints else None

def get_global_step(args) -> int:
    if args.checkpoint:
        return get_step_of_checkpoint(args.checkpoint)
    latest_checkpoint = get_latest_checkpoint(args.output)
    if latest_checkpoint is None:
        return None
    return get_step_of_checkpoint(latest_checkpoint)

def get_latest_run_path(logdir: str) -> str:
    runs = glob(os.path.join(logdir, 'events.out.tfevents*'))
    runs.sort(key=os.path.getmtime)
    return runs[-1]

def get_run_df(training_args=None, predictions=None, args=None):
    if training_args is not None:
        run_path = get_latest_run_path(training_args.logging_dir)
        reader = SummaryReader(run_path)
        run_df = reader.scalars
        return run_df
    global_step = get_global_step(args)
    metric_list = []
    for k, v in predictions.metrics.items():
        metric_list.append({
            'tag': k,
            'value': v,
            'step': global_step
        })
    return pd.DataFrame(metric_list)

def merge_eval_or_test_data(data_list, incoming_data):
    """
    Try to find an existing data object where all keys match except for `events`.
    If none is found, append to end of list.
    """
    incoming_events = incoming_data['events']
    for tgt_data in data_list:
        for k, v in tgt_data.items():
            if k=='events':
                continue
            if incoming_data[k]!=v:
                break
        else: # no breaks; tgt_data is identical to incoming_data
            tgt_data['events'].extend(incoming_events)
            return data_list
    data_list.append(incoming_data)
    return data_list

def make_experiment_json(args, training_args=None, predictions=None, add_execution_object=True):
    # TODO to combine later executions back into main file we need to:
    # - [X] only write experiment-level keys if json doesnt already exist
    # - [X] append execution-specific data to `executions` list
    # - [ ] BEFORE TRAINING make sure train data is same as original run if applicable
    # - [X] pass predictions object from evaluatr/test calls
    # - [X] convert predictions object to df
    # - [X] get step for model being evaluated/tested
    # - [X] append train/eval/test events to respective list
    # - [X] merge val data objects if they have the same metadata
    # - [X] single execution object when evaluating multiple checkpoints
    json_path = os.path.join(args.output, 'experiment.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            exp_json = json.load(f)
    else:
        exp_json = {
            'experiment_name': os.path.basename(args.output),
            'experiment_path': args.output,
            'base_checkpoint': args.model,
            'executions': [],
            'train_data': {},
            'train_events': [],
            'test_data': [],
            'eval_data': [],
        }
    tb_df = get_run_df(training_args=training_args, predictions=predictions, args=args)
    if add_execution_object:
        exp_json = add_exec_json(args, exp_json=exp_json)
    if args.action == 'train':
        train_data, train_events = gather_train_dataset_metadata(args, tb_df)
        eval_data = gather_val_dataset_metadata(args, tb_df)
        if not exp_json['train_data']:
            exp_json['train_data'] = train_data
        exp_json['train_events'].extend(train_events)
        for eval_data_obj in eval_data:
            merge_eval_or_test_data(exp_json['eval_data'], eval_data_obj)
    elif args.action == 'evaluate':
        eval_data = gather_val_dataset_metadata(args, tb_df)
        for eval_data_obj in eval_data:
            merge_eval_or_test_data(exp_json['eval_data'], eval_data_obj)
    elif args.action == 'test':
        test_data = gather_test_dataset_metadata(args, tb_df)
        for test_data_obj in test_data:
            merge_eval_or_test_data(exp_json['test_data'], test_data_obj)
    else:
        raise NotImplementedError('`experiment.json` only saved for train, evaluate and test.')

    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(exp_json, f, indent=2, ensure_ascii=False)

def add_exec_json(args, exp_json=None) -> Optional[Dict[str, Any]]:
    """
    Add metadata for current execution to `experiment.json`.
    If passing `exp_json` object, append object with execution metadata to `executions`
    attr then return.
    If not passing, load `experiment.json`, append and save.
    """
    exec_json = {
        'uuid': os.environ['UUID'],
        'start_time': os.environ['STARTTIME'],
        'argv': ' '.join(sys.argv),
        'action': args.action,
    }
    for k in DEFAULT_TRAINER_HYPERPARAMS.keys():
        exec_json[k] = getattr(args, k)
    
    if exp_json is not None:
        exp_json['executions'].append(exec_json)
        return exp_json
    json_path = os.path.join(args.output, 'experiment.json')
    with open(json_path) as f:
        exp_json = json.load(f)
    exp_json['executions'].append(exec_json)
    with open(json_path, 'w') as f:
        json.dump(exp_json, f)
        

def gather_train_dataset_metadata(args, tb_df):
    split = 'train'
    split_data = gather_dataset_metadata(args, split, tb_df)
    # train events kept as separate key from train_data
    event_list = gather_events_for_ds(split, tb_df)
    return split_data, event_list

def gather_val_dataset_metadata(args, tb_df):
    split = 'eval'
    split_data = gather_dataset_metadata(args, split, tb_df)
    return split_data

def gather_test_dataset_metadata(args, tb_df):
    split = 'test'
    split_data = gather_dataset_metadata(args, split, tb_df)
    return split_data

def gather_dataset_metadata(args, split, tb_df):
    lang = '+'.join(args.language) if args.language else None
    ds_list = [args.dataset,] + (getattr(args, f'{split}_datasets', None) or [])
    ds_langs = [lang] + (getattr(args, f'{split}_dataset_languages', None) or [])
    if len(ds_list)>1 and len(ds_langs)==1:
        ds_langs*=len(ds_list)
    split_data = []
    for ds, lang in zip(ds_list, ds_langs):
        ds_basename = os.path.basename(ds)
        ds_metadata = {
                'dataset': ds_basename,
                'dataset_path': ds,
                'language': lang
        }
        # add dataset-specific arg values
        for k in ['num_records', 'train_data_pct', 'skip_idcs', 'skip_recordings']:
            # 'skip' and 'train' args only used for train dataset
            if (split!='train') and (('train' in k) or ('skip' in k)):
                continue
            v = getattr(args, k, None)
            if v is not None:
                ds_metadata[k]=v
        # for testing and validation, save events under dataset object
        if split!='train':
            # if only one dataset, name will not be included in tag
            if len(ds_list)==1:
                ds_name = None
            # if using multiple languages, will consist of dataset name + language name
            # event will be stored under 'dataset_stem' which,
            elif args.eval_dataset_languages:
                ds_name = f'{ds_basename}-{lang}'
            # only one language, dataset stem has no suffix
            else:
                ds_name = ds_basename
            # include eval/test-specific args
            for arg in [*GENERATE_ARGS, *LM_ARGS.keys()]:
                ds_metadata[arg]=getattr(args, arg, None)
            ds_metadata['events']=gather_events_for_ds(split, tb_df, ds_name=ds_name)
        split_data.append(ds_metadata)
    return split_data

def gather_events_for_ds(split, tb_df, ds_name=None):
    uuid = os.environ['UUID']
    timestr = os.environ['STARTTIME']
    split_mask = tb_df['tag'].str.contains(split, regex=False)
    if ds_name is not None:
        ds_mask = tb_df['tag'].str.contains(ds_name, regex=False)
        masked_df = tb_df[split_mask&ds_mask]
    else:
        masked_df = tb_df[split_mask]
    masked_df['uuid']=uuid
    masked_df['start_time']=timestr
    event_list = masked_df.to_dict(orient='records')
    return event_list