from glob import glob
from tbparse import SummaryReader
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Sequence, Optional, List, Tuple
import re
import torch

# ---------------- #
# filepath helpers #
# ---------------- #

def get_runs_with_date(run_dir: str) -> List[Tuple[str,str]]:
    run_files = glob(os.path.join(run_dir, 'runs', '*'))
    run_date_strs = [os.path.basename(run_file)[:14] for run_file in run_files]
    run_file_dates = [datetime.strptime(run_date, '%b%d_%H-%M-%S') for run_date in run_date_strs]
    run_file_tuples = list(
        zip(run_files, run_file_dates)
    )
    if not run_file_tuples:
        print(f"No runs found in {run_dir}.")
        return
    run_file_tuples.sort(
        key=lambda t:t[1],
        reverse=True
    )
    return run_file_tuples

def get_checkpoint_evals(run_dirs: Sequence[str]) -> List[str]:
    return sum(
        [
            glob(os.path.join(run_dir, '*', 'checkpoints-eval.csv'))
            for run_dir in run_dirs
        ],
        start=[]
    )

def get_test_predictions(run_dirs: Sequence[str]) -> List[str]:
    return sum(
        [
            glob(os.path.join(run_dir, '*', 'test-predictions.pt'))
            for run_dir in run_dirs
        ],
        start=[]
    )

# ----------------- #
# dataframe helpers #
# ----------------- #

def get_runs_df(run_dirs: Sequence[str], all_run_dates=False) -> pd.DataFrame:
    df_list = []
    for run_dir in tqdm(run_dirs):
        run_tuples = get_runs_with_date(run_dir)
        if not run_tuples:
            continue
        run_name = os.path.basename(run_dir.removesuffix('/'))
        exp_df_list = []
        for run_path, run_date in run_tuples:
            reader = SummaryReader(run_path)
            run_df = reader.scalars
            if len(run_df)==0:
                continue
            run_df['experiment_name'] = run_name
            run_df['date']=run_date
            exp_df_list.append(run_df)
        exp_df = pd.concat(exp_df_list)
        if not all_run_dates:
            exp_df = latest_run_per_event(exp_df)
        df_list.append(exp_df)
    df = pd.concat(df_list)
    return df

def get_csv_df(csv_list: Sequence[str]) -> pd.DataFrame:
    df_list = []
    for csv in csv_list:
        csv_df = pd.read_csv(csv)
        dirlist = os.path.normpath(csv).split(os.sep)
        model = dirlist[-3]
        csv_name = dirlist[-2]
        csv_relpath = os.path.join(model, csv_name)
        csv_df['csv_name']=csv_relpath
        csv_df['experiment_name']=model.removesuffix('/')
        df_list.append(csv_df)
    df = pd.concat(df_list)
    return df

def get_pt_df(pt_list: Sequence[str]) -> pd.DataFrame:
    df_list = []
    for pt_file in pt_list:
        predictions = torch.load(pt_file)
        dirlist = os.path.normpath(pt_file).split(os.sep)
        model = dirlist[-3]
        checkpoint = dirlist[-2]
        pt_relpath = os.path.join(model,checkpoint)
        rows = [{'tag':k, 'value':v} for k,v in predictions.metrics.items()]
        pt_df = pd.DataFrame(rows)
        pt_df['experiment_name']=model
        pt_df['step']=int(re.match(r'checkpoint-(\d+)', checkpoint).groups()[0])
        pt_df['preds_name']=pt_relpath
        df_list.append(pt_df)
    return pd.concat(df_list)

def latest_run_per_event(df: pd.DataFrame):
    df=df.reset_index()
    latest_idcs = []
    for tag in df['tag'].unique():
        tag_mask = df['tag']==tag
        for step in df['step'].unique():
            step_mask = df['step']==step
            step_tag_df = df[step_mask&tag_mask]
            if len(step_tag_df)==0:
                continue
            step_tag_df = step_tag_df.sort_values('date', ascending=False)
            latest_date = step_tag_df.iloc[0].name.item()
            latest_idcs.append(latest_date)
    return df.loc[latest_idcs]

def add_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    # lid loss alpha col
    get_lid_loss_alpha = lambda s: float(re.search(r'lid-alpha-([\d.]+)', s).groups()[0]) if 'lid-alpha-' in s else None
    df['lid_loss_alpha']=df['experiment_name'].apply(get_lid_loss_alpha)

    df['LoRA'] = df['experiment_name'].str.contains('LoRA')

    # lang prompt col
    get_lang_prompt = lambda s: 'swahili' if 'swahili' in s else\
        'croatian' if 'croatian' in s else\
        'yoruba' if 'yoruba' in s else\
        'LID' if 'LID' in s else\
        'softprompt' if 'softprompt' in s else\
        -1
    df['lang_prompt'] = df['experiment_name'].apply(get_lang_prompt)

    # lang token finetuning cols
    df['train_lang_token'] = df['experiment_name'].str.contains('lang-token')
    df['embedding_distance_regularization_type'] = None
    df.loc[
        df['experiment_name'].str.contains('euc'),
        'embedding_distance_regularization_type'
    ] = 'euc'
    df.loc[
        df['experiment_name'].str.contains('cosine'),
        'embedding_distance_regularization_type'
    ] = 'cosine'
    get_regdist_lmd = lambda s: float(re.search(r'regdist-lmd-([\d.]+)', s).groups()[0]) if 'regdist-lmd-' in s else None
    df['distance_regularization_lambda'] = df['experiment_name'].apply(get_regdist_lmd)
    get_ewc_lmd = lambda s: float(re.search(r'ewc-lambda-([\d.]+)', s).groups()[0]) if 'ewc-lambda-' in s else None
    df['ewc_lambda'] = df['experiment_name'].apply(get_ewc_lmd)

    get_lm_beta = lambda s: float(re.search(r'beta-([\d.]+)', s).groups()[0]) if 'beta-' in s else None
    get_lm_alpha = lambda s: float(re.search(r'lm-alpha-([\d.]+)', s).groups()[0]) if 'lm-alpha-' in s else None
    get_beams = lambda s: float(re.search(r'beam-([\d.]+)', s).groups()[0]) if 'beam-' in s else None
    df['csv_name']=df['csv_name'].fillna('')
    df['lm_beta']=df['csv_name'].apply(get_lm_beta)
    df['lm_alpha']=df['csv_name'].apply(get_lm_alpha)
    df['beam']=df['csv_name'].apply(get_beams)


    df['epoch']=0
    epoch_mask = df['tag'].str.contains('epoch')
    for i, row in df[epoch_mask].iterrows():
        step_mask = df['step']==row['step']
        exp_mask = df['experiment_name']==row['experiment_name']
        df.loc[step_mask & exp_mask, 'epoch'] = int(row['value'])
    return df

# ---- #
# main #
# ---- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--globstr', '-g', nargs='+')
    parser.add_argument('--output', '-o')
    parser.add_argument('--all_run_dates', action='store_true')

    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)

    run_dirs = []
    print("Globbing runs...")
    for globstr in args.globstr:
        run_dirs.extend(glob(globstr))
    # remove any filepaths that aren't directories
    run_dirs = [run_dir for run_dir in run_dirs if os.path.isdir(run_dir)]
    print(f"\tFound {len(run_dirs)} runs.")

    runs_df = get_runs_df(run_dirs)
    csv_list = get_checkpoint_evals(run_dirs)
    print(f"\tFound {len(csv_list)} evaluation datafiles.")
    if csv_list:
        csv_df = get_csv_df(csv_list)
        df=pd.concat([runs_df,csv_df])
    pt_list = get_test_predictions(run_dirs)
    print(f"\tFound {len(pt_list)} test prediction files...")
    if pt_list:
        pt_df = get_pt_df(pt_list)
        df = pd.concat([runs_df,pt_df])


    print("Adding metadata from experiment names...")
    cols_orig = df.columns
    df = add_df_columns(df)
    new_cols = [col for col in df.columns if col not in cols_orig]
    print(f"\tAdded columns: {new_cols}")

    print(f"Dataframe has {len(df)} rows from {len(df['experiment_name'].unique())} experiments")

    print(f"Saving to {args.output}...")
    df.to_csv(args.output, index=False)
    print("\tDone!")
    return 0

if __name__ == '__main__':
    main()