from glob import glob
from tbparse import SummaryReader
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Sequence, Optional
import re

# ---------------- #
# filepath helpers #
# ---------------- #

def get_latest_run(run_dir: str) -> str:
    run_files = glob(os.path.join(run_dir, 'runs', '*'))
    run_date_strs = [os.path.basename(run_file)[:14] for run_file in run_files]
    run_file_dates = [datetime.strptime(run_date, '%b%d_%H-%M-%S') for run_date in run_date_strs]
    run_file_tuples = list(
        zip(run_files, run_file_dates)
    )
    run_file_tuples.sort(
        key=lambda t:t[1],
        reverse=True
    )
    return run_file_tuples[0][0]

# ----------------- #
# dataframe helpers #
# ----------------- #

def get_runs_df(run_dirs: Sequence[str]) -> pd.DataFrame:
    df_list = []
    for run_dir in tqdm(run_dirs):
        run_path = get_latest_run(run_dir)
        run_name = os.path.basename(run_dir)
        reader = SummaryReader(run_path)
        run_df = reader.scalars
        run_df['experiment_name'] = run_name
        df_list.append(run_df)
    return pd.concat(df_list)


def add_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    # lid loss alpha col
    get_lid_loss_alpha = lambda s: float(re.search(r'lid-alpha-([\d.]+)').groups()[0]) if 'lid-alpha-' in s else None
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
    get_regdist_lmd = lambda s: float(re.search(r'regdist-lmd-([\d.]+)').groups()[0]) if 'regdist-lmd-' in s else None
    df['distance_regularization_lambda'] = df['experiment_name'].apply(get_regdist_lmd)
    return df

# ---- #
# main #
# ---- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--globstr', '-g', nargs='+')
    parser.add_argument('--output', '-o')

    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)

    run_dirs = []
    print("Globbing runs...")
    for globstr in args.globstr:
        run_dirs.extend(glob(globstr))
    print(f"\tFound {len(run_dirs)} runs.")

    df = get_runs_df(run_dirs)
    print("Adding metadata from experiment names...")
    cols_orig = df.colnames
    df = add_df_columns(df)
    new_cols = [col for col in df.colnames if col not in cols_orig]
    print(f"\tAdded columns: {new_cols}")

    print(f"Saving to {args.output}...")
    df.to_csv(args.output, index=False)
    print("\tDone!")
    return 0

if __name__ == '__main__':
    main()