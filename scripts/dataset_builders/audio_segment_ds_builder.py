from typing import *
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import os
import pandas as pd

import sys
sys.path.append('scripts')
from kws import get_sliding_window, textgrid_to_df

VERSION = "0.1.0"

def get_windows(row, framelengths, frameshifts, window_ds_list):
    audio = row['audio']['array']
    sr = row['audio']['sampling_rate']
    for framelength, frameshift in zip(framelengths, frameshifts):
        windows = get_sliding_window(
            audio,
            framelength_s=framelength,
            frameshift_s=frameshift,
            sample_rate=sr,
            return_timestamps=True,
        )
        window_ds = Dataset.from_list(windows)
        window_ds = window_ds.add_column("framelength", [framelength]*len(window_ds))
        window_ds = window_ds.add_column("frameshift", [frameshift]*len(window_ds))
        window_ds = window_ds.add_column('level', ['sliding_window']*len(window_ds))
        window_ds = window_ds.add_column('index', [row['index']]*len(window_ds))
        window_ds = window_ds.add_column('clip_name', [row['clip_name']]*len(window_ds))
        window_ds = window_ds.add_column('text', ['']*len(window_ds))
        window_ds_list.append(window_ds)

def get_phone_word_rows(row, tg_df, window_ds_list):
    index = row['index']
    # if index not in `tg_df`, MFA produced no output for this record
    if index not in tg_df.index:
        return
    clip_name = row['clip_name']
    audio = row['audio']['array']
    sr = row['audio']['sampling_rate']
    row_list = []
    for _, tg_row in tg_df.loc[index].iterrows():
        start_s=tg_row['start']
        end_s=tg_row['end']
        tier=tg_row['tier']
        start_i = int(start_s*sr)
        end_i = int(end_s*sr)
        clip = audio[start_i:end_i]
        row_list.append({
            "start_s": start_s,
            "end_s": end_s,
            "samples": clip,
            "framelength": 0,
            "frameshift": 0,
            "level": tier,
            "clip_name": clip_name,
            "index": index,
        })
    ds = Dataset.from_list(row_list)
    window_ds_list.append(ds)

def get_agg_tg_df(tg_paths):
    df_list = []
    for tg_path in tqdm(tg_paths):
        tg_df = textgrid_to_df(tg_path, words_only=False)
        tg_df['filename']=tg_path
        df_list.append(tg_df)
    df=pd.concat(df_list)
    df['index']=df['filename'].apply(os.path.basename).str.replace('.TextGrid', '').astype(int)
    df=df.set_index('index')
    return df

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d")
    parser.add_argument("--textgrid_dir")
    parser.add_argument("--framelengths", nargs="+", type=float, default=[0.020, 0.500])
    parser.add_argument("--frameshifts", nargs="+", type=float, default=[0.015, 0.400])
    args = parser.parse_args(argv)
    ds = load_from_disk(args.dataset)
    if type(ds) is DatasetDict:
        ds = ds['train']
    window_ds_list = []
    ds.map(
        lambda row: get_windows(row, args.framelengths, args.frameshifts, window_ds_list),
    )
    tg_paths = glob(os.path.join(args.textgrid_dir, '*.TextGrid'))
    tg_df=get_agg_tg_df(tg_paths)
    ds.map(
        lambda row: get_phone_word_rows(row, tg_df, window_ds_list)
    )

    window_ds = concatenate_datasets(window_ds_list)
    breakpoint()
    return 0

if __name__ == '__main__':
    main()