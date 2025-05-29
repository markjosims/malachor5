from typing import *
from tira_elan_scraper import LIST_PATH
import os
import pandas as pd
from string import Template
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from argparse import ArgumentParser

import sys
sys.path.append('scripts')
from kws import get_sliding_window

VERSION = "0.1.0"

def get_windows(row, framelengths, frameshifts):
    audio = row['audio']['array']
    sr = row['audio']['sampling_rate']
    window_ds_list = []
    for framelength, frameshift in zip(framelengths, frameshifts):
        windows = get_sliding_window(
            audio,
            framelength_s=framelength,
            frameshift_s=frameshift,
            sample_rate=sr,
            return_timestamps=True,
        )
        window_ds = Dataset.from_list(windows)
        window_ds = window_ds.add_column("framelength", [framelength]*len(windows_ds))
        window_ds = window_ds.add_column("frameshift", [frameshift]*len(windows_ds))
        window_ds_list.append(window_ds)
    all_windows_ds = pd.concat(window_ds_list)
    all_windows_ds = all_windows_ds.add_column('level', ['sliding_window']*len(all_windows_ds))
    all_windows_ds = all_windows_ds.add_column('index', [row['index']]*len(all_windows_ds))
    all_windows_ds = all_windows_ds.add_column('clip_name', [row['clip_name']]*len(all_windows_ds))
    return {"window_ds": all_windows_ds}

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d")
    parser.add_argument("--framelengths", nargs="+", type=float, default=[0.020, 0.500])
    parser.add_argument("--frameshifts", nargs="+", type=float, default=[0.015, 0.400])
    args = parser.parse_args(argv)
    ds = load_from_disk(args.dataset)
    if type(ds) is DatasetDict:
        ds = ds['train']
    window_ds = ds.map(
        lambda row: get_windows(row, args.framelengths, args.frameshifts),
        remove_columns=ds.column_names
    )
    window_ds = concatenate_datasets(window_ds['window_ds'])
    breakpoint()
    return 0

if __name__ == '__main__':
    main()