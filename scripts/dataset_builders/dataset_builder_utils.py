import sys
sys.path.append('scripts')
from lid_utils import get_word_language
from longform import load_and_resample, SAMPLE_RATE

import os
import pandas as pd
from typing import Dict, Any, Optional, List, Literal
from collections import defaultdict
import torchaudio
import torch
from datasets import Dataset, Audio, load_dataset
from tempfile import TemporaryDirectory
from tqdm import tqdm
tqdm.pandas()

# --------------------- #
# audio loading helpers #
# --------------------- #

def load_clips_to_ds(
        df: pd.DataFrame,
        audio_dir: str,
) -> Dataset:
    """
    `df` is a dataframe with columns 'audio_basename', 'start', 'end'
    `audio_dir` is a dirpath to load long audio from, `clip_dir` is a dirpath to save
    clips for individual records to. Converts `df` to a HuggingFace Dataset
    and loads individual clips indicated by rows in `df` into `audio` column
    """
    with TemporaryDirectory() as temp_dir:
        save_clips(df, audio_dir, temp_dir)
        df.to_csv(os.path.join(temp_dir, 'clips.csv'), index=False)
        ds = load_dataset("audiofolder", data_dir=temp_dir)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return ds

def save_clips(
        df: pd.DataFrame,
        audio_dir: str,
        clip_dir: str,
    ) -> pd.DataFrame:
    """
    `df` is a dataframe with columns 'audio_basename', 'start', 'end'
    `audio_dir` is a dirpath to load long audio from, `clip_dir` is a dirpath to save
    clips for individual records to.
    Loads individual clips indicated by rows in `df` and saves as .wav files in `clip_dir`.
    Adds 'file_name' column to `df` containing clip paths (this colname chosen
    for compatability with HuggingFace audio folder datasets).
    """
    df['file_name'] = ''
    for audio_basename in tqdm(df['audio_basename'].unique()):
        audio_mask = df['audio_basename'] == audio_basename
        audio_path = os.path.join(audio_dir, audio_basename)
        wav = load_and_resample(audio_path)
        df.loc[audio_mask, 'file_name'] = df.loc[audio_mask].progress_apply(
            lambda row: save_clip(wav, row, clip_dir),
            axis=1,
        )
    return df

def save_clip(wav: torch.Tensor, row: Dict[str, Any], clip_dir: str):
    """
    Calls `get_clip` then saves resulting wav to disk in `clip_dir` using row index as a basename. 
    """
    start_ms = row['start']
    end_ms = row['end']
    clip = get_clip(wav, start_ms, end_ms)

    basename = row['audio_basename']
    i = row.name
    clip_basename = f'{basename}_{i}.wav'
    clip_path = os.path.join(clip_dir, clip_basename)
    torchaudio.save(clip_path, clip, SAMPLE_RATE)
    return clip_path


def get_clip(wav: torch.Tensor, start_ms: int, end_ms: int):
    """
    `wav` is a 2d tensor of audio samples, `start_ms` and `end_ms`
    are start and end times in milliseconds.
    Returns a slice of `wav` corresponding to the start and end timestamps.
    """
    samples_per_ms = SAMPLE_RATE/1_000
    start_samples = int(start_ms*samples_per_ms)
    end_samples = int(end_ms*samples_per_ms)
    return wav[:, start_samples:end_samples]

# ------------------------ #
# dataset metadata helpers #
# ------------------------ #

def get_words_per_language(df: pd.DataFrame, langs: Optional[List[str]]=None) -> Dict[str, int]:
    """
    `df` is a DataFrame containing the column 'transcription'.
    Returns a dict of shape {'$lang': $num_words}
    `langs` is an optional list of language strs to restrict the languages considered when counting.
    """
    words_by_language = defaultdict(lambda:0)
    def update_words_per_lang(sentence):
        words = sentence.split()
        for word in words:
            language = get_word_language(word, langs)
            words_by_language[language]+=1
    df['transcription'].apply(update_words_per_lang)
    return words_by_language

def get_duration_by_column(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """
    `df` is a DataFrame containing the column `duration`, and `col` is a str
    indicating the column to pivot on. Returns a dict of shape {'$col_value': $duration_ms}
    """
    duration_by_col = {}
    pivot = df.pivot_table("duration", col, aggfunc='sum')
    for index in pivot.index:
        duration_by_col[index]=pivot.at[index,'duration']
    return duration_by_col

def get_readable_duration(duration, time_unit: Literal['s', 'ms']='ms') -> str:
    if time_unit=='ms':
        total_s = duration/1_000
    else: # time_unit=='s'
        total_s = duration
    if total_s < 60:
        return f"{total_s:.2f} seconds"
    total_m = total_s//60
    remain_s = int(total_s%60)
    if total_m < 60:
        return f"{total_m:02d} min {remain_s:02d} sec"
    total_h = total_m//60
    remain_m = int(total_m%60)
    return f"{total_h} hr {remain_m:02d} min {remain_s:02d} sec"
    
def get_df_duration(df, time_unit: Literal['s', 'ms']='ms', agg: Literal['sum', 'mean']='sum') -> str:
    if agg=='mean':
        return get_readable_duration(df['duration'].mean(), time_unit=time_unit)
    return get_readable_duration(df['duration'].sum(), time_unit=time_unit)