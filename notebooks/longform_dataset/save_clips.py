import pandas as pd
import os
import sys
from tqdm import tqdm
import numpy as np
import torchaudio
import torch
from glob import glob
sys.path.append('scripts')
from longform import load_and_resample
import shutil
from datasets import load_dataset
import json

tqdm.pandas()
balance_df_path = 'notebooks/longform_dataset/balance_df.csv'
balance_data_dir = 'notebooks/longform_dataset/data'
clip_dir = 'data/elicitation-wavs/wav/clips'
hf_ds_dir = 'data/hf-datasets'
pyarrow_ds_dir = 'data/pyarrow-datasets'

def load_clip(i):
    clippath = os.path.join(clip_dir, str(i)+'.wav')
    return load_and_resample(clippath)

def concat_recordings(idcs):
    clips = [load_clip(i) for i in idcs]
    half_second_silence = torch.zeros(1,8_000)
    padded_clips = []
    for clip in clips:
        padded_clips.append(clip)
        padded_clips.append(half_second_silence)
    clips = torch.concatenate(padded_clips, dim=1)
    return clips

def sort_indices(indices, df):
    return sorted(indices, key=lambda i: df.loc[i, 'wav_source']+str(df.loc[i, 'start']))

if __name__ == '__main__':
    df = pd.read_csv(balance_df_path, index_col='index')
    mono_lists = glob(os.path.join(balance_data_dir, '*mono*.txt'))
    for list_file in mono_lists:
        with open(list_file) as f:
            train_list = f.read().splitlines()
        train_list = [int(i) for i in train_list]
        train_df = pd.DataFrame({
            'transcription': df.loc[train_list, 'transcription'],
        })
        ds_dirname = os.path.basename(list_file).replace('_indices.txt', '')
        ds_dirpath = os.path.join(hf_ds_dir, ds_dirname)
        os.makedirs(os.path.join(ds_dirpath, 'train'), exist_ok=True)
        def move_to_train_dir(i):
            clip_source = os.path.join(clip_dir, str(i)+'.wav')
            clip_relpath = os.path.join('train', str(i)+'.wav')
            clip_target = os.path.join(ds_dirpath, clip_relpath)
            shutil.move(clip_source, clip_target)
            return clip_relpath
        train_df['file_name']=[move_to_train_dir(i) for i in tqdm(train_list)]

        train_df.to_csv(os.path.join(ds_dirpath, 'metadata.csv'))
        ds = load_dataset('audiofolder', data_dir=ds_dirpath)
        pyarrow_path = os.path.join(pyarrow_ds_dir, ds_dirname)
        ds.save_to_disk(pyarrow_path)

    cs_lists = glob(os.path.join(balance_data_dir, 'asr_idx2cs*.json'))
    for list_file in cs_lists:
        with open(list_file) as f:
            idx_map = json.load(f)
        idx_map = {k:sort_indices(v, df) for k,v in idx_map.items()}
        list_file_basename = os.path.basename(list_file)
        ds_dirname = 'tira_' + list_file_basename.removeprefix('asr_idx2').removesuffix('_idcs.json')
        ds_dirpath = os.path.join(hf_ds_dir, ds_dirname)
        os.makedirs(os.path.join(ds_dirpath, 'train'), exist_ok=True)
        file_names = []
        transcriptions = []
        for asr_index, i_list in tqdm(idx_map.items()):
            clip = concat_recordings(i_list)
            clip_relpath = os.path.join('train', str(asr_index)+'.wav')
            clip_path = os.path.join(ds_dirpath, clip_relpath)
            torchaudio.save(clip_path, clip, 16_000)
            del clip
            transcription = ' '.join(df.loc[i_list, 'transcription'].str.strip().tolist())
            transcriptions.append(transcription)
            file_names.append(clip_relpath)
        train_df = pd.DataFrame({
            'file_name': file_names,
            'transcription': transcriptions,
        })
        train_df.to_csv(os.path.join(ds_dirpath, 'metadata.csv'))
        ds = load_dataset('audiofolder', data_dir=ds_dirpath)
        pyarrow_path = os.path.join(pyarrow_ds_dir, ds_dirname)
        ds.save_to_disk(pyarrow_path)

