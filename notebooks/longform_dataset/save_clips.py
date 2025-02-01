import pandas as pd
import os
import sys
from tqdm import tqdm
import numpy as np
import torchaudio
import torch
import argparse
sys.path.append('scripts')
from longform import load_and_resample
import shutil

tqdm.pandas()
balance_df_path = 'notebooks/longform_dataset/balance_df.csv'
elicitation_wav_dir = 'data/elicitation-wavs/wav'
tira_cs_dir = 'data/hf-datasets/tira_cs_balanced'
tira_eng_dir = 'data/hf-datasets/tira_eng_balanced'
tira_tira_dir = 'data/hf-datasets/tira_tira_balanced'
os.makedirs(os.path.join(tira_cs_dir, 'train'), exist_ok=True)

def load_clip(clipname):
    clippath = os.path.join(elicitation_wav_dir, clipname)
    return load_and_resample(clippath)

def concat_recordings_and_save(rows, i):
    clip_basename = str(i)+'.wav'
    clip_relpath = os.path.join('train', clip_basename)
    clip_path = os.path.join(tira_cs_dir, clip_relpath)
    clips = rows['clip_name'].apply(load_clip).tolist()
    clips = torch.concatenate(clips, dim=1)
    # clips = [snip_record(row) for _, row in rows.iterrows()]
    torchaudio.save(clip_path, clips, 16_000)
    return clip_relpath

def make_cs_labels(df):
    cs_df = df[df['lang_balanced_dataset']=='cs']
    cs_labels = []
    for i in tqdm(cs_df['asr_index'].unique().tolist()):
        if np.isnan(i):
            continue
        i=int(i)
        has_index = cs_df['asr_index']==i
        min_start = cs_df[has_index]['start'].min()
        max_end = cs_df[has_index]['end'].max()
        transcription = ' '.join([
            s.strip() for s in cs_df[has_index].sort_values('start')['transcription']
        ]).strip()
        filestem = cs_df[has_index]['filestem'].unique()
        clip_relpath = concat_recordings_and_save(cs_df[has_index], i)
        cs_labels.append({
            'asr_index': i,
            'start': min_start,
            'end': max_end,
            'transcription': transcription,
            'indices': cs_df[has_index].sort_values('start').index.tolist(),
            'duration': max_end-min_start,
            'split': 'train',
            'filestem': filestem,
            'file_name': clip_relpath,
        })
    return pd.DataFrame(cs_labels)

def make_monoling_ds(df, lang):
    labels = []
    if lang == 'tira':
        tgt_dir = tira_tira_dir
    else:
        tgt_dir = tira_eng_dir
    os.makedirs(os.path.join(tgt_dir, 'train'), exist_ok=True)
    lang_df = df[df['lang_balanced_dataset']==lang]
    lang_df['clip_name'].progress_apply(
        lambda clipname: shutil.move(
            os.path.join(elicitation_wav_dir, clipname),
            os.path.join(tgt_dir, 'train', os.path.basename(clipname))
        )
    )
    lang_df['file_name'] = 'train/'+lang_df['clip_name']
    return lang_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', '-l', choices=['tira', 'eng', 'cs'], required=True)
    args = parser.parse_args()
    df=pd.read_csv(balance_df_path)
    if args.language=='cs':
        cs_labels = make_cs_labels(df)
        cs_labels.to_csv(os.path.join(tira_cs_dir, 'metadata.csv'), index=False)
    elif args.language=='eng':
        eng_labels = make_monoling_ds(df, lang='eng')
        eng_labels.to_csv(os.path.join(tira_eng_dir, 'metadata.csv'), index=False)
    elif args.language=='tira':
        tira_labels = make_monoling_ds(df, lang='tira')
        tira_labels.to_csv(os.path.join(tira_tira_dir, 'metadata.csv'), index=False)