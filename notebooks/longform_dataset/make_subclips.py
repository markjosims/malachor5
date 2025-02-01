import pandas as pd
import os
import sys
from tqdm import tqdm
import numpy as np
import torchaudio
import torch
sys.path.append('scripts')
from longform import load_and_resample

tqdm.pandas()
balance_df_path = 'notebooks/longform_dataset/balance_df.csv'
longform_dir_path = 'notebooks/longform_dataset/'
elicitation_wav_dir = 'data/elicitation-wavs/wav/'
tira_cs_dir = 'data/hf-datasets/tira_cs_balanced/'
os.makedirs(os.path.join(elicitation_wav_dir, 'clips'), exist_ok=True)
ms_to_frames = lambda ms: int(ms*16)

def clip_record(row, wav):
    start_frame = ms_to_frames(row['start'])
    end_frame = ms_to_frames(row['end'])
    clip = wav[:,start_frame:end_frame]
    clip_path = os.path.join(elicitation_wav_dir, 'clips', f'{row.name}.wav')
    torchaudio.save(clip_path, clip, 16_000)
    return clip_path

if __name__ == '__main__':
    df = pd.read_csv(balance_df_path)
    df['clip_name']=''
    for wavpath in tqdm(df['wav_source'].unique().tolist()):
        wavpath_local = os.path.join(
            elicitation_wav_dir,
            os.path.basename(wavpath)
        )
        wav_mask = df['wav_source']==wavpath
        if all(os.path.exists(
            os.path.join(
                elicitation_wav_dir, 'clips', f'{index}.wav'
            )) for index in df.loc[wav_mask].index
        ):
            df.loc[wav_mask, 'clip_name']='clips/'+df.loc[wav_mask].index.astype(str)+'.wav'
            continue
        wav = load_and_resample(wavpath_local)
        df.loc[wav_mask, 'clip_name']=df.loc[wav_mask].progress_apply(clip_record, wav=wav, axis=1)
    breakpoint()
    df.to_csv(balance_df_path)