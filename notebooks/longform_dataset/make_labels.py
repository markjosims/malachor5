import pandas as pd
import os
import sys
from tqdm import tqdm
sys.path.append('scripts')

tqdm.pandas()
balance_df_path = 'notebooks/longform_dataset/balance_df.csv'
longform_dir_path = 'notebooks/longform_dataset/'
elicitation_wav_dir = 'data/elicitation-wavs/wav/'
tira_cs_dir = 'data/hf-datasets/tira_cs_balanced/'
os.makedirs(os.path.join(elicitation_wav_dir, 'labels'), exist_ok=True)

def save_label(row):
    label_path = os.path.join(elicitation_wav_dir, 'labels', f'{row.name}.txt')
    with open(label_path, 'w') as f:
        f.write(row['transcription'])
    return label_path

if __name__ == '__main__':
    df = pd.read_csv(balance_df_path, index_col='index')
    df['label_path']=df.progress_apply(save_label, axis=1)
    df.to_csv(balance_df_path)