from typing import *
from tira_elan_scraper import LIST_PATH
from tira_asr_dataset_builder import perform_textnorm
import os
import pandas as pd
from string import Template

import sys
sys.path.append('scripts')

AUDIO_DIR = os.environ.get("TIRA_ELICITATION_WAVS")
TIRA_MORPH_DIR = os.environ.get("TIRA_MORPH")
VERSION = "0.1.0"

README_HEADER = Template(
"""
# tira_morph
Dataset of unique Tira sentences for purposes of training morphological segmentation.
Uses same textnorm steps as `tira_asr` with additional deduplication.
For now, dataset only contains Tira strings, no ground-truth morphological
analyses, glosses or translations. These will be introduced at a later stage.
Contains $num_records unique sentences for a total of $num_words words
($num_word_unique unique words) averaging $mean_sentence_len words per sentence.
"""
)
PREPROCESSING_STEPS = []

def main() -> int:
    df = pd.read_csv(LIST_PATH)
    print(len(df))
    
    df, _ = perform_textnorm(df, PREPROCESSING_STEPS)
    df = df.drop_duplicates("text")
    all_words = []
    df["text"].str.split().apply(all_words.extend)
    unique_words = set(all_words)

    # drop audio-related cols
    df=df.drop(["audio_basename", "start", "end", "duration"], axis=1)

    num_words = len(all_words)
    num_word_unique = len(unique_words)

    mean_sentence_len = df["text"].str.split().apply(len).mean()

    readme_header_str = README_HEADER.substitute(
        num_records=len(df),
        num_words=num_words,
        num_word_unique=num_word_unique,
        mean_sentence_len=round(mean_sentence_len, 2),
    )
    readme_out = os.path.join(TIRA_MORPH_DIR, 'README.md')
    os.makedirs(TIRA_MORPH_DIR, exist_ok=True)
    with open(readme_out, 'w', encoding='utf8') as f:
        f.write(readme_header_str+'\n')
        f.write('\n'.join(PREPROCESSING_STEPS))

    csv_out = os.path.join(TIRA_MORPH_DIR, 'tira-morph.csv')
    df.to_csv(csv_out, index_label='index')

if __name__ == '__main__':
    main()