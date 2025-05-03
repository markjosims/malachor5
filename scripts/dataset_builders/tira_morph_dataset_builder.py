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
Uses same textnorm steps as `tira_asr`. For now, dataset only contains sentence
transcriptions and, where applicable, morphological decompositions. No glosses or
transcriptions are included for now. Contains $num_sentences unique sentences for
a total of $num_words words ($num_word_unique unique words) averaging
$mean_sentence_len words per sentence. Of these, $num_analyses sentences have
morphological decompositions, for $num_word_analyzed unique analyzed words and
$num_morphs unique morphemes.
"""
)
PREPROCESSING_STEPS = []

def main() -> int:
    df = pd.read_csv(LIST_PATH)
    print(len(df))
    print("Dropping NaN...")
    df = df.dropna()
    
    transcription_mask = df['tier'] == 'IPA Transcription'
    morph_mask = df['tier'] == 'Word'

    PREPROCESSING_STEPS.append("## Preprocessing sentence transcriptions:")
    sentence_df, _ = perform_textnorm(df[transcription_mask], PREPROCESSING_STEPS)

    all_words = []
    sentence_df["text"].str.split().apply(all_words.extend)
    unique_words = set(all_words)
    num_words = len(all_words)
    num_word_unique = len(unique_words)
    mean_sentence_len = sentence_df["text"].str.split().apply(len).mean()

    PREPROCESSING_STEPS.append("## Preprocessing morphological analyses:")
    morph_df, _ = perform_textnorm(df[morph_mask], PREPROCESSING_STEPS, keep_punct='-')
    all_words_analyzed = []
    morph_df["text"].str.split().apply(all_words_analyzed.extend)
    unique_words_analyzed = set(all_words_analyzed)
    num_word_analyzed = len(unique_words_analyzed)
    all_morphs = []
    pd.Series(all_words_analyzed).str.split('-').apply(all_morphs.extend)
    num_morphs = len(all_morphs)

    readme_header_str = README_HEADER.substitute(
        num_sentences=len(sentence_df),
        num_words=num_words,
        num_word_unique=num_word_unique,
        mean_sentence_len=round(mean_sentence_len, 2),
        num_analyses=len(morph_df),
        num_word_analyzed=num_word_analyzed,
        num_morphs=num_morphs,
    )
    readme_out = os.path.join(TIRA_MORPH_DIR, 'README.md')
    os.makedirs(TIRA_MORPH_DIR, exist_ok=True)
    with open(readme_out, 'w', encoding='utf8') as f:
        f.write(readme_header_str+'\n')
        f.write('\n'.join(PREPROCESSING_STEPS))


    sentence_out = os.path.join(TIRA_MORPH_DIR, 'tira-sentences.txt')
    with open(sentence_out, mode='w', encoding='utf8') as f:
        f.write('\n'.join(sentence_df['text'].tolist()))

    wordlist_out = os.path.join(TIRA_MORPH_DIR, 'tira-wordlist.txt')
    word_counts = pd.Series(all_words).value_counts()
    with open(wordlist_out, mode='w', encoding='utf8') as f:
        for word, count in word_counts.items():
            f.write(f"{count} {word}\n")
    morph_out = os.path.join(TIRA_MORPH_DIR, 'tira-segmentations.txt')
    with open(morph_out, mode='w', encoding='utf8') as f:
        for analysis in all_words_analyzed:
            morphs = analysis.split('-')
            f.write(f"{''.join(morphs)} {' '.join(morphs)}\n")
if __name__ == '__main__':
    main()