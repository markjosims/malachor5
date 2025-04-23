import sys
sys.path.append('scripts')
from lid_utils import get_word_language

import pandas as pd
from typing import Dict, Any, Literal
from collections import defaultdict
from argparse import ArgumentParser

# --------------------- #
# audio loading helpers #
# --------------------- #


# ------------------------ #
# dataset metadata helpers #
# ------------------------ #

def get_words_per_language(df: pd.DataFrame, langs=None) -> Dict[str, int]:
    words_by_language = defaultdict(lambda:0)
    def update_words_per_lang(sentence):
        words = sentence.split()
        for word in words:
            language = get_word_language(word, langs)
            words_by_language[language]+=1
    df['transcription'].apply(update_words_per_lang)
    return words_by_language

def get_duration_by_columns(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    duration_by_col = {}
    pivot = df.pivot_table("duration", col)
    for index in pivot.index:
        duration_by_col[index]=pivot.at[index,'duration']
    return duration_by_col

# -------------- #
# script helpers #
# -------------- #

def init_dataset_builder_parser(list_type: Literal['list', 'timestamps'] = 'list') -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--audio', '-a', help="Directory of audio files to generate dataset from")
    if list_type == 'list':
        parser.add_argument('--list', '-l', help="Text file with list of basenames audio files to include (if each audio file is a single record)")
    else:
        parser.add_argument('--timestamps', '-t', help=".csv file with columns audio_basename,start,end,(split) indicating timestamps for each record (if records are sliced from a longer audio file)")
    parser.add_argument('--output', '-o', help="Directory path to save PyArrow dataset to")
    parser.add_argument('--version', '-v', help="Dataset version.")