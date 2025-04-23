import sys
sys.path.append('scripts')
from lid_utils import get_word_language

import pandas as pd
from typing import Dict, Any
from collections import defaultdict

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