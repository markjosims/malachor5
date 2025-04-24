from argparse import ArgumentParser
from typing import Sequence, Optional
from tira_elan_scraper import LIST_PATH
import os
import pandas as pd
from string import Template

import sys
sys.path.append('scripts')
from string_norm import unicode_normalize, strip_diacs, remove_punct

AUDIO_DIR = os.environ.get("TIRA_ELICITATION_WAVS")
TIRA_ASR_CLIPS_DIR = os.environ.get("TIRA_ASR_CLIPS")
TIRA_ASR_PYARROW_DIR = os.environ.get("TIRA_ASR_PYARROW")
VERSION = "0.1.0"

README_HEADER = Template(
"""
# tira_asr
Dataset of monolingual Tira generated from ELAN annotations of Tira elicitation sessions.
Transcriptions were taken from the `IPA Transcription` tier.
Noisy transcriptions were filtered out by various steps of preprocessing, described below.
Remaining transcriptions were also preprocessed with various steps of text normalization.
The resulting dataset has $num_records records for $duration hours of speech, with each record
averaging $mean_duration seconds.
"""
)
PREPROCESSING_STEPS = []

def main(argv: Optional[Sequence[str]]=None) -> int:
    df = pd.read_csv(LIST_PATH)
    print(len(df))
    
    # only interested in 'IPA Transcription', no other tiers
    print("Dropping non-transcription annotations...")
    ipa_mask = df['tier'] == 'IPA Transcription'
    df=df[ipa_mask]
    df=df.drop(columns=['tier'])
    print(len(df))

    # drop na rows
    df=df.dropna()
    nan_str = f"- {len(df)} non-NaN transcriptions in dataset"
    print(nan_str)
    PREPROCESSING_STEPS.append(nan_str)

    # performing NFKD unicode normalization
    df["text"] = df["text"].apply(unicode_normalize)
    nfkd_str = f"- applied NFKD unicode normalization to text"
    print(nfkd_str)
    PREPROCESSING_STEPS.append(nfkd_str)

    # skip all toneless entries
    print("Dropping rows with no tone diacritics")
    no_diac_mask = df["text"].apply(strip_diacs) == df["text"]
    prev_len = len(df)
    df = df[~no_diac_mask]
    toneless_rows = prev_len-len(df)
    toneless_str = f"- removed {toneless_rows} rows with no tone marked, {len(df)} rows remaining"
    print(toneless_str)
    PREPROCESSING_STEPS.append(toneless_str)

    tone_markers = {
        'grave': "\u0300",
        'macrn': "\u0304",
        'acute': "\u0301",
        'circm': "\u0302",
        'caron': "\u030C",
    }
    has_tone = lambda s: any(diac in s for diac in tone_markers.values())

    df_tone=df.copy()[df['text'].apply(has_tone)]
    print(len(df_tone), len(df), len(df)-len(df_tone))
    # for some reason the first step is missing 72 toneless rows!

if __name__ == '__main__':
    main()