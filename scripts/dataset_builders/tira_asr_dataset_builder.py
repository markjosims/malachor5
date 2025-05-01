from typing import Sequence, Optional
from tira_elan_scraper import LIST_PATH
import os
import pandas as pd
from string import Template
from dataset_builder_utils import get_readable_duration, load_clips_to_ds
import json

import sys
sys.path.append('scripts')
from string_norm import unicode_normalize, has_diac, remove_punct, unicode_description, make_replacements
from lid_utils import is_en_word

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
The resulting dataset has $num_records records for $duration of speech, with each record
averaging $mean_duration.
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
    get_df_duration = lambda: get_readable_duration(df['duration'].sum())
    nan_str = f"- {len(df)} non-NaN transcriptions in dataset, {get_df_duration()}"
    print(nan_str)
    PREPROCESSING_STEPS.append(nan_str)

    # drop ungrammatical rows
    ungrammatical_mask = df['text'].str.contains('*', regex=False)
    df = df[~ungrammatical_mask]
    ungrammatical_str = f"- removed {int(ungrammatical_mask.sum())} ungrammatical rows, {len(df)} rows remaining, {get_df_duration()}"
    print(ungrammatical_str)
    PREPROCESSING_STEPS.append(ungrammatical_str)

    # basic string normalization
    print("String normalization...")
    df["text"] = df["text"].apply(unicode_normalize)
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].apply(remove_punct)
    nfkd_str = f"- applied NFKD unicode normalization to text, set to lowercase and removed punctuation"
    print(nfkd_str)
    PREPROCESSING_STEPS.append(nfkd_str)

    # skip all toneless entries
    print("Dropping rows with no tone diacritics")
    has_tone_mask = df["text"].apply(lambda s: has_diac(s, tone_only=True))
    prev_len = len(df)
    df = df[has_tone_mask]
    toneless_row_num = prev_len-len(df)
    toneless_str = f"- removed {toneless_row_num} rows with no tone marked, {len(df)} rows remaining, {get_df_duration()}"
    print(toneless_str)
    PREPROCESSING_STEPS.append(toneless_str)


    # remove all rows with English words
    print("Removing rows with English")
    en_words = set()
    tira_words = set()
    def detect_en_words(sentence):
        has_en_word = False
        for word in sentence.split():
            if is_en_word(word) and (len(word)>1):
                en_words.add(word)
                has_en_word=True
            else:
                tira_words.add(word)
        return has_en_word

    has_en_mask = df["text"].apply(detect_en_words)

    # save detected words for manual verification
    en_words_path = os.path.join(TIRA_ASR_CLIPS_DIR, "english_words.txt")
    tira_words_path = os.path.join(TIRA_ASR_CLIPS_DIR, "tira_words.txt")
    with open(en_words_path, 'w', encoding='utf8') as f:
        f.writelines(['\n'.join(en_words)])
    with open(tira_words_path, 'w', encoding='utf8') as f:
        f.writelines(['\n'.join(tira_words)])

    prev_len = len(df)
    df = df[~has_en_mask]
    en_row_num = prev_len-len(df)
    no_en_str = f"- removed {en_row_num} rows with English words, {len(df)} rows remaining, {get_df_duration()}"
    unique_words_str = "- saved all detected English words to $TIRA_ASR_CLIPS/english_words"+\
        "and Tira words to $TIRA_ASR_CLIPS/tira_words.txt"
    print(no_en_str)
    print(unique_words_str)
    PREPROCESSING_STEPS.append(no_en_str)
    PREPROCESSING_STEPS.append(unique_words_str)

    # remove tone words (e.g. HLL, LHL,...)
    print("Removing tone words from transcriptions...")
    is_tone_word = lambda s: all(c in 'hml' for c in s)
    has_tone_word = lambda s: any(is_tone_word(w) for w in s.split())
    remove_tone_word = lambda s: ' '.join(word for word in s.split() if not is_tone_word(word))
    has_tone_word_mask = df['text'].apply(has_tone_word)
    df['text'] = df['text'].apply(remove_tone_word)
    remove_tone_word_str = f"- removed tone words (e.g. HLL, LHL, LLHH) from transcription, {int(has_tone_word_mask.sum())} rows affected"
    print(remove_tone_word_str)
    PREPROCESSING_STEPS.append(remove_tone_word_str)

    # normalize IPA charset
    print("Normalizing IPA character set...")
    char_rep_json_path = os.path.join(TIRA_ASR_CLIPS_DIR, 'char_replacements.json')
    # # Uncomment to overwrite `char_rep_json`
    # unique_chars = set()
    # df['text'].apply(unique_chars.update)
    # rep_dict = {
    #     char: {
    #         'target': char,
    #         'comment': '',
    #         **unicode_description(char)
    #     } for char in unique_chars
    # }
    # with open(char_rep_json_path, 'w', encoding='utf8') as f:
    #     json.dump(rep_dict, f, ensure_ascii=True, indent=2)
    with open(char_rep_json_path, encoding='utf8') as f:
        rep_dict = json.load(f)
    rep_dict = {k: v['target'] for k, v in rep_dict.items()}
    normalize_ipa = lambda s: make_replacements(s, rep_dict)
    # apply twice since some diacritics may interfere with replacing digraphs
    df['text']=df['text'].apply(normalize_ipa)
    df['text']=df['text'].apply(normalize_ipa)

    print("Checking only expected chars are found in dataset...")
    expected_chars_basename = 'tira_asr_unique_chars.json'
    expected_chars_path = os.path.join('meta', expected_chars_basename)
    with open(expected_chars_path, encoding='utf8') as f:
        expected_ipa_chars = json.load(f)
    unexpected_chars = set()
    def find_unexpected_chars(sentence):
        found_unexpected_char = False
        for c in sentence:
            if c not in expected_ipa_chars:
                unexpected_chars.add(c)
                found_unexpected_char=True
        return found_unexpected_char

    unexpected_chars_mask = df['text'].apply(find_unexpected_chars)
    if (unexpected_chars_mask).sum()>0:
        raise ValueError("Found unexpected chars after normalizing IPA. Inspect 'text' col in dataframe.")
    expected_char_str = "- Checked that only expected IPA chars are found in dataset, "+\
        f"as defined by JSON file {expected_chars_basename}"
    print(expected_char_str)
    PREPROCESSING_STEPS.append(expected_char_str)

    print("Loading audio files into HuggingFace dataset...")
    ds = load_clips_to_ds(df, AUDIO_DIR)

    readme_header_str = README_HEADER.substitute(
        num_records=len(df),
        duration=get_df_duration(),
        mean_duration=get_readable_duration(df['duration'].mean()),
    )
    readme_out = os.path.join(TIRA_ASR_CLIPS_DIR, 'README.md')
    with open(readme_out, 'w', encoding='utf8') as f:
        f.write(readme_header_str+'\n')
        f.write('\n'.join(PREPROCESSING_STEPS))

if __name__ == '__main__':
    main()