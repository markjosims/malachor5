from typing import *
from tira_elan_scraper import LIST_PATH
import os
import pandas as pd
from string import Template
from dataset_builder_utils import get_readable_duration, load_clips_to_ds, get_df_duration, overwrite_dataset, save_row_and_label
import json
from datasets import load_from_disk, Dataset, DatasetDict
import torch
from glob import glob

import sys
sys.path.append('scripts')
from longform import load_vad_pipeline, perform_vad
from string_norm import unicode_normalize, has_diac, remove_punct, unicode_description, make_replacements
from lid_utils import is_en_word
from kws import embed_speech, embed_text, dataloader
from clap.encoders import SpeechEncoder, PhoneEncoder
from string_norm import tira2mfa, condense_tones

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

def perform_textnorm(
        df: pd.DataFrame,
        preproc_steps: List[str],
        norm_col: str = 'text',
        keep_punct: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:

    # drop ungrammatical rows
    ungrammatical_mask = df[norm_col].str.contains('*', regex=False)
    df = df[~ungrammatical_mask]
    ungrammatical_str = f"- removed {int(ungrammatical_mask.sum())} ungrammatical rows, "+\
        f"{len(df)} rows remaining, {get_df_duration(df)}"
    print(ungrammatical_str)
    preproc_steps.append(ungrammatical_str)

    # basic string normalization
    print("String normalization...")
    df[norm_col] = df[norm_col].apply(unicode_normalize)
    df[norm_col] = df[norm_col].str.lower()
    df[norm_col] = df[norm_col].apply(lambda s: remove_punct(s, keep=keep_punct))
    nfkd_str = f"- applied NFKD unicode normalization to text, set to lowercase and removed punctuation"
    print(nfkd_str)
    preproc_steps.append(nfkd_str)

    # skip all toneless entries
    print("Dropping rows with no tone diacritics")
    has_tone_mask = df[norm_col].apply(lambda s: has_diac(s, tone_only=True))
    prev_len = len(df)
    df = df[has_tone_mask]
    toneless_row_num = prev_len-len(df)
    toneless_str = f"- removed {toneless_row_num} rows with no tone marked, {len(df)} rows remaining, {get_df_duration(df)}"
    print(toneless_str)
    preproc_steps.append(toneless_str)


    # remove all rows with English words
    print("Removing rows with English")
    en_words = set()
    tira_words = set()
    def detect_en_words(sentence):
        has_en_word = False
        for word in sentence.split():
            if is_en_word(word) and (len(word)>1) or word in ['downstep']:
                en_words.add(word)
                has_en_word=True
            else:
                tira_words.add(word)
        return has_en_word

    has_en_mask = df[norm_col].apply(detect_en_words)

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
    no_en_str = f"- removed {en_row_num} rows with English words, {len(df)} rows remaining, {get_df_duration(df)}"
    unique_words_str = "- saved all detected English words to $TIRA_ASR_CLIPS/english_words "+\
        "and Tira words to $TIRA_ASR_CLIPS/tira_words.txt"
    print(no_en_str)
    print(unique_words_str)
    preproc_steps.append(no_en_str)
    preproc_steps.append(unique_words_str)

    # remove tone words (e.g. HLL, LHL,...)
    print("Removing tone words from transcriptions...")
    is_tone_word = lambda s: all(c in 'hml' for c in s)
    has_tone_word = lambda s: any(is_tone_word(w) for w in s.split())
    remove_tone_word = lambda s: ' '.join(word for word in s.split() if not is_tone_word(word))
    has_tone_word_mask = df[norm_col].apply(has_tone_word)
    df[norm_col] = df[norm_col].apply(remove_tone_word)
    remove_tone_word_str = f"- removed tone words (e.g. HLL, LHL, LLHH) from transcription, {int(has_tone_word_mask.sum())} rows affected"
    print(remove_tone_word_str)
    preproc_steps.append(remove_tone_word_str)

    # normalize IPA charset
    print("Normalizing IPA character set...")
    char_rep_json_path = os.path.join(TIRA_ASR_CLIPS_DIR, 'char_replacements.json')
    # # Uncomment to overwrite `char_rep_json`
    # unique_chars = set()
    # df[norm_col].apply(unique_chars.update)
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
    df[norm_col]=df[norm_col].apply(normalize_ipa)
    df[norm_col]=df[norm_col].apply(normalize_ipa)

    print("Checking only expected chars are found in dataset...")
    expected_chars_basename = 'tira_asr_unique_chars.json'
    expected_chars_path = os.path.join('meta', expected_chars_basename)
    with open(expected_chars_path, encoding='utf8') as f:
        expected_ipa_chars = json.load(f)
    if keep_punct:
        expected_ipa_chars.extend([p for p in keep_punct])
    unexpected_chars = set()
    def find_unexpected_chars(sentence):
        found_unexpected_char = False
        for c in sentence:
            if c not in expected_ipa_chars:
                unexpected_chars.add(c)
                found_unexpected_char=True
        return found_unexpected_char

    unexpected_chars_mask = df[norm_col].apply(find_unexpected_chars)
    if (unexpected_chars_mask).sum()>0:
        raise ValueError(f"Found unexpected chars after normalizing IPA. Inspect {norm_col} col in dataframe.")
    expected_char_str = "- Checked that only expected IPA chars are found in dataset, "+\
        f"as defined by JSON file {expected_chars_basename}"
    print(expected_char_str)
    preproc_steps.append(expected_char_str)

    no_stacked_tones = df['text'].apply(condense_tones)
    stacked_tone_mask = df['text']!=no_stacked_tones
    duplicate_tone_str = f"- Found {stacked_tone_mask.sum()} rows where on or more words had several consecutive tone markers. "+\
        "Only keeping first tone marker."
    df['text']=no_stacked_tones
    print(duplicate_tone_str)
    preproc_steps.append(duplicate_tone_str)

    return df, preproc_steps

def main() -> int:
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
    nan_str = f"- {len(df)} non-NaN transcriptions in dataset, {get_df_duration(df)}"
    print(nan_str)
    PREPROCESSING_STEPS.append(nan_str)
    
    df, _ = perform_textnorm(df, PREPROCESSING_STEPS)

    print("Loading audio files into HuggingFace dataset...")

    unproc_ds_path = os.path.join(TIRA_ASR_CLIPS_DIR, 'unprocessed_audio_ds')

    try:
        hf_ds = load_from_disk(unproc_ds_path)
    except FileNotFoundError:
        hf_ds = load_clips_to_ds(df, AUDIO_DIR, ds_dir=unproc_ds_path)
    if type(hf_ds) is DatasetDict:
        hf_ds = hf_ds['train']
    unproc_audio_str = "- saved HF dataset with remaining audio records to 'unprocessed_audio_ds/' in $TIRA_ASR_CLIPS_DIR"
    PREPROCESSING_STEPS.append(unproc_audio_str)
    print(unproc_audio_str)

    col_added = False

    # hf_ds = hf_ds.select(range(10)) # uncomment for debugging
    if 'duration' not in hf_ds.column_names:
        duration = [row['audio']['array'].shape[-1] / row['audio']['sampling_rate'] for row in hf_ds]
        hf_ds = hf_ds.add_column('duration', duration)
        col_added = True

    if 'vad_chunks' not in hf_ds.column_names:
        vad_pipe = load_vad_pipeline()
        hf_ds = hf_ds.map(lambda row: perform_vad(row['audio']['array'], pipe=vad_pipe))
        col_added = True
    vad_str = "- Detect regions of speech with PyAnnote VAD"
    PREPROCESSING_STEPS.append(vad_str)
    print(vad_str)

    if 'vad_duration' not in hf_ds.column_names:
        vad_duration = [
            sum(
                chunk['timestamp'][1]-chunk['timestamp'][0] for chunk in row['vad_chunks']
            ) for row in hf_ds
        ]
        hf_ds = hf_ds.add_column('vad_duration', vad_duration)
        col_added = True
    
    if 'vad_pct' not in hf_ds.column_names:
        vad_pct = [row['vad_duration']/row['duration'] for row in hf_ds]
        hf_ds = hf_ds.add_column('vad_pct', vad_pct)
        col_added = True

    if 'speech_embed' not in hf_ds.column_names:
        audio_list = [row['audio']['array'] for row in hf_ds]
        speech_embeds = []
        speech_enc = SpeechEncoder.from_pretrained('anyspeech/clap-ipa-small-speech')
        for batch in dataloader(audio_list, batch_size=64):
            speech_embeds.extend(t.cpu().numpy() for t in embed_speech(batch, speech_enc))
        hf_ds = hf_ds.add_column("speech_embed", speech_embeds)
        col_added = True
    
    if 'text_embed' not in hf_ds.column_names:
        text_enc = PhoneEncoder.from_pretrained('anyspeech/clap-ipa-small-phone')
        text_list = df['text'].tolist()
        text_embeds = []
        for batch in dataloader(text_list, batch_size=64):
            text_embeds.extend(t.cpu().numpy() for t in embed_text(batch, text_enc))
        hf_ds = hf_ds.add_column("text_embed", text_embeds)
        col_added = True
    
    if 'embed_cos_sim' not in hf_ds.column_names:
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(hf_ds['speech_embed']),
            torch.tensor(hf_ds['text_embed']),
            dim=-1
        ).tolist()
        hf_ds = hf_ds.add_column("embed_cos_sim", similarity)
        col_added = True

    embed_str = "- Calculated CLAP IPA text and phone embeddings for dataset"
    print(embed_str)
    PREPROCESSING_STEPS.append(embed_str)

    if ('wada_snr' not in hf_ds.column_names) or ('nist_snr' not in hf_ds.column_names):
        raise ValueError(
            "Audio dataset does not have SNR values, add these columns using Matlab and restart program."
        )
    
    snr_str = "- Calculated Wada SNR and NIST SNR on audio"
    print(snr_str)
    PREPROCESSING_STEPS.append(snr_str)

    if col_added:
        overwrite_dataset(unproc_ds_path, hf_ds)

    print("Saving audio for MFA")
    mfa_dir = os.path.join(TIRA_ASR_CLIPS_DIR, 'mfa_input')
    mfa_audio = os.path.join(mfa_dir, 'himidan')
    os.makedirs(mfa_dir, exist_ok=True)
    mfa_num_wavs = len(glob(os.path.join(mfa_audio, '*.wav')))
    mfa_num_labs = len(glob(os.path.join(mfa_audio, '*.lab')))
    if not mfa_num_wavs == len(hf_ds) and mfa_num_labs == len(hf_ds):
        hf_ds.map(lambda row: save_row_and_label(row, mfa_audio, df))
    mfa_str = "- Saved audio and text labels for alignment with MFA in dir `mfa_input`"
    PREPROCESSING_STEPS.append(mfa_str)
    print(mfa_str)

    unique_words = set()
    df['text'].str.split().apply(unique_words.update)
    wordlist_path = os.path.join(mfa_dir, 'tira.dict')
    with open(wordlist_path, encoding='utf8', mode='w') as f:
        f.write('\n'.join(
            f"{word}\t{tira2mfa(word)}" for word in unique_words if len(word)>1
        ))
    unique_word_str = f"- Saved MFA dictionary for Tira to `mfa_input/tira.dict`"
    PREPROCESSING_STEPS.append(unique_word_str)
    print(unique_word_str)

    readme_header_str = README_HEADER.substitute(
        num_records=len(df),
        duration=get_df_duration(df),
        mean_duration=get_df_duration(df, agg='mean'),
    )
    readme_out = os.path.join(TIRA_ASR_CLIPS_DIR, 'README.md')
    with open(readme_out, 'w', encoding='utf8') as f:
        f.write(readme_header_str+'\n')
        f.write('\n'.join(PREPROCESSING_STEPS))

    transcriptions_path = os.path.join(TIRA_ASR_CLIPS_DIR, 'transcriptions.csv')
    df.to_csv(transcriptions_path, index_label='index')

if __name__ == '__main__':
    main()