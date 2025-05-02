from pympi import TextGrid
from argparse import ArgumentParser
import sys
from typing import *
sys.path.append('scripts')
from lid_utils import is_tira_word
from kws import textgrid_to_df
import pandas as pd
from tqdm import tqdm
from string import Template
import numpy as np

"""
Given input TextGrids, divides each TextGrid into chunks and creates a prompt for each chunk
using Tira words present. Assumes TextGrid has tiers labeled "$SPEAKER - {phones/words}",
as per format output by Montreal Forced Aligner. Saves a `.json` file with the resulting timestamps
and prompt sentences.
"""

# ---------------- #
# textgrid helpers #
# ---------------- #

def get_window(
        tg_df: pd.DataFrame,
        start: float,
        min_end: float,
        max_end: float,
        prompt_template: Template,
    ) -> Dict[str, Union[str, float]]:
    """
    Given a DataFrame with TextGrid intervals, a start time value, and both
    minimum and maximum end time values, return a window whose end time falls
    between the specified min and max times with key `prompt` containing a prompt
    with all Tira words in the interval.
    """
    after_min_end_mask = tg_df['start'] > min_end
    before_max_end_mask = tg_df['start'] < max_end
    end_window_mask = after_min_end_mask&before_max_end_mask
    non_tira_mask = ~tg_df['is_tira']
    tira_end_mask = tg_df['tira_phrase_edge']==-1

    # window should not cut off a Tira interval
    # if no non-Tira intervals, extend window up to 29s
    extended_window = False
    xmax = tg_df['end'].max()
    while (len(tg_df[end_window_mask&non_tira_mask])==0) and\
        (len(tg_df[end_window_mask&tira_end_mask])==0) and\
        (max_end<xmax) and\
        ((max_end-start)<29):
        max_end+=1
        before_max_end_mask = tg_df['end'] < max_end
        end_window_mask = after_min_end_mask&before_max_end_mask
        extended_window=True
    if max_end>xmax:
        window_end=xmax
    elif (len(tg_df[end_window_mask&non_tira_mask])>0):
        window_end = tg_df.loc[end_window_mask&non_tira_mask, 'start'].iloc[0]
    elif len(tg_df[end_window_mask&tira_end_mask])>0:
        window_end = tg_df.loc[end_window_mask&tira_end_mask, 'start'].iloc[0]
    else: # max_end-start>=29sec
        window_end=max_end
    if extended_window:
        tqdm.write(f"Extended window past max, new duration:, {window_end-start}")
    tira_phrases = get_tira_phrases_in_interval(tg_df, start, window_end)
    prompt = prompt_template.substitute(tira_str=", ".join(tira_phrases))
    return {
        'start': start,
        'end': window_end,
        'prompt': prompt,
    }


def get_tira_phrases_in_interval(tg_df: pd.DataFrame, start: float, end: float) -> List[str]:
    """
    Given specified start and end time value, return a list
    of all Tira phrases in the time interval from `tg_df`.
    """
    start_mask = tg_df['end'] >= start
    end_mask = tg_df['start'] <= end
    interval_df = tg_df[start_mask&end_mask]
    tira_phrases = []
    for speaker in interval_df['speaker'].unique():
        speaker_mask = interval_df['speaker']==speaker
        speaker_text: pd.Series = interval_df[speaker_mask].sort_values('start', inplace=False)['text']
        tira_phrase = ''
        for word in speaker_text.values:
            if is_tira_word(word):
                tira_phrase += ' '+word
            elif tira_phrase:
                tira_phrases.append(tira_phrase)
                tira_phrase = ''
        # case where text ends w Tira phrase
        if tira_phrase:
            tira_phrases.append(tira_phrase)
    return tira_phrases

def get_tira_intervals(tg_df: pd.DataFrame) -> pd.Series:
    """
    Using `is_tira_word()`, identify intervals belonging to
    OR overlapping with Tira speech and add as column 'is_tira'.
    Also identify onsets and offsets of Tira phrases and add as
    column 'tira_phrase_edge'.
    """
    tira_mask=tg_df['text'].apply(is_tira_word)
    tg_df['tira_phrase_edge']=0
    for speaker in tg_df['speaker'].unique():
        speaker_mask = tg_df['speaker']==speaker
        tira_phrase_edges = np.diff(tira_mask[speaker_mask].to_numpy(dtype=int))
        if tira_mask.iloc[0]:
            tira_phrase_edges = np.insert(tira_phrase_edges, 0, 1)
        else:
            tira_phrase_edges = np.insert(tira_phrase_edges, 0, 0)
        tg_df.loc[speaker_mask, 'tira_phrase_edge']=tira_phrase_edges

    for _, row in tg_df[tira_mask.copy()].iterrows():
        start_mask = tg_df['end'] >= row['start']
        end_mask = tg_df['start'] <= row['end']
        tira_mask[start_mask&end_mask]=True
    tg_df['is_tira']=tira_mask
    return tg_df

# ------ #
# script #
# ------ #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input", '-i', help='TextGrid file paths', nargs='+')
    parser.add_argument("--output", '-o', help='Directory to save JSON files to')
    parser.add_argument('--window_len', '-w', type=float, default=15.0,
                        help="Length of window to use (in seconds)"
    )
    parser.add_argument('--window_tol', '-t', type=float, default=5.0,
                        help='Tolerance for window length'
    )
    parser.add_argument('--prompt_template', '-p', type=Template, default="Yeah $tira_str umm")
    return parser

def main(argv: Optional[Sequence[str]]=None):
    parser = init_parser()
    args = parser.parse_args(argv)
    for tg_path in args.input:
        tg_df = textgrid_to_df(tg_path)
        tg_df=get_tira_intervals(tg_df)
        windows = []
        xmin = 0
        xmax = tg_df['end'].max()
        min_windowlen = args.window_len-args.window_tol
        max_windowlen = args.window_len+args.window_tol
        with tqdm(total=xmax//min_windowlen) as pbar:
            while xmin<xmax:
                pbar.update(1)
                window = get_window(
                    tg_df,
                    xmin,
                    xmin+min_windowlen,
                    xmin+max_windowlen,
                    prompt_template=args.prompt_template,
                )
                windows.append(window)
                xmin = window['end']

if __name__ == '__main__':
    main(sys.argv[1:])
