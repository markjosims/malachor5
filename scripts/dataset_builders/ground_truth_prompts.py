from pympi import TextGrid
from argparse import ArgumentParser
import sys
from typing import *
sys.path.append('scripts')
from lid_utils import is_tira_word
from kws import textgrid_to_df
import pandas as pd

"""
Given input TextGrids, divides each TextGrid into chunks and creates a prompt for each chunk
using Tira words present. Assumes TextGrid has tiers labeled "$SPEAKER - {phones/words}",
as per format output by Montreal Forced Aligner. Saves a `.json` file with the resulting timestamps
and prompt sentences.
"""

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
    return parser

def get_window(tg_df: pd.DataFrame, start: float, min_end: float, max_end: float):
    """
    Given a DataFrame with TextGrid intervals,
    """

def get_tira_phrases_in_interval():
    ...

def main(argv: Optional[Sequence[str]]=None):
    parser = init_parser()
    args = parser.parse_args(argv)
    for tg_path in args.input:
        tg_df = textgrid_to_df(tg_path)
        tg_df['is_tira']=tg_df['word'].apply(is_tira_word)
        breakpoint()
        windows = []
        


if __name__ == '__main__':
    main(sys.argv[1:])
