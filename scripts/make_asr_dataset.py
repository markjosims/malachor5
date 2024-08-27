from glob import glob
from pympi import Elan
from typing import Optional, Sequence
from argparse import ArgumentParser
import pandas as pd
import os

GDRIVE_DIR = '/Users/markjos/Library/CloudStorage/GoogleDrive-mjsimmons@ucsd.edu/Shared drives/Tira/Recordings'

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', default=GDRIVE_DIR)
    parser.add_argument('--output', '-o')
    parser.add_argument('--transcription_tier', '-t')
    return parser

def main(argv: Optional[Sequence[str]]) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    
    eafs = glob(os.path.join(args.input, '**/*.eaf'), recursive=True)
    

    return 0

if __name__ == '__main__':
    main()