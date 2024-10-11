from argparse import ArgumentParser
from pympi.Elan import Eaf
from glob import glob
import os
from typing import Optional, Sequence

"""
For all .eaf files in `args.output`, find an eaf file in `args.input` with the same basename
and add all tiers from the input .eaf file.
If `args.media` is specified, add a linked file pointing to the respective wav in the media dir.
"""

# ---- #
# main #
# ---- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    parser.add_argument('--media', '-m')
    parser.add_argument('--recursive', '-r')


def main(argv:Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args=parser.parse_args(argv)
    if args.recursive:
        in_eafs = glob(os.path.join(args.input, '**', '*.wav'), recursive=True)
        out_eafs = glob(os.path.join(args.output, '**', '*.wav'), recursive=True)
    else:
        in_eafs = glob(os.path.join(args.input, '*.wav'))
        out_eafs = glob(os.path.join(args.output, '*.wav'))

    return 0

if __name__ == '__main__':
    main()