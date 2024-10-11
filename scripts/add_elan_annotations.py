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


def main(argv:Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args=parser.parse_args(argv)

    return 0

if __name__ == '__main__':
    main()