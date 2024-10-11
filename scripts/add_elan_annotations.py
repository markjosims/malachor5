from argparse import ArgumentParser
from pympi.Elan import Eaf
from glob import glob
from pathlib import Path
from typing import Optional, Sequence
from collections import defaultdict

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
    media=[]
    if args.recursive:
        in_eafs = glob(Path(args.input)/'**/*.eaf', recursive=True)
        out_eafs = glob(Path(args.output)/'**/*.eaf', recursive=True)
        if args.media:
            media = glob(Path(args.media)/'**/*.wav', recursive=True)
    else:
        in_eafs = glob(Path(args.input)/'*.eaf')
        out_eafs = glob(Path(args.output)/'*.eaf')
        if args.media:
            media = glob(Path(args.media)/'**/*.wav', recursive=True)
    eaf_map=defaultdict(dict)
    for eaf_fp in in_eafs:
        stem=Path(eaf_fp).stem
        eaf_map[stem]['in_eaf']=eaf_fp
    for eaf_fp in out_eafs:
        stem=Path(eaf_fp).stem
        eaf_map[stem]['out_eaf']=eaf_fp
    for wav_fp in media:
        stem=Path(wav_fp).stem
        eaf_map[stem]['media']=wav_fp


    return 0

if __name__ == '__main__':
    main()