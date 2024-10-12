from argparse import ArgumentParser
from pympi.Elan import Eaf
from glob import glob
from pathlib import Path
import os
from typing import Optional, Sequence
from collections import defaultdict

"""
For all .eaf files in `args.output`, find an eaf file in `args.input` with the same basename
and add all tiers from the input .eaf file.
If `args.media` is specified, add a linked file pointing to the respective wav in the media dir.
"""

# ---------------- #
# filepath helpers #
# ---------------- #

def get_file_mapping(media, in_eafs, out_eafs):
    eaf_map=defaultdict(dict)
    for eaf_fp in in_eafs:
        stem=Path(eaf_fp).stem
        eaf_map[stem]['in_eaf']=eaf_fp
    for eaf_fp in out_eafs:
        stem=Path(eaf_fp).stem
        eaf_map[stem]['out_eaf']=eaf_map[stem].get('out_eaf', [])+[eaf_fp,]
    for wav_fp in media:
        stem=Path(wav_fp).stem
        eaf_map[stem]['media']=wav_fp
    return eaf_map

# ------------ #
# elan helpers #
# ------------ #

def add_metadata_to_list(eaf_dict):
    in_eaf_fp = eaf_dict['in_eaf']
    in_eaf=Eaf(in_eaf_fp)
    out_eafs=eaf_dict['out_eaf']
    media=eaf_dict.get('media', None)
    for out_eaf_fp in out_eafs:
        print(f"Adding tiers from {in_eaf_fp} to {out_eaf_fp}")
        out_eaf=Eaf(out_eaf_fp)
        out_eaf=add_metadata_to_file(in_eaf, out_eaf, media)
        out_eaf.to_file(out_eaf_fp)

def add_metadata_to_file(in_eaf: Eaf, out_eaf: Eaf, media: str=None) -> Eaf:
    tiers = in_eaf.get_tier_names()
    for tier in tiers:
        out_eaf.add_tier(tier)
        tier_annotations=in_eaf.get_annotation_data_for_tier(tier)
        [out_eaf.add_annotation(tier, *annotation) for annotation in tier_annotations]
    if media:
        out_eaf.add_linked_file(media)
    return out_eaf
        

# ---- #
# main #
# ---- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    parser.add_argument('--media', '-m')
    parser.add_argument('--recursive', '-r', action='store_true')
    return parser


def main(argv:Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args=parser.parse_args(argv)
    media=[]
    if args.recursive:
        in_eafs = glob(os.path.join(args.input,'**/*.eaf'), recursive=True)
        out_eafs = glob(os.path.join(args.output,'**/*.eaf'), recursive=True)
        if args.media:
            media = glob(os.path.join(args.media,'**/*.wav'), recursive=True)
    else:
        in_eafs = glob(os.path.join(args.input,'*.eaf'))
        out_eafs = glob(os.path.join(args.output,'*.eaf'))
        if args.media:
            media = glob(os.path.join(args.media,'**/*.wav'), recursive=True)
    eaf_map=get_file_mapping(media, in_eafs, out_eafs)
    
    for stem, eaf_dict in eaf_map.items():
        print(stem)
        add_metadata_to_list(eaf_dict)


    return 0

if __name__ == '__main__':
    main()