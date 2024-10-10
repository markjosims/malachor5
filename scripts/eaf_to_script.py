from pympi import Elan
from typing import Optional, Sequence, Dict, List, Union
from argparse import ArgumentParser
from glob import glob
import os

"""
Copied from https://github.com/markjosims/montague_archiving 10 Oct 2024
"""

def ms_to_human_time(ms: int) -> str:
    time_s = ms//1000
    hours = time_s//3600
    minutes = time_s%3600//60
    seconds = int(time_s%60)
    return f"{hours}:{minutes:0>2d}:{seconds:0>2d}"

def human_time_to_ms(timestr: str) -> int:
    hours, minutes, seconds = (int(n) for n in timestr.split(sep=':'))
    return 1000 * (hours*3600+minutes*60+seconds)

def write_script(
        eaf: Union[str, Elan.Eaf],
        out_fp: str,
        merge_turns: bool=True,
        keep_line_breaks=True,
    ) -> str:
    if type(eaf) is str:
        eaf = Elan.Eaf(eaf)
    turns = []
    speakers = eaf.get_tier_names()
    for speaker in speakers:
        annotations = eaf.get_annotation_data_for_tier(speaker)
        for start, end, val in annotations:
            turns.append({'start': start, 'end': end, 'text': val, 'speaker': speaker})
    if merge_turns:
        turns = merge_turn_list(turns, keep_line_breaks=keep_line_breaks)

    with open(out_fp, 'w', encoding='utf-8') as f:
        for turn in turns:
            speaker = turn['speaker']
            start = ms_to_human_time(turn['start'])
            end = ms_to_human_time(turn['end'])
            f.write(f"{speaker}: {start}-{end}\n")
            f.write(turn['text'])
            f.write("\n\n")

def merge_turn_list(turns: List[Dict[str, str]], keep_line_breaks=True) -> List[Dict[str, str]]:
    joinstr = '\n\n' if keep_line_breaks else ' '
    turns = sorted(turns, key=lambda d:d['start'])
    merged_turns = []
    i=0
    while i < len(turns):
        turn = turns[i]
        next_turn = turns[i+1] if i<len(turns)-1 else None
        i+=1
        while next_turn and (turn['speaker'] == next_turn['speaker']):
            turn = merge_turn_pair(turn, next_turn, joinstr)
            next_turn = turns[i+1] if i<len(turns)-1 else None
            i+=1
        merged_turns.append(turn)
    
    return merged_turns

def merge_turn_pair(turn1: Dict[str, str], turn2: Dict[str, str], joinstr: str) -> Dict[str, str]:
    merged_turn = {}
    merged_turn['start'] = turn1['start']
    merged_turn['end'] = turn2['end']
    merged_turn['text'] = joinstr.join([turn1['text'], turn2['text']])
    merged_turn['speaker'] = turn1['speaker']

    return merged_turn

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = ArgumentParser(description="Render .eaf files as text transcripts.")
    parser.add_argument("--input", "-i", help=".eaf file or folder of .eaf files")
    parser.add_argument("--recursive", '-r', help='If input is dir, find .eaf files recursively', action='store_true')
    args = parser.parse_args(argv)
    input_fp = args.input
    if os.path.isdir(input_fp):
        if args.recursive:
            glob_str = os.path.join(input_fp, '**', '*.eaf')
        else:
            glob_str = os.path.join(input_fp, '*.eaf')
        eafs = glob(glob_str, recursive=args.recursive)
        for eaf_fp in eafs:
            out_fp = eaf_fp.replace('.eaf', '.txt')
            write_script(eaf_fp, out_fp)
        return 0
    out_fp = input_fp.replace('.eaf', '.txt')
    write_script(input_fp, out_fp)
    return 0

if __name__ == '__main__':
    main()