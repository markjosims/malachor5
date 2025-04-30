from glob import glob
from pympi import Eaf
from typing import *
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os

"""
Iterate through every interval in every tier, find any intervals that overlap and remove
the shorter of the two intervals if the overlap portion is greater than 10%.
"""

SPEAKER_TIERS = ['SHA', 'NIN', 'HIM', 'MAR', 'PET', 'MISC']
OVERLAP_CUTOFF = 0.5

def get_all_annotations_for_file(eaf: Eaf) -> List[Tuple[int, int, str, str]]:
    """
    Returns a list of annotations from all SPEAKER_TIERS in the given .eaf file.
    """
    annotations = []
    for tier in SPEAKER_TIERS:
        if tier not in eaf.get_tier_names():
            continue
        tier_annotations = eaf.get_annotation_data_for_tier(tier)
        tier_annotations = [a + (tier,) for a in tier_annotations]
        annotations.extend(tier_annotations)
    return annotations

def get_transcript_from_annotations(annotations: List[Tuple[int, int, str, str]]) -> str:
    """
    Sort annotations by start time, concatenate text values and return.
    """
    annotations.sort(key=lambda t:t[0])
    transcript = " ".join(a[2] for a in annotations)
    return transcript

def get_all_annotations_at_time(eaf: Eaf, time: int) -> List[Tuple[int, int, str, str]]:
    """
    Gets annotations at time for all speaker tiers in eaf object.
    Returns list of tuples of shape (start, end, val, tier).
    """
    annotations = []
    for tier in SPEAKER_TIERS:
        if tier not in eaf.get_tier_names():
            continue
        for interval in eaf.get_annotation_data_at_time(tier, time):
            start, end, val = interval[:3]
            annotations.append((start, end, val, tier))
    return annotations

def get_common_overlap(interval1: Tuple[int, int, str, str], interval2: Tuple[int, int, str, str]) -> int:
    """
    For two Eaf intervals, returns number of ms of overlap between them.
    """
    start1, end1 = interval1[:2]
    start2, end2 = interval2[:2]
    max_end = max(end1, end2)
    overlap_array = np.zeros(max_end, dtype=int)
    overlap_array[start1:end1]+=1
    overlap_array[start2:end2]+=1
    is_overlap = overlap_array==2
    return len(overlap_array[is_overlap])

def get_shorter_interval(
        interval1: Tuple[int, int, str, str],
        interval2: Tuple[int, int, str, str],
    ) -> Tuple[Tuple[int, int, str, str], int]:
    """
    For two Eaf intervals, return the interval with shorter duration, along with its duration.
    """
    start1, end1 = interval1[:2]
    duration1 = end1-start1
    start2, end2 = interval2[:2]
    duration2 = end2-start2
    if duration1<duration2:
        return interval1, duration1
    return interval2, duration2

def remove_overlap(eaf: Eaf, overlap_pcts: List[int], overlap_annotations: List[Tuple[int, int, str, str]]) -> int:
    """
    Given list of two overlapping annotations, check if the amount of overlap is greater than
    the cutoff percentage of the duration of the shorter interval. If so, remove the shorter
    interval from the Eaf object, add the percentage point to the `overlap_pcts` list and return
    the index of the interval removed (from the list of two intervals).
    If interval is not removed, return -1.
    """
    overlap=get_common_overlap(*overlap_annotations)
    short_interval, duration=get_shorter_interval(*overlap_annotations)
    short_interval_i = overlap_annotations.index(short_interval)
    time, tier = short_interval[0], short_interval[-1]
    pcnt_overlap = overlap/duration
    overlap_pcts.append(pcnt_overlap)
    if pcnt_overlap >= OVERLAP_CUTOFF:
        eaf.remove_annotation(tier, time)
        return short_interval_i
    return -1

def main(argv: Optional[Sequence[str]]=None):
    parser = ArgumentParser()
    parser.add_argument('--output', '-o', help='Folder to save transcripts in.')
    args = parser.parse_args(argv)
    eaf_paths = glob('meta/*.eaf')
    for eaf_path in eaf_paths:
        eaf = Eaf(eaf_path)
        print(eaf_path)

        # save transcript before removing overlap
        annotations = get_all_annotations_for_file(eaf)
        print("Total annotations:", len(annotations))
        transcript = get_transcript_from_annotations(annotations)
        transcript_basename = os.path.basename(eaf_path).replace('.eaf', '.txt')
        transcript_path = os.path.join(args.output, transcript_basename)
        with open(transcript_path, encoding='utf8', mode='w') as f:
            f.write(transcript)
        print("Saved transcript to", transcript_path)

        # remove overlap
        maxlen = eaf.get_full_time_interval()[1]
        speaker_ct = np.zeros(maxlen, dtype=int)
        for i, tier in enumerate(SPEAKER_TIERS):
            if tier not in eaf.get_tier_names():
                continue
            for interval in eaf.get_annotation_data_for_tier(tier):
                start, end = interval[:2]
                speaker_ct[start:end]+=1
        speaker_ct = pd.Series(speaker_ct)
        print("Duration per speaker count over total speech duration")
        print(speaker_ct.value_counts()/(speaker_ct>0).sum())
        # set all non-overlapping intervals to 0
        speaker_ct[speaker_ct==1]=0
        overlap_starts = (np.diff(speaker_ct)>0).nonzero()[0]+1
        # for data introspection
        overlap_pcts = []
        for start in overlap_starts:
            overlap_annotations = get_all_annotations_at_time(eaf, start)
            if len(overlap_annotations)<2:
                # tqdm.write(f"No overlaps found at time {start}")
                continue
            if len(overlap_annotations)>2:
                # need to process annotations two at a time
                overlap_annotations.sort(key=lambda t: t[0])
                while len(overlap_annotations)>=2:
                    first_two = overlap_annotations[:2]
                    outcome = remove_overlap(eaf, overlap_pcts, first_two)
                    if outcome == -1:
                        # neither tier was removed
                        overlap_annotations.pop(0)
                    else:
                        # pop annotation that was removed
                        overlap_annotations.pop(outcome)
            else: # len(overlap_annotations)==2
                remove_overlap(eaf, overlap_pcts, overlap_annotations)
        overlap_pcts = pd.Series(overlap_pcts)
        # print("Distribution of overlap:")
        # print(overlap_pcts.describe())
        print("Number of intervals removed:", (overlap_pcts>OVERLAP_CUTOFF).sum())
        annotations_nooverlap = get_all_annotations_for_file(eaf)
        print("Number of remaing annotations:", len(annotations_nooverlap))
        transcript_nooverlap = get_transcript_from_annotations(annotations_nooverlap)
        nooverlap_basename = os.path.basename(eaf_path).replace('.eaf', '-no-overlap.txt')
        nooverlap_path = os.path.join(args.output, nooverlap_basename)
        with open(nooverlap_path, encoding='utf8', mode='w') as f:
            f.write(transcript_nooverlap)
        print("Saved transcript with overlap removed to", nooverlap_path)

if __name__ == '__main__':
    main()