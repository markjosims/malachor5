from pympi import Elan
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import IdentificationErrorRate
from typing import Union, Dict, Sequence, Optional
import pandas as pd
from argparse import ArgumentParser
import os
from glob import glob

def elan_to_pyannote(eaf: Union[str, Elan.Eaf], tgt_tiers: Optional[Sequence[str]]=None) -> Dict[str, Annotation]:
    """
    Returns a dict of pyannote `Annotation` objects from an Elan file.
    The label for each annotation segment is the language being spoken for that interval.
    Creates one annotation for each speaker tier (where speaker tiers are
    are assumed to be those whose names are three capital letters, e.g. HIM or MAR),
    as well as 'combined' and 'verbose' `Annotation` objects, both of which
    pool annotations from all speakers into the same timeline, but where 'combined'
    only includes the language label and `verbose` includes both language and speaker label.
    """
    if type(eaf) is str:
        eaf = Elan.Eaf(eaf)
    annotations_dict = {
        'combined': Annotation(),
        'verbose': Annotation(),
    }

    if not tgt_tiers:
        tiers = eaf.get_tier_names()
        tgt_tiers = [t for t in tiers if len(t)==3 and t.isupper()]
    for speaker in tgt_tiers:
        annotations_dict[speaker]=Annotation()
        speaker_annotations = eaf.get_annotation_data_for_tier(speaker)
        for start_ms, end_ms, val in speaker_annotations:
            if val == 'NOLING':
                continue
            # map ENGTIC > ENG and TICENG > TIC
            if val.startswith('ENG'):
                val='ENG'
            elif val.startswith('TIC'):
                val='TIC'
            # if not a recognized label, assume text is transcribed sentence and default to ENG
            # since transcribed sentences are generally English matrix clauses w/ Tira words
            elif val not in ('ENG', 'TIC'):
                val='ENG'
            start = start_ms/1000
            end = end_ms/1000
            annotations_dict[speaker][Segment(start, end)] = val
            annotations_dict['combined'][Segment(start, end)] = val
            annotations_dict['verbose'][Segment(start, end)] = f'{val} ({speaker})'
    return annotations_dict

def get_diarization_metrics(
        ref: Union[str, Elan.Eaf, Dict[str, Annotation]],
        hyp: Union[str, Elan.Eaf, Annotation],
        return_df: bool = False,
        return_pct: bool = True,
    ) -> Dict[str, Dict[str, float]]:
    """
    `ref` is a path to an Elan file, an Eaf object, or dictionary of pyannote `Annotation` objects
    from the `elan_to_pyannote` function.
    `hyp` is a path to an Elan file, an Eaf object, or a pyannote `Annotation` object.
    Assume that `ref` contains tiers for multiple speakers but `hyp` contains a single tier
    reflecting VAD+SLI output.
    Returns a dictionary for each speaker tier in the reference, containing
    a dictionary of diarization metrics for that speaker tier,
    as well as a 'combined' key for overall metrics.
    Not that the 'confusion' metric here is language confusion, not speaker confusion.
    """
    if type(ref) in (str, Elan.Eaf):
        ref = elan_to_pyannote(ref)
    if type(hyp) in (str, Elan.Eaf):
        hyp = elan_to_pyannote(hyp)['combined']
    der = IdentificationErrorRate(collar=0.0)
    metrics_dict = {}
    for speaker, annotation in ref.items():
        if speaker == 'verbose':
            continue
        if speaker == 'combined':
            uem=hyp.get_timeline().union(annotation.get_timeline())
        else:
            uem=annotation.get_timeline()
        metrics = der(annotation, hyp, detailed=True, uem=uem)
        if speaker != 'combined':
            # don't measure false alarm or DER for individual speakers
            metrics.pop('false alarm')
            metrics.pop('identification error rate')
        # add language-specific metrics
        for lang in ['ENG', 'TIC']:
            lang_str = 'tira' if lang=='TIC' else 'eng'
            if speaker == 'combined':
                # for false alarm compare `ref` all segments
                # to `hyp` with only `lang` segments
                hyp_lang = hyp.subset([lang])
                metrics[f'{lang_str} false alarm'] = der(annotation, hyp_lang, detailed=True, uem=uem)['false alarm']
            # for correct, confusion and missed detection, compare `ref` w/ only `lang` segments
            # to `hyp` w/ all segments
            ref_lang = annotation.subset([lang])
            lang_der = der(ref_lang, hyp, detailed=True, uem=uem)
            metrics[f'{lang_str} correct'] = lang_der['correct']
            metrics[f'{lang_str} confusion'] = lang_der['confusion']
            metrics[f'{lang_str} missed detection'] = lang_der['missed detection']
            metrics[f'{lang_str} total'] = lang_der['total']

        metrics_dict[speaker]=metrics
    if return_pct:
        for speaker, metrics in metrics_dict.items():
            for key, value in metrics.copy().items():
                if key in ('total', 'tira total', 'eng total'):
                    continue
                if 'rate' in key:
                    continue
                total = metrics['total']
                if 'eng' in key:
                    total = metrics['eng total']
                elif 'tic' in key:
                    total = metrics['tira total']
                metrics[key+' rate'] = value/total
    if return_df:
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        return metrics_df
    return metrics_dict

def average_metrics_by_speaker(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of diarization metrics for each speaker in a file,
    adds rows for average metrics across all speakers.
    """
    average_metrics = []
    for speaker in metrics_df['speaker']:
        speaker_df = metrics_df[metrics_df['speaker']==speaker]
        speaker_metrics = speaker_df.mean()
        speaker_metrics['file']='average'
        average_metrics.append(speaker_metrics)
    average_metrics_df = pd.DataFrame(average_metrics)
    metrics_df = pd.concat([metrics_df, average_metrics_df])
    return metrics_df

def init_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate diarization metrics.")
    parser.add_argument('reference', type=str, help="Path to the reference Elan file or directory.")
    parser.add_argument('hypothesis', type=str, help="Path to the hypothesis Elan file or directory.")
    parser.add_argument('output', type=str, help="Path to the output file to save the results.")
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)

    if os.path.isdir(args.ref):
        ref = glob(args.ref+'/*.eaf')
        hyp = glob(args.hyp+'/*.eaf')
        df_list = []
        for ref_path, hyp_path in zip(ref, hyp):
            file_metrics = get_diarization_metrics(ref_path, hyp_path, return_df=True)
            file_metrics=file_metrics.reset_index(names=['speaker'])
            file_metrics['file']=os.path.basename(ref_path)
            df_list.append(file_metrics)
        df = pd.concat(df_list)
        df = average_metrics_by_speaker(df)
        df.to_csv(args.output)
        return 0
    metrics = get_diarization_metrics(args.ref, args.hyp, return_df=True)
    metrics.to_csv(args.output)
    return 0

if __name__ == '__main__':
    main()