from pympi import Elan
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import IdentificationErrorRate
from pyannote.metrics.detection import DetectionErrorRate
from typing import Union, Dict, Sequence, Optional, Literal
import pandas as pd
from argparse import ArgumentParser
import os
from glob import glob
from tqdm import tqdm

import sys
sys.path.append('scripts')
from longform import perform_vad, perform_sli, load_and_resample, pipeout_to_eaf, VAD_URI

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
        task: Literal['sli', 'vad'] = 'sli',
        return_df: bool = False,
        return_pct: bool = True,
        collar: float = 0.0,
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
        # assuming for now that `hyp` Elan object contains a tier with the name of the task performed
        hyp = elan_to_pyannote(hyp, tgt_tiers=[task,])['combined']
    calc_metric = IdentificationErrorRate(collar=collar) if task=='sli' else DetectionErrorRate(collar=collar)
    metrics_dict = {}
    for speaker, annotation in ref.items():
        if speaker == 'verbose':
            continue
        if speaker == 'combined':
            uem=hyp.get_timeline().union(annotation.get_timeline())
        else:
            uem=annotation.get_timeline()
        metrics = calc_metric(annotation, hyp, detailed=True, uem=uem)
        if speaker != 'combined':
            # don't measure false alarm or DER for individual speakers
            metrics.pop('false alarm')
            metrics.pop('detection error rate', None)
            metrics.pop('diarization error rate', None)
        # make vad keys consistent with sli
        if task=='vad':
            metrics['missed detection'] = metrics.pop('miss')
            metrics['correct'] = metrics['total']-metrics['missed detection']
        # add language-specific metrics
        for lang in ['ENG', 'TIC']:
            lang_str = 'tira' if lang=='TIC' else 'eng'
            # only calculate false alarm when considering all speakers combined
            if speaker == 'combined' and task=='sli':
                # for false alarm compare `ref` all segments
                # to `hyp` with only `lang` segments
                hyp_lang = hyp.subset([lang])
                metrics[f'{lang_str} false alarm'] = calc_metric(annotation, hyp_lang, detailed=True, uem=uem)['false alarm']
            # for correct, confusion and missed detection, compare `ref` w/ only `lang` segments
            # to `hyp` w/ all segments
            ref_lang = annotation.subset([lang])
            lang_metric = calc_metric(ref_lang, hyp, detailed=True, uem=uem)
            metrics[f'{lang_str} total'] = lang_metric['total']
            # IER keys
            if task == 'sli':
                metrics[f'{lang_str} missed detection'] = lang_metric['missed detection']
                metrics[f'{lang_str} correct'] = lang_metric['correct']
                metrics[f'{lang_str} confusion'] = lang_metric['confusion']
            # detection error rate keys
            elif task == 'vad':
                metrics[f'{lang_str} missed detection'] = lang_metric['miss']
                metrics[f'{lang_str} correct'] = lang_metric['total']-lang_metric['miss']


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
                metrics[key+' rate'] = value/total if total!=0 else 0
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
        speaker_metrics = speaker_df.mean(numeric_only=True)
        speaker_metrics['speaker']=speaker
        speaker_metrics['file']='average'
        average_metrics.append(speaker_metrics)
    average_metrics_df = pd.DataFrame(average_metrics)
    metrics_df = pd.concat([metrics_df, average_metrics_df])
    return metrics_df

def evaluate_diarization(args):
    if os.path.isdir(args.ref):
        # assuming files have same basename so calling sort will make sure they correspond
        ref = sorted(glob(args.ref+'/*.eaf'))
        if args.hyp:
            hyp = sorted(glob(args.hyp+'/*.eaf'))
        else:
            # perform VAD+SLI on each file in `wav` directory
            wavs = sorted(glob(args.wav+'/*.wav'))
            hyp=[]
            for wav_fp in tqdm(
                wavs,
                desc=f"Performing {'VAD and SLI' if args.logreg else 'VAD'} on audio files"
            ):
                tqdm.write(wav_fp)
                wav = load_and_resample(wav_fp)
                vad_out = perform_vad(wav, return_wav_slices=True, vad_uri=args.vad_uri)
                if args.logreg:
                    sli_out, _ = perform_sli(vad_out['vad_chunks'], lr_model=args.logreg)
                    eaf = pipeout_to_eaf(sli_out, chunk_key='sli_pred', tier_name='sli')
                else:
                    eaf = pipeout_to_eaf(vad_out['vad_chunks'], label='vad', tier_name='vad')
                hyp.append(eaf)
        df_list = []
        for ref_path, hyp_path in tqdm(zip(ref, hyp), desc='Getting diarization metrics', total=len(ref)):
            file_metrics = get_diarization_metrics(
                ref_path,
                hyp_path,
                return_df=True,
                task='sli' if args.logreg else 'vad',
                collar=args.collar,
            )
            file_metrics=file_metrics.reset_index(names=['speaker'])
            file_metrics['file']=os.path.basename(ref_path)
            df_list.append(file_metrics)
        df = pd.concat(df_list)
        df = average_metrics_by_speaker(df)
        print(f"Saving metrics to {args.output}")
        df.to_csv(args.output, index=False)
        return 0
    metrics = get_diarization_metrics(args.ref, args.hyp, return_df=True, collar=args.collar)
    print(f"Saving metrics to {args.output}")
    metrics.to_csv(args.output, index=False)
    return 0

def init_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate diarization metrics.")
    parser.add_argument('--ref', '-r', type=str, help="Path to the reference Elan file or directory.")
    parser.add_argument('--hyp', type=str, help="Path to the hypothesis Elan file or directory.")
    parser.add_argument('--output', '-o', type=str, help="Path to the output file to save the results.")
    parser.add_argument('--wav', '-w', type=str, help="Path to the directory containing the audio files.")
    parser.add_argument('--logreg', '-l', type=str, help="Path to the logreg model.")
    parser.add_argument('--vad_uri', type=str, help="URI for the VAD model.", default=VAD_URI)
    parser.add_argument('--collar', '-c', type=float, help="Collar for diarization metrics.", default=0.0)
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    return evaluate_diarization(args)

if __name__ == '__main__':
    main()