from pympi import Elan
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import IdentificationErrorRate
from typing import Union, Dict
import pandas as pd

def elan_to_pyannote(eaf: Union[str, Elan.Eaf]) -> Dict[str, Annotation]:
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

    tiers = eaf.get_tier_names()
    speaker_tiers = [t for t in tiers if len(t)==3 and t.isupper()]
    for speaker in speaker_tiers:
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
    if return_df:
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        return metrics_df
    return metrics_dict