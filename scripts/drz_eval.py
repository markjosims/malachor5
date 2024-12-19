from pympi import Elan
from pyannote.core import Annotation, Segment, Timeline
from typing import Union, Dict

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