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
    ...