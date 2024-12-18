from pympi import Elan
from pyannote.core import Annotation, Segment, Timeline
from typing import Union, Dict

def elan_to_pyannote(eaf: Union[str, Elan.Eaf]) -> Dict[str, Annotation]:
    ...