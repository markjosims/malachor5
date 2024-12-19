from glob import glob
import os
from pyannote.core import *
from pympi import Elan

import sys
sys.path.append('scripts')
from drz_eval import elan_to_pyannote
from tokenization_utils import TIRA_LONGFORM

def test_elan_to_pyannote_tiers():
    """
    Make pyannote objects out of all .eaf files in TIRA_LONGFORM dataset.
    Ensure that `elan_to_pyannote` returns a dict containing keys for each speaker
    each mapping to an `Annotation` object.
    """
    eaf_paths = glob(os.path.join(TIRA_LONGFORM, 'eaf', '*.eaf'))
    assert len(eaf_paths)==8
    tier_names = ['HIM', 'SHA', 'NIN', 'PET', 'MAR', 'combined', 'verbose']
    for eaf in eaf_paths:
        pyannote_dict = elan_to_pyannote(eaf)
        for tier in tier_names:
            assert tier in pyannote_dict
            assert type(pyannote_dict[tier]) is Annotation

def test_elan_to_pyannote_data():
    """
    Create a toy Elan object and test that the corresponding pyannote `Annotation`
    objects have the right data.
    """
    eaf = Elan.Eaf()
    eaf.add_tier('HIM')
    eaf.add_annotation('HIM', 0, 500, 'TIC')
    eaf.add_annotation('HIM', 1000, 1500, "yeah that's right àpɾí")
    eaf.add_annotation('HIM', 3000, 3500, 'ENG')
    eaf.add_tier('MAR')
    eaf.add_annotation('MAR', 600, 900, 'TIC')
    eaf.add_annotation('MAR', 1600, 2200, 'ENG')

    pyannote_dict = elan_to_pyannote(eaf)

    him_annotation = Annotation()
    him_annotation[Segment(0, 0.5)] = 'TIC'
    him_annotation[Segment(1, 1.5)] = 'ENG'
    him_annotation[Segment(3, 3.5)] = 'ENG'

    mar_annotation = Annotation()
    mar_annotation[Segment(0.6, 0.9)] = 'TIC'
    mar_annotation[Segment(1.6, 2.2)] = 'ENG'

    assert pyannote_dict['HIM'] == him_annotation
    assert pyannote_dict['MAR'] == mar_annotation