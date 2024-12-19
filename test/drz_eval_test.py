from glob import glob
import os
from pyannote.core import *
from pympi import Elan

import sys
sys.path.append('scripts')
from drz_eval import elan_to_pyannote, get_diarization_metrics
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

def test_diarization_metrics():
    ref = {
        'HIM': Annotation(),
        'MAR': Annotation(),
        'combined': Annotation(),
    }
    ref['HIM'][Segment(0, 1)] = 'TIC'
    ref['HIM'][Segment(3.5, 4.5)] = 'ENG'
    ref['MAR'][Segment(2, 2.5)] = 'ENG'

    ref['combined'][Segment(0, 1)] = 'TIC'
    ref['combined'][Segment(3.5, 4.5)] = 'ENG'
    ref['combined'][Segment(2, 2.5)] = 'ENG'

    hyp = Annotation()
    # missed detection of Tira from 0.5 to 1.0
    hyp[Segment(0, 0.5)] = 'TIC'
    # false alarm of English from 2.5 to 3.0
    hyp[Segment(2, 3)] = 'ENG'
    # confusion of English for Tira from 3.5 to 4.5
    hyp[Segment(3.5, 4.5)] = 'TIC'

    metrics = get_diarization_metrics(ref, hyp)

    assert metrics['combined'] == {
        'total': 2.5,                   # 2.5 seconds of speech overall
        'missed detection': 0.5,        # 0.5 seconds of Tira missed
        'false alarm': 0.5,             # 0.5 seconds of English falsely detected
        'confusion': 1.0,               # 1.0 seconds of confusion between English and Tira
        'correct': 1.0,                 # 1.0 seconds of correct detection
        'diarization error rate': 0.8,  # (0.5 + 0.5 + 1.0) / 2.5 = 0.8
    }
    assert metrics['MAR'] == {
        'total': 0.5,                   # 0.5 seconds of speech for MAR
        'missed detection': 0.0,        # no missed detections
        'confusion': 0.0,               # no language confusion
        'correct': 0.5,                 # 0.5 seconds of correct detection
    }
    assert metrics['HIM'] == {
        'total': 2.0,                   # 2.0 seconds of speech for HIM
        'missed detection': 0.5,        # 0.5 seconds of missed detection for HIM
        'confusion': 0.5,               # 1.0 second language confusion
        'correct': 1.0,                 # 0.5 seconds of correct detection
    }