from glob import glob
import os
from pyannote.core import *

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