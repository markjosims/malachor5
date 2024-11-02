import numpy as np

from test_utils import assert_chunk_dict_shape

import sys
sys.path.append('scripts')
from transcribe_longform import perform_vad, load_and_resample
SAMPLE_WAVPATH = 'test/data/sample_biling.wav'

def test_load_and_resample():
    """
    `load_and_resample` should return a single-channel numpy array of samples
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    assert type(wav) is np.ndarray
    assert len(wav.shape)==1

def test_vad():
    """
    `perform_vad` should return a dict with key 'chunks' which maps to
    a list of dicts each with a `timestamps` key mapping to a 2-tuple of floats
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav)
    assert_chunk_dict_shape(vad_out)
