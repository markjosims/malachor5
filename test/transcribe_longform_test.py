import torch

from test_utils import assert_chunk_dict_shape

import sys
sys.path.append('scripts')
from transcribe_longform import perform_vad, perform_asr, diarize, load_and_resample
SAMPLE_WAVPATH = 'test/data/sample_biling.wav'

def test_load_and_resample():
    """
    `load_and_resample` should return a single-channel numpy array of samples
    """
    wav = load_and_resample(SAMPLE_WAVPATH, to_mono=True, flatten=False)
    assert type(wav) is torch.Tensor
    assert len(wav.shape)==2

def test_vad():
    """
    `perform_vad` should return a dict with key 'vad_chunks' which maps to
    a list of dicts each with a `timestamps` key mapping to a 2-tuple of floats
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav)
    assert_chunk_dict_shape(vad_out, 'vad_chunks')

def test_asr():
    """
    `perform_asr` should return a dict with key `text` mapping to a str
    and key 'chunks' which mapping to a list of dicts each with `timestamps`
    and `text` keys
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    asr_out = perform_asr(wav, model_path='openai/whisper-tiny')
    assert_chunk_dict_shape(asr_out)
    assert 'text' in asr_out
    assert type(asr_out['text']) is str

def test_drz():
    """
    `diarize` should return a dict with key 'drz_chunks' which maps to
    a list of dicts each with a `timestamps` key mapping to a 2-tuple of floats
    and a `speaker` key mapping to a str
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    drz_out = diarize(wav)
    assert_chunk_dict_shape(drz_out, 'drz_chunks')
    for chunk in drz_out['drz_chunks']:
        assert 'speaker' in chunk
        assert type(chunk['speaker']) is str