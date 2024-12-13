import torch

from test_utils import assert_chunk_dict_shape

import numpy as np
import sys
sys.path.append('scripts')
from transcribe_longform import perform_vad, perform_asr, diarize, load_and_resample, perform_sli
from dataset_utils import build_sb_dataloader
from model_utils import LOGREG_PATH
SAMPLE_WAVPATH = 'test/data/sample_biling.wav'

def test_load_and_resample():
    """
    `load_and_resample` should return a single-channel torch tensor of samples
    """
    wav = load_and_resample(SAMPLE_WAVPATH, to_mono=True, flatten=False)
    assert type(wav) is torch.Tensor
    assert len(wav.shape)==2

def test_chunk_dataloader():
    """
    `build_sb_dataloader` should accept a list of chunks containing wav slices
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav, return_wav_slices=True)
    vad_chunks = vad_out['vad_chunks']
    dataloader = build_sb_dataloader(vad_chunks, batch_size=2, dataset_type='chunk_list')
    assert len(dataloader) == len(vad_chunks)//2
    batch = next(iter(dataloader))
    assert type(batch) is torch.Tensor
    assert len(batch) == 2

def test_vad():
    """
    `perform_vad` should return a dict with key 'vad_chunks' which maps to
    a list of dicts each with a `timestamps` key mapping to a 2-tuple of floats
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav)
    assert_chunk_dict_shape(vad_out, chunks_key='vad_chunks')

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
    for chunk in asr_out['chunks']:
        assert 'text' in chunk
        assert type(chunk['text']) is str

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

def test_return_wavslices():
    """
    `perform_vad` and `diarize` should return a wav slice
    with each chunk if the option `return_wav_slices` is passed
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav, return_wav_slices=True)
    drz_out = diarize(wav, return_wav_slices=True)

    for chunk in vad_out['vad_chunks']:
        assert 'wav' in chunk
        assert type(chunk['wav']) is torch.Tensor
    
    for chunk in drz_out['drz_chunks']:
        assert 'wav' in chunk
        assert type(chunk['wav']) is torch.Tensor

def test_sli():
    """
    `perform_sli` should accept an `annotations` dict with
    a list of wav slices and add the `sli_pred` key to each chunk
    in the list
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav, return_wav_slices=True)
    vad_chunks = vad_out['vad_chunks']
    sli_chunks, args = perform_sli(vad_chunks, lr_model=LOGREG_PATH)
    for chunk in sli_chunks:
        assert 'sli_pred' in chunk
        assert chunk['sli_pred'] in ('TIC', 'ENG')

def test_vad_sli_asr_pipeline():
    """
    First, segement audio with `perform_vad`.
    Then perform SLI on vad chunks with `perform_sli`.
    Then pass chunks to `perform_asr`
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav, return_wav_slices=True)
    vad_chunks = vad_out['vad_chunks']
    sli_chunks, args = perform_sli(vad_chunks, lr_model=LOGREG_PATH)
    asr_chunks = perform_asr(audio=sli_chunks, sli_map=args.sli_map)
    for chunk in asr_chunks:
        assert 'text' in chunk
        assert type(chunk['text']) is str
        if chunk['sli_pred'] == 'ENG':
            assert '<|en|>' in chunk['text']
        else:
            assert '<|sw|>' in chunk['text']