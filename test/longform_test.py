import torch
import torchaudio

from test_utils import assert_chunk_dict_shape
from pympi import Elan
import pandas as pd
import sys
sys.path.append('scripts')
from longform import perform_vad, perform_asr, diarize, load_and_resample, perform_sli, pipeout_to_eaf, init_parser, vad_sli_asr_pipeline, SAMPLE_RATE
from dataset_utils import build_sb_dataloader
from model_utils import LOGREG_PATH
import os
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

def test_asr_whisper():
    """
    `perform_asr` should return a dict with key `text` mapping to a str
    and key 'chunks' which mapping to a list of dicts each with `timestamps`
    and `text` keys
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    asr_out = perform_asr(wav, model_path='openai/whisper-tiny', model_family='whisper', return_timestamps=True)
    assert_chunk_dict_shape(asr_out)
    assert 'text' in asr_out
    assert type(asr_out['text']) is str
    for chunk in asr_out['chunks']:
        assert 'text' in chunk
        assert type(chunk['text']) is str

def test_asr_wav2vec2():
    """
    `perform_asr` should return a dict with key `text` mapping to a str
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    asr_out = perform_asr(wav, model_path='facebook/wav2vec2-base-960h', return_timestamps='word')
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
    To test language selection is having an effect on output,
    run twice with different language maps
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav, return_wav_slices=True)
    vad_chunks = vad_out['vad_chunks']
    sli_chunks, _ = perform_sli(vad_chunks, lr_model=LOGREG_PATH)
    sli_map1 = [
        {
            "language": "Tira",
            "label": "TIC",
            "id": 0,
            "whisper_lang_code": "swahili",
            "whisper_checkpoint": "openai/whisper-tiny"
        },
        {
            "language": "English",
            "label": "ENG",
            "id": 1,
            "whisper_lang_code": "english",
            "whisper_checkpoint": "openai/whisper-tiny"
        }
    ]
    asr_chunks1 = perform_asr(audio=sli_chunks, sli_map=sli_map1)
    for chunk in asr_chunks1:
        assert 'text' in chunk
        assert type(chunk['text']) is str
        assert 'sli_pred' in chunk

    # run ASR again with swapped languages
    sli_map2 = [
        {
            "language": "Tira",
            "label": "TIC",
            "id": 0,
            "whisper_lang_code": "english",
            "whisper_checkpoint": "openai/whisper-tiny"
        },
        {
            "language": "English",
            "label": "ENG",
            "id": 1,
            "whisper_lang_code": "swahili",
            "whisper_checkpoint": "openai/whisper-tiny"
        }
    ]
    asr_chunks2 = perform_asr(audio=sli_chunks, sli_map=sli_map2)
    for chunk in asr_chunks2:
        assert 'sli_pred' in chunk
        assert 'text' in chunk
        assert type(chunk['text']) is str

    text1=' '.join(chunk['text'] for chunk in asr_chunks1)
    text2=' '.join(chunk['text'] for chunk in asr_chunks2)
    assert text1!=text2


def test_vad_asr_pipeline():
    """
    Test that `perform_asr` can handle being passed a list of audio segments
    such as those output by `perform_vad` or `diarize`
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav, return_wav_slices=True)
    vad_chunks = vad_out['vad_chunks']
    asr_out = perform_asr(audio=vad_chunks, model_path='openai/whisper-tiny')
    for chunk in asr_out:
        assert 'text' in chunk
        assert type(chunk['text']) is str

def test_pipeout_to_eaf():
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav, return_wav_slices=True)
    vad_chunks = vad_out['vad_chunks']
    vad_eaf = pipeout_to_eaf(vad_chunks, label='vad')
    assert type(vad_eaf) is Elan.Eaf

    sli_out, _ = perform_sli(vad_chunks, lr_model=LOGREG_PATH)
    sli_eaf = pipeout_to_eaf(sli_out, chunk_key='sli_pred')
    assert type(sli_eaf) is Elan.Eaf

    asr_out = perform_asr(wav, model_path='openai/whisper-tiny', return_timestamps=True)
    asr_eaf = pipeout_to_eaf(asr_out['chunks'])
    assert type(asr_eaf) is Elan.Eaf

def test_annotate(tmpdir):
    wav = load_and_resample(SAMPLE_WAVPATH)
    # save 3 copies of wav to tmpdir
    for i in range(3):
        torchaudio.save(str(tmpdir/f'sample{i}.wav'), wav, SAMPLE_RATE)
    args = init_parser().parse_args([])
    args.lr_model = LOGREG_PATH
    args.input=tmpdir
    args.output=tmpdir
    vad_sli_asr_pipeline(args)

    for i in range(3):
        assert os.path.exists(tmpdir/f'sample{i}.eaf')
        eaf = Elan.Eaf(tmpdir/f'sample{i}.eaf')
        tier_names = eaf.get_tier_names()
        assert 'asr' in tier_names
        assert 'sli_pred' in tier_names
    assert os.path.exists(tmpdir/'metadata.csv')
    df = pd.read_csv(tmpdir/'metadata.csv')
    assert df['wav_source'].nunique() == 3
    assert df['eaf_path'].nunique() == 3
    assert df['sli_pred'].nunique() == 2
    assert 'asr' in df['tier_name'].values