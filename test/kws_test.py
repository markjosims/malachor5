import sys
sys.path.append('scripts')
from kws import embed_speech, embed_text, get_sliding_window
from test_utils import NYEN_PATH, ALBRRIZO_PATH, XDDERE_PATH
from test_utils import NYEN_IPA, ALBRRIZO_IPA, XDDERE_IPA
from longform import load_and_resample
import torch
import torch.nn.functional as F
import pytest

def test_embed_speech():
    # test loading embedding from filepath
    for filepath in [NYEN_PATH, ALBRRIZO_PATH, XDDERE_PATH]:
        speech_embed = embed_speech(filepath)
        assert type(speech_embed) is torch.Tensor
        assert speech_embed.shape[1] == 384
        assert speech_embed.shape[0] == 1
    # test loading embedding from tensor of samples
    for filepath in [NYEN_PATH, ALBRRIZO_PATH, XDDERE_PATH]:
        wav = load_and_resample(filepath)
        speech_embed = embed_speech(wav)
        assert type(speech_embed) is torch.Tensor
        assert speech_embed.shape[1] == 384
        assert speech_embed.shape[0] == 1

def test_embed_speech_batch():
    # test loading embedding from filepaths
    audio_paths = [NYEN_PATH, ALBRRIZO_PATH, XDDERE_PATH]
    speech_embed = embed_speech(audio_paths)
    assert type(speech_embed) is torch.Tensor
    assert speech_embed.shape[0] == 3
    assert speech_embed.shape[1] == 384

    wavs = [load_and_resample(fp) for fp in audio_paths]
    speech_embed = embed_speech(wavs)
    assert type(speech_embed) is torch.Tensor
    assert speech_embed.shape[0] == 3
    assert speech_embed.shape[1] == 384

def test_embed_text():
    for ipa_str in [NYEN_IPA, XDDERE_IPA, ALBRRIZO_IPA]:
        text_embed = embed_text(ipa_str)
        assert type(text_embed) is torch.Tensor
        assert text_embed.shape[1] == 384
        assert text_embed.shape[0] == 1

def test_embed_text_batch():
    ipa_strs = [NYEN_IPA, XDDERE_IPA, ALBRRIZO_IPA]
    text_embed = embed_text(ipa_strs)
    assert type(text_embed) is torch.Tensor
    assert text_embed.shape[1] == 384
    assert text_embed.shape[0] == 3

@pytest.mark.parametrize(
    "encoder_size,encoder_dim",
    [
        ('tiny', 384),
        ('base', 512),
        ('small', 512),
    ]
)
def test_embed_size(encoder_size, encoder_dim):
    speech_embed = embed_speech(NYEN_PATH, encoder_size=encoder_size)
    assert speech_embed.shape[1]==encoder_dim
    
    text_embed = embed_text(NYEN_IPA, encoder_size=encoder_size)
    assert text_embed.shape[1]==encoder_dim

def test_cos_sim():
    nyen_speech = embed_speech(NYEN_PATH)
    albrrizo_speech = embed_speech(ALBRRIZO_PATH)
    xddere_speech = embed_speech(XDDERE_PATH)

    nyen_text = embed_text(NYEN_IPA)
    albrrizo_text = embed_text(ALBRRIZO_IPA)
    xddere_text = embed_text(XDDERE_IPA)

    speech_embeds = [nyen_speech, albrrizo_speech, xddere_speech]
    text_embeds = [nyen_text, albrrizo_text, xddere_text]
    for i, speech_embed in enumerate(speech_embeds):
        cos_sim_w_self = F.cosine_similarity(speech_embed, text_embeds[i], dim=-1)
        for j, text_embed in enumerate(text_embeds):
            if j==i:
                continue
            cos_sim_w_other = F.cosine_similarity(speech_embed, text_embed)
            assert cos_sim_w_self > cos_sim_w_other

# sliding window unit tests courtesy of Her Probabilistic Majesty, Lady ChatGPT
@pytest.mark.parametrize(
    "audio,sample_rate,framelength_s,frameshift_s,expected",
    [
        # Test with list input
        (
            list(range(16)), 4, 2, 1,
            [
                list(range(0, 8)),
                list(range(4, 12)),
                list(range(8, 16)),
            ]
        ),
        # Test with partial frame at the end
        (
            list(range(10)), 5, 1, 0.6,
            [
                list(range(0, 5)),
                list(range(3, 8)),
                list(range(6, 10)),
            ]
        ),
        # Test with exact fit
        (
            list(range(12)), 4, 1, 1,
            [
                list(range(0, 4)),
                list(range(4, 8)),
                list(range(8, 12)),
            ]
        ),
        # Test with short audio
        (
            list(range(5)), 10, 1, 0.5,
            [list(range(5))]
        ),
        # Test with non-integer stride
        (
            list(range(20)), 10, 0.5, 0.3,
            [
                list(range(0, 5)),
                list(range(3, 8)),
                list(range(6, 11)),
                list(range(9, 14)),
                list(range(12, 17)),
                list(range(15, 20)),
            ]
        ),
    ]
)
def test_get_sliding_window_list(audio, sample_rate, framelength_s, frameshift_s, expected):
    result = get_sliding_window(audio, framelength_s, frameshift_s, sample_rate)
    assert result == expected


@pytest.mark.parametrize(
    "audio,sample_rate,framelength_s,frameshift_s,expected",
    [
        (
            torch.arange(16), 4, 2, 1,
            [
                torch.arange(0, 8),
                torch.arange(4, 12),
                torch.arange(8, 16),
            ]
        ),
        (
            torch.arange(40), 10, 1, 0.5,
            [torch.arange(i, i + 10) for i in range(0, 31, 5)]
        )
    ]
)
def test_get_sliding_window_tensor(audio, sample_rate, framelength_s, frameshift_s, expected):
    result = get_sliding_window(audio, framelength_s, frameshift_s, sample_rate)
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert torch.equal(r, e)