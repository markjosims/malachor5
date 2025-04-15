import sys
sys.path.append('scripts')
from kws import embed_speech, embed_text
from test_utils import NYEN_PATH, ALBRRIZO_PATH, XDDERE_PATH
from longform import load_and_resample
import torch
import torch.nn.functional as F

def test_embed_speech():
    # test loading embedding from filepath
    for filepath in [NYEN_PATH, ALBRRIZO_PATH, XDDERE_PATH]:
        speech_embed = embed_speech(filepath)
        assert type(speech_embed) is torch.Tensor
        assert speech_embed.shape == [1,256]
    # test loading embedding from tensor of samples
    for filepath in [NYEN_PATH, ALBRRIZO_PATH, XDDERE_PATH]:
        wav = load_and_resample(filepath)
        speech_embed = embed_speech(wav)
        assert type(speech_embed) is torch.Tensor
        assert speech_embed.shape == [1,256]

def test_embed_text():
    for ipa_str in ["ɲɛ̂n", "ɜ̀lbrìðɔ̀", "èd̪ɛ̀ɾɛ̀"]:
        text_embed = embed_text(ipa_str)
        assert type(text_embed) is torch.Tensor
        assert text_embed.shape == [1,256]

def test_cos_sim():
    nyen_speech = embed_speech(NYEN_PATH)
    albrrizo_speech = embed_speech(ALBRRIZO_PATH)
    xddere_speech = embed_speech(XDDERE_PATH)

    nyen_text = embed_text("ɲɛ̂n")
    albrrizo_text = embed_text("ɜ̀lbrìðɔ̀")
    xddere_text = embed_text("èd̪ɛ̀ɾɛ̀")

    speech_embeds = [nyen_speech, albrrizo_speech, xddere_speech]
    text_embeds = [nyen_text, albrrizo_text, xddere_text]
    for i, speech_embed in enumerate(speech_embeds):
        cos_sim_w_self = F.cosine_similarity(speech_embed, text_embeds[i], dim=-1)
        for j, text_embed in enumerate(text_embeds):
            if j==i:
                continue
            cos_sim_w_other = F.cosine_similarity(speech_embed, text_embed)
            assert cos_sim_w_self > cos_sim_w_other
