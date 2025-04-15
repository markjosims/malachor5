from typing import Union, List
from model_utils import DEVICE
from clap.encoders import SpeechEncoder, PhoneEncoder
from transformers import DebertaV2Tokenizer, AutoProcessor
from longform import load_and_resample
import torch
import numpy as np

def embed_speech(audio: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
    speech_encoder = SpeechEncoder.from_pretrained('anyspeech/clap-ipa-tiny-speech')
    speech_encoder.eval().to(DEVICE)

    processor = AutoProcessor.from_pretrained('openai/whisper-tiny')

    if type(audio) is str:
        audio = load_and_resample(audio)
    audio = audio.squeeze()
    audio_input = processor(
        audio,
        sampling_rate=16_000, # these kwargs avoid bugs
        return_tensors='pt',
        return_attention_mask=True,
    )
    audio_input=audio_input.to(DEVICE)

    with torch.no_grad():
        speech_embed = speech_encoder(**audio_input)['pooler_output']
    return speech_embed

def embed_text(text: Union[str, List[str]]) -> torch.Tensor:
    phone_encoder = PhoneEncoder.from_pretrained('anyspeech/clap-ipa-tiny-phone')
    phone_encoder.eval().to(DEVICE)

    tokenizer = DebertaV2Tokenizer.from_pretrained('charsiu/IPATokenizer')

    ipa_input = tokenizer(
        text,
        return_tensors='pt', # these kwargs avoid bugs
        return_token_type_ids=False,
        padding=True,
    )
    ipa_input=ipa_input.to(DEVICE)

    with torch.no_grad():
        phone_embed = phone_encoder(**ipa_input)['pooler_output']
    return phone_embed
