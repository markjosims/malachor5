from typing import Union, List, Literal
from model_utils import DEVICE
from clap.encoders import SpeechEncoder, PhoneEncoder
from transformers import DebertaV2Tokenizer, AutoProcessor
from longform import load_and_resample, prepare_tensor_for_feature_extraction, SAMPLE_RATE
import torch
import numpy as np

# ----------------- #
# embedding helpers #
# ----------------- #

def embed_speech(
        audio: Union[str, List[str], torch.Tensor],
        speech_encoder: Union[str, SpeechEncoder]=None,
        encoder_size: Literal["tiny", "base", "small"]="tiny",
    ) -> torch.Tensor:
    if speech_encoder is None:
        speech_encoder = f'anyspeech/clap-ipa-{encoder_size}-speech'
    if type(speech_encoder) is str:
        speech_encoder = SpeechEncoder.from_pretrained(speech_encoder)
    speech_encoder.eval().to(DEVICE)

    processor = AutoProcessor.from_pretrained('openai/whisper-tiny')

    if (type(audio) is str) or (type(audio) is list and type(audio[0]) is str):
        audio = load_and_resample(audio)
    audio = prepare_tensor_for_feature_extraction(audio)

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

def embed_text(
        text: Union[str, List[str]],
        phone_encoder: Union[str, PhoneEncoder]=None,
        encoder_size: Literal["tiny", "base", "small"]="tiny",
    ) -> torch.Tensor:
    if phone_encoder is None:
        phone_encoder = f'anyspeech/clap-ipa-{encoder_size}-phone'
    if type(phone_encoder) is str:
        phone_encoder = PhoneEncoder.from_pretrained(phone_encoder)
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

# ------------- #
# audio helpers #
# ------------- #

def get_sliding_window(
        audio: torch.Tensor,
        framelength_s: float,
        frameshift_s: float,
        sample_rate: int = SAMPLE_RATE,
    ):
    if len(audio)==0:
        return []
    framelength_samples = int(framelength_s * sample_rate)
    frameshift_samples = int(frameshift_s * sample_rate)
    
    frame_start = 0
    frame_end = framelength_samples
    windows = []
    while frame_end<len(audio):
        windows.append(audio[frame_start:frame_end])
        frame_start+=frameshift_samples
        frame_end+=frameshift_samples
    windows.append(audio[frame_start:])
    return windows