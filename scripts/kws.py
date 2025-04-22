from typing import Union, List, Literal, Sequence, Any, Generator
from model_utils import DEVICE
from clap.encoders import SpeechEncoder, PhoneEncoder
from transformers import DebertaV2Tokenizer, AutoProcessor
from longform import load_and_resample, prepare_tensor_for_feature_extraction, SAMPLE_RATE
import torch
from argparse import ArgumentParser
import json
import sys
import os
from tqdm import tqdm

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

def get_keyword_sim(
        audio_list: Union[List[str], List[torch.Tensor]],
        text_list: List[str],
        speech_encoder=None,
        phone_encoder=None,
        encoder_size='tiny',
):  
    speech_embeds = embed_speech(audio_list, speech_encoder, encoder_size)
    text_embeds = embed_text(text_list, phone_encoder, encoder_size)

    speech_embed_norm = torch.linalg.vector_norm(speech_embeds, dim=1)[:, None]
    normalized_speech_embeds = speech_embeds/speech_embed_norm
    
    text_embed_norm = torch.linalg.vector_norm(text_embeds, dim=1)[:, None]
    normalized_text_embeds = text_embeds/text_embed_norm

    sim_mat = torch.mm(normalized_speech_embeds, normalized_text_embeds.transpose(0,1))
    return sim_mat

# ------------- #
# audio helpers #
# ------------- #

def dataloader(data: Sequence[Any], batch_size: int) -> Generator[Sequence[any], None, None]:
    for start_index in tqdm(range(0, len(data), batch_size)):
        end_index = start_index+batch_size
        yield data[start_index:end_index]

def get_frame(
        audio: torch.Tensor,
        frame_start: int,
        frame_end: int,
        sample_rate: int = SAMPLE_RATE,
        return_timestamps: bool = False,        
):
    f"""
    Slice a frame from the audio tensor indicated by the sample indices `frame_start` and `frame_end`.
    If `return_timestamps=True`, instead return a dict with keys `start_s` (start time in seconds),
    `end_s` (end time in seconds) and `samples` (tensor of wav samples for the given frame).
    Pass `sample_rate` to override the default sample rate of {SAMPLE_RATE}.
    """
    if return_timestamps:
        frame_start_s = frame_start/sample_rate
        frame_end_s = frame_end/sample_rate
        return {
            'start_s': frame_start_s,
            'end_s': frame_end_s,
            'samples': audio[frame_start:frame_end]
        }
    return audio[frame_start:frame_end]

def get_sliding_window(
        audio: torch.Tensor,
        framelength_s: float,
        frameshift_s: float,
        sample_rate: int = SAMPLE_RATE,
        return_timestamps: bool = False,
    ):
    f"""
    Split audio tensor into a list of tensors, each corresponding to a frame of length `framelength_s`
    staggered by `frameshift_s`. If `return_timestamps=True`, return a list of dictionaries with keys `start_s`
    (start time in seconds), `end_s` (end time in seconds) and `samples` (tensor of wav samples for the given frame).
    Pass `sample_rate` to override the default sample rate of {SAMPLE_RATE}.
    """
    if len(audio)==0:
        return []
    framelength_samples = int(framelength_s * sample_rate)
    frameshift_samples = int(frameshift_s * sample_rate)
    
    frame_start = 0
    frame_end = framelength_samples
    windows = []
    while frame_end<len(audio):
        frame = get_frame(audio, frame_start, frame_end, sample_rate, return_timestamps)
        windows.append(frame)
        frame_start+=frameshift_samples
        frame_end+=frameshift_samples
    # append last truncated frame
    frame = get_frame(audio, frame_start, len(audio), sample_rate, return_timestamps)
    windows.append(frame)
    
    return windows

# ------ #
# script #
# ------ #

def init_kws_parser():
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', help="Input audio files", nargs='+')
    parser.add_argument('--keyword_file', '-kf')
    parser.add_argument('--keywords', '-kws', nargs='+')
    parser.add_argument('--framelength_s', default=2)
    parser.add_argument('--frameshift_s', default=0.5)
    parser.add_argument('--sample_rate', default=SAMPLE_RATE)
    parser.add_argument('--encoder_size', default='tiny')
    parser.add_argument('--speech_encoder')
    parser.add_argument('--phone_encoder')
    parser.add_argument('--output_dir', '-o')
    parser.add_argument('--batch_size', '-b', type=int, default=32)

    return parser

def perform_kws(args):
    keyword_list = args.keywords
    if keyword_list is None:
        with open(args.keyword_file, encoding='utf8') as f:
            keyword_list = [line.strip() for line in f.readlines()]
    for audio in args.input:
        wav = load_and_resample(audio).squeeze()
        sliding_windows = get_sliding_window(
            wav,
            framelength_s=args.framelength_s,
            frameshift_s=args.frameshift_s,
            sample_rate=args.sample_rate,
            return_timestamps=True,
        )
        audio_frames = [frame.pop('samples') for frame in sliding_windows]
        sim_mat = []
        for batch in dataloader(audio_frames, batch_size=args.batch_size):
            batch_sim_mat = get_keyword_sim(
                audio_list=batch,
                text_list=keyword_list,
                speech_encoder=args.speech_encoder,
                phone_encoder=args.phone_encoder,
                encoder_size=args.encoder_size,
            )
            sim_mat.extend(batch_sim_mat.tolist())
        
        json_obj = {
            'audio_input': audio,
            'keywords': keyword_list,
            'timestamps': sliding_windows,
            'similarity_matrix': sim_mat
        }
        json_path = audio.replace('.wav', '.json')
        if args.output_dir:
            json_basename = os.path.basename(json_path)
            json_path = os.path.join(args.output_dir, json_basename)
        with open(json_path, 'w', encoding='utf8') as f:
            json.dump(json_obj, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = init_kws_parser()
    args = parser.parse_args(sys.argv[1:])
    perform_kws(args)
