from typing import Union, List, Literal, Sequence, Any, Generator, Tuple
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
from pympi import Praat
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


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

# ------------------------------ #
# eval helpers w Praat textgrids #
# ------------------------------ #

def textgrid_to_df(textgrid_path):
    tg = Praat.TextGrid(textgrid_path)
    rows = []
    for tier in tg.get_tiers():
        if 'word' not in tier.name:
            continue
        speaker = tier.name.split()[0]
        for start, end, val in tier.get_all_intervals():
            rows.append({
                'start': start,
                'end': end,
                'text': val,
                'speaker': speaker
            })
    return pd.DataFrame(rows)

def is_keyword_hit(df, keyword, timestamp):
    midpoints = (df['start']+df['end'])/2
    start_mask = midpoints>=timestamp['start_s']
    end_mask = midpoints<=timestamp['end_s']
    keyword_mask = df['text'].isin(keyword.split())
    return 1 if (start_mask & end_mask & keyword_mask).sum() > 0 else 0

def get_midpoints(timestamps):
    return [(timestamp['end_s']+timestamp['start_s'])/2 for timestamp in timestamps]

def timestamp_hits(df, keyword, timestamps):
    return np.array([is_keyword_hit(df, keyword, t) for t in timestamps])

def get_equal_error_rate(ground_truth, keyword_probs) -> Tuple[float, float]:
    fpr, tpr, thresholds=roc_curve(ground_truth, keyword_probs)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def max_in_window(sim_matrix: np.ndarray, start_s, end_s, framelength_s=2.0, frameshift_s=0.5):
    start_i = int(start_s//frameshift_s)
    end_i = int((end_s-framelength_s)//frameshift_s)
    sim_matrix_windowed = sim_matrix[start_i:end_i+1]
    max_sim_col = sim_matrix_windowed.max(axis=0)
    return max_sim_col

def get_windowed_sim_mat(sim_mat, chunks):
    max_probs = []
    for chunk in chunks:
        start = chunk['start_s']
        end = chunk['end_s']
        max_probs.append(max_in_window(sim_mat, start, end))
    max_probs=np.matrix(max_probs)
    max_probs=np.asarray(max_probs)
    return max_probs

def all_intervals_at_time_empty(df, time):
    start_mask = df['start']<=time
    end_mask = df['end']>=time
    intervals_at_time = df[start_mask&end_mask]
    text_vals = intervals_at_time['text'].unique()
    if len(text_vals)==1 and text_vals.item()=='':
        return True
    return False

def get_chunks_w_silent_edges(df, chunklen_s=10):
    chunk_start=0
    max_time = df['end'].max()
    chunk_timestamps = []
    while chunk_start<max_time:
        chunk_end = chunk_start + chunklen_s
        if chunk_end>max_time:
            break
        while not all_intervals_at_time_empty(df, chunk_end):
            if chunk_end>max_time:
                break
            chunk_end+=0.5
        chunk_timestamps.append({'start_s': chunk_start, 'end_s': chunk_end})
        chunk_start = chunk_end
    return chunk_timestamps

# ------ #
# script #
# ------ #

def init_kws_parser():
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', help="Input audio files", nargs='+')
    parser.add_argument('--keyword_file', '-kf')
    parser.add_argument('--keywords', '-kws', nargs='+')
    parser.add_argument('--inference_type', choices=['single_word', 'hmm'], default='single_word')
    parser.add_argument('--framelength_s', default=2)
    parser.add_argument('--frameshift_s', default=0.5)
    parser.add_argument('--sample_rate', default=SAMPLE_RATE)
    parser.add_argument('--encoder_size', default='tiny')
    parser.add_argument('--speech_encoder')
    parser.add_argument('--phone_encoder')
    parser.add_argument('--output_dir', '-o')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--textgrid', '-tg', help='Textgrid files with ground truth timestamps.', nargs='+')
    parser.add_argument('--eval_window', type=float)

    return parser

def perform_kws(args):
    keyword_list = args.keywords
    if keyword_list is None:
        with open(args.keyword_file, encoding='utf8') as f:
            keyword_list = [line.strip() for line in f.readlines()]
    audio_files = args.input
    textgrids = args.textgrid if args.textgrid else [None for _ in audio_files]
    for audio, textgrid in zip(audio_files, textgrids):
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
            'framelength_s': args.framelength_s,
            'frameshift_s': args.frameshift_s,
            'timestamps': sliding_windows,
            'similarity_matrix': sim_mat,
        }

        if textgrid:
            # textgrid file passed, use to perform evaluation
            tg_df = textgrid_to_df(textgrid)
            json_obj['eer']=[]
            if args.eval_window:
                json_obj['eval_window']=args.eval_window
            for i, keyword in tqdm(enumerate(keyword_list), desc="Calculating EER per keyword"):
                ground_truth = timestamp_hits(tg_df, keyword, sliding_windows)
                kw_probs = sim_mat[:,i]
                eer, thresh = get_equal_error_rate(ground_truth, kw_probs)
                json_obj['eer'].append({
                    'keyword': keyword,
                    'value': eer,
                    'eer_threshold': thresh,
                })
                if args.eval_window:
                    eval_windows = get_chunks_w_silent_edges(tg_df, chunklen_s=args.eval_window)
                    sim_mat_windowed = get_windowed_sim_mat(sim_mat, eval_windows)
                    for i, keyword in enumerate(tqdm(keyword_list)):
                        kw_probs_windowed = sim_mat_windowed[:,i]
                        ground_truth_windowed = timestamp_hits(tg_df, keyword, eval_windows)
                        eer_windowed, thresh_windowed = get_equal_error_rate(kw_probs_windowed, ground_truth_windowed)
                        json_obj['eer'][-1]['eer_eval_window']=eer_windowed
                        json_obj['eer'][-1]['eer_threshold_eval_window']=thresh_windowed
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
