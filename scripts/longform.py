from typing import Optional, Sequence, Dict, List, Union, Any, Tuple
from argparse import ArgumentParser, Namespace
from transformers import Pipeline, pipeline, WhisperTokenizer
import pandas as pd
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment
from eaf_to_script import write_script
import torch
import torchaudio
import numpy as np
from pympi import Elan
from glob import glob
import os
from tqdm import tqdm
from model_utils import load_whisper_pipeline
from tokenization_utils import get_forced_decoder_ids
from sli import infer_lr, load_lr
from copy import deepcopy

SAMPLE_RATE = 16000
DIARIZE_URI = "pyannote/speaker-diarization-3.1"
VAD_URI = "pyannote/segmentation-3.0"
ASR_URI = "openai/whisper-large-v3"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

"""
Copied from `annotate.py` in https://github.com/markjosims/montague_archiving on 10 Oct 2024
"""

# ------------------------------------- #
# Pyannote and HuggingFace entry points #
# ------------------------------------- #

def perform_asr(
        audio: Union[torch.Tensor, np.ndarray],
        pipe: Union[Pipeline, Dict[str, Pipeline], None] = None,
        model_path: str = ASR_URI,
        return_timestamps = True,
        generate_kwargs=None,
        sli_map=None,
        **kwargs,
) -> str:
    if sli_map:
        tokenizer=WhisperTokenizer.from_pretrained(model_path)
        audio=deepcopy(audio)
        if type(audio) is not list:
            raise ValueError('Must pass list of audio chunks if passing `sli_map`')
        if type(pipe) is not dict:
            pipe = load_asr_pipelines_for_sli(sli_map)
        for language_obj in sli_map:
            sli_label=language_obj['label']
            model_for_language=language_obj['whisper_checkpoint']
            chunks_with_language=[chunk for chunk in audio if chunk['sli_pred']==sli_label]
            language_code=language_obj['whisper_lang_code']
            generate_kwargs=generate_kwargs if generate_kwargs else {}
            generate_kwargs['forced_decoder_ids']=get_forced_decoder_ids(language=language_code, tokenizer=tokenizer)#['language']=language_code
            language_result = perform_asr(
                audio=chunks_with_language,
                pipe=pipe[model_for_language],
                generate_kwargs=generate_kwargs,
                return_timestamps=return_timestamps,
                **kwargs
            )
            for orig_chunk, result_chunk in zip(chunks_with_language, language_result):
                orig_chunk.update(**result_chunk)
        return audio
    if not pipe:
        pipe = pipeline("automatic-speech-recognition", model=model_path)
    if type(audio) is torch.Tensor:
        audio = audio[0,:].numpy()
    if type(audio) is list:
        if type(audio[0]['wav']) is torch.Tensor:
            for chunk in audio:
                chunk['wav']=chunk['wav'][0,:].numpy()
        audio = [chunk['wav'] for chunk in audio]
    result = pipe(
        audio,
        return_timestamps=return_timestamps,
        generate_kwargs=generate_kwargs,
        **kwargs,
    )
    return result

def load_asr_pipelines_for_sli(sli_map: Dict[str, Any]) -> Dict[str, Pipeline]:
    pipelines = {}
    for language_obj in sli_map:
        model_for_language = language_obj['whisper_checkpoint']
        peft_type = language_obj.get('peft_type', None)
        args = Namespace(model=model_for_language, peft_type=peft_type)
        if model_for_language not in pipelines:
            pipelines[model_for_language]=load_whisper_pipeline(args)
    return pipelines

def perform_vad(
        audio: torch.Tensor,
        pipe: Optional[PyannotePipeline] = None,
        vad_uri: str = VAD_URI,
        annotations: Dict[str, Any] = dict(),
        return_wav_slices: bool = False,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
):

    if not pipe:
        pipe = load_vad_pipeline(vad_uri, min_duration_on, min_duration_off)

    with ProgressHook() as hook:
        result = pipe(
            {"waveform": audio, "sample_rate": SAMPLE_RATE},
            hook=hook,
        )
    vad_chunks = [{'timestamp':(seg.start, seg.end)} for seg in result.itersegments()]
    if return_wav_slices:
        for chunk in vad_chunks:
            wav_slice=get_segment_slice(audio, chunk['timestamp'])
            chunk['wav']=wav_slice
    annotations['vad_chunks'] = vad_chunks
    return annotations

def load_vad_pipeline(vad_uri, min_duration_on, min_duration_off):
    pipe = VoiceActivityDetection(segmentation=vad_uri)
    hyperparams = {
            # remove speech regions shorter than that many seconds.
            "min_duration_on": min_duration_on,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": min_duration_off
        }
    pipe.instantiate(hyperparams)
    return pipe

def diarize(
        audio: torch.Tensor,
        pipe: Optional[PyannotePipeline] = None,
        num_speakers: int = 2,
        annotations: Dict[str, Any] = dict(),
        return_wav_slices: bool = False,
):

    if not pipe:
        pipe = PyannotePipeline.from_pretrained(DIARIZE_URI)

    with ProgressHook() as hook:
        result = pipe(
            {"waveform": audio, "sample_rate": SAMPLE_RATE},
            num_speakers=num_speakers,
            hook=hook,
        )
    speakers = result.labels()
    drz_chunks=[]
    for speaker in speakers:
        drz_chunks.extend([
            {'timestamp':(seg.start, seg.end), 'speaker':speaker}
            for seg in result.itersegments()
        ])
    annotations['drz_chunks']=drz_chunks
    if return_wav_slices:
        for chunk in drz_chunks:
            wav_slice=get_segment_slice(audio, chunk['timestamp'])
            chunk['wav']=wav_slice
    return annotations

def perform_sli(
        chunks: List[Dict[str, torch.Tensor]] = dict(),
        args: Optional[Namespace]=None,
        lr_model: Optional[str]=None,
        **kwargs
) -> Tuple[List[Dict[str, Any]], Namespace]:
    """
    Simple wrapper for `infer_lr` that always passes `dataset_type='chunk_list'`
    """
    return infer_lr(
        args=args,
        dataset=chunks,
        lr_model=lr_model,
        dataset_type='chunk_list',
        **kwargs
    )

# ------------ #
# ELAN methods #
# ------------ #

def get_ipa_labels(elan_fp: str) -> List[Dict[str, Union[str, float]]]:
    """
    Read data from IPA Transcription tier in a .eaf file
    indicated by `elan_fp`. Return list of dicts containing
    start time, end time and value for each annotation.
    """
    eaf = Elan.Eaf(elan_fp)
    ipa_tuples = eaf.get_annotation_data_for_tier('IPA Transcription')
    ipa_labels = [{'start': a[0], 'end': a[1], 'value': a[2]} for a in ipa_tuples]
    return ipa_labels
    
def change_file_suffix(media_fp: str, ext: str, tgt_dir: Optional[str]=None) -> str:
    media_suff = os.path.splitext(media_fp)[-1]
    wav_fp = media_fp.replace(media_suff, ext)
    if tgt_dir:
        basename = os.path.basename(wav_fp)
        return os.path.join(tgt_dir, basename)
    return wav_fp

# ---------------------- #
# Audio handling methods #
# ---------------------- #

def load_and_resample(
        fp: str,
        sr: int = SAMPLE_RATE,
        to_mono: bool = True,
        flatten: bool = False,
    ) -> torch.Tensor:
    f"""
    Load a wavfile at filepath `fp` into a torch tensor.
    Resample to `sr` (default {SAMPLE_RATE}).
    If `to_mono` is passed, convert to mono by dropping the second channel.
    If `flatten` is also passed, squeeze.
    """
    wav_orig, sr_orig = torchaudio.load(fp)
    wav = torchaudio.functional.resample(wav_orig, sr_orig, sr)
    if to_mono and len(wav.shape)==2:
        tqdm.write("Converting stereo wav to mono")
        wav=wav[:1,:]
        if flatten:
            wav=wav.squeeze()
    elif flatten:
        raise ValueError("Cannot flatten wav unless converting to mono!")
    return wav

def sec_to_samples(time_sec: float) -> int:
    """`time_sec` is a time value in seconds.
    Returns same time value in samples using
    global constant SAMPLE_RATE.
    """
    return int(time_sec*SAMPLE_RATE)

def sec_to_ms(time_sec: float) -> int:
    return int(time_sec*1000)

def get_segment_slice(
        audio: torch.Tensor,
        segment: Union[Segment, Tuple[float, float]],
) -> np.ndarray:
    """
    Takes torchaudio tensor and a pyannote segment,
    returns slice of tensor corresponding to segment endpoints.
    """
    if type(segment) is Segment:
        start_sec = segment.start
        end_sec = segment.end
    else: # type(segment) is tuple
        start_sec, end_sec = segment
    start_idx = sec_to_samples(start_sec)
    end_idx = sec_to_samples(end_sec)
    return audio[:,start_idx:end_idx]

def fix_whisper_timestamps(start: float, end: float, wav: torch.Tensor):
    if end is None:
        # whisper may not predict an end timestamp for the last chunk in the recording
        end = len(wav[0])/SAMPLE_RATE
    if end<=start:
        # Whisper may predict 0 length for short speech turns
        # default to setting length of 200ms
        end=start+200
    return start, end

# ------------ #
# Elan helpers #
# ------------ #

def pipeout_to_eaf(
        chunk_list: List[Dict[str, Any]],
        chunk_key: str = 'text',
        tier_name: str = 'asr',
        eaf: Elan.Eaf = Elan.Eaf(),
        label: Optional[str] = None,
) -> Elan.Eaf:
    eaf.add_tier(tier_name)
    for chunk in chunk_list:
        start_ms = sec_to_ms(chunk['timestamp'][0])
        end_ms = sec_to_ms(chunk['timestamp'][1])
        val = label if label else chunk[chunk_key]
        eaf.add_annotation(tier_name, start_ms, end_ms, val)
    return eaf

def pipeout_to_df(
        chunk_list: List[Dict[str, Any]],
        chunk_key: str = 'text',
        tier_name: str = 'asr',
        df: Optional[pd.DataFrame] = None,
        label: Optional[str] = None,
        eaf_path: Optional[str] = None,
        wav_source: Optional[str] = None,
):
    annotations = []
    for chunk in chunk_list:
        chunk_annotations = {
            'start': sec_to_ms(chunk['timestamp'][0]),
            'end': sec_to_ms(chunk['timestamp'][1]),
            'tier_name': tier_name,
            'transcription': label if label else chunk[chunk_key],
        }
        if eaf_path:
            chunk_annotations['eaf_path'] = eaf_path
        if wav_source:
            chunk_annotations['wav_source'] = wav_source
        annotations.append(chunk_annotations)
    out_df = pd.DataFrame(annotations)
    if df:
        return pd.concat([df, out_df])
    return out_df

# ---- #
# main #
# ---- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser("Annotation runner")
    parser.add_argument("-i", "--input", help=".wav file or directory of .wav files to annotate")
    parser.add_argument("-o", "--output", help="directory files to save output to")
    parser.add_argument(
        "-m", "--model",
        help=f"ASR model path. Default is {ASR_URI}.",
        default=ASR_URI,
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to load model checkpoint from, if different from model path."
    )
    parser.add_argument(
        "--peft",
        action="store_true"
    )
    parser.add_argument(
        "-z", "--drz_uri",
        help=f"DRZ model path. Default is {DIARIZE_URI}.",
        default=DIARIZE_URI,
    )
    parser.add_argument(
        '-v', '--vad_uri',
        help=f"VAD model path. Default is {VAD_URI}.",
        default=VAD_URI,
    )
    parser.add_argument(
        '--min_duration_on', type=float, default=0.0,
    )
    parser.add_argument(
        '--min_duration_off', type=float, default=0.0,
    )
    parser.add_argument(
        "--lr_model",
        "--lr",
    )
    parser.add_argument(
        "-D", "--device",
        default=DEVICE,
        help=f"Device to run model on. Default {torch.device(DEVICE)}",
    )
    parser.add_argument(
        "-c",
        "--chunk_length_s",
        type=float,
        help="Chunk size to input to Whisper pipeline (default 30s). "\
        +"Note that this will not affect the size of chunks OUTPUT by Whisper, "\
        +"so this should only be modified if there is not enough memory to process "\
        +"30s at a time.",
        default=30,
    )
    parser.add_argument(
        "-w",
        "--return_word_timestamps",
        action='store_true',
        help="Whisper by default chunks speech more or less into utterances. "\
        +"Use this option to chunk by word, which may give more precise time accuracy "\
        "When detecting speaker changes."
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="If input is a directory, will search for files .wav files recursively"
    )
    parser.add_argument(
        '-b',
        "--batch_size",
        type=int, default=8,
        help="Inference batch size for ASR. Default 8."
    )
    parser.add_argument(
        '--file_extension', '-x', default='.mp3',
    )
    parser.add_argument(
        '--sb_savedir', default='models/speechbrain',
    )
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    return annotate(args)


def annotate(args) -> int:
    """
    For now assuming a VAD>SLI>ASR pipeline
    TODO: Implement other pipelines later
    """
    wav_paths = glob(os.path.join(args.input, '*.wav'))
    vad_pipe = load_vad_pipeline(args.vad_uri, args.min_duration_on, args.min_duration_off)
    _, args = load_lr(args=args)
    asr_pipelines = load_asr_pipelines_for_sli(args.sli_map)
    df = pd.DataFrame(columns=['wav_path', 'tier_name', 'start', 'end', 'transcription'])
    for wav_path in tqdm(wav_paths):
        wav = load_and_resample(wav_path)
        vad_out = perform_vad(wav, pipe=vad_pipe, return_wav_slices=True)
        vad_chunks = vad_out['vad_chunks']
        sli_chunks, _ = perform_sli(vad_chunks, args=args)
        asr_out = perform_asr(
            sli_chunks,
            pipe=asr_pipelines,
            return_timestamps=args.return_word_timestamps,
            sli_map=args.sli_map,
        )
        eaf=pipeout_to_eaf(asr_out, tier_name='asr')
        eaf_path = change_file_suffix(wav_path, '.eaf', tgt_dir=args.output)
        eaf.to_file(eaf_path)
        df=pipeout_to_df(asr_out, tier_name='asr', df=df, wav_source=wav_path, eaf_path=eaf_path)
    df.to_csv(os.path.join(args.output, 'metadata.csv'))
    return 0


if __name__ == '__main__':
    main()