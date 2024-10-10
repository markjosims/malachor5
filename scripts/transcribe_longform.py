from typing import Optional, Sequence, Dict, List, Union
from argparse import ArgumentParser
from transformers import Pipeline, pipeline, WhisperTokenizer
from pyannote.audio import Pipeline as PyannotePipeline
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

SAMPLE_RATE = 16000
DIARIZE_URI = "pyannote/speaker-diarization-3.1"
ASR_URI = "openai/whisper-large-v2"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

"""
Copied from `annotate.py` in https://github.com/markjosims/montague_archiving on 10 Oct 2024
"""

# ------------------------------------- #
# Pyannote and HuggingFace entry points #
# ------------------------------------- #

def perform_asr(
        audio: Union[torch.Tensor, np.ndarray],
        pipe: Optional[Pipeline] = None,
        **kwargs,
    ) -> str:
    if not pipe:
        pipe = pipeline("automatic-speech-recognition", model=ASR_URI)
    if type(audio) is torch.Tensor:
        audio = np.array(audio[0,:])
    result = pipe(audio,**kwargs)
    return result

def diarize(
        audio: torch.Tensor,
        pipe: Optional[PyannotePipeline] = None,
        num_speakers: int = 2,
    ):

    if not pipe:
        pipe = PyannotePipeline.from_pretrained(DIARIZE_URI)

    with ProgressHook() as hook:
        result = pipe(
            {"waveform": audio, "sample_rate": SAMPLE_RATE},
            num_speakers=num_speakers,
            hook=hook,
        )
    return result


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
    
def change_file_suffix(media_fp: str, ext: str) -> str:
    media_suff = os.path.splitext(media_fp)[-1]
    wav_fp = media_fp.replace(media_suff, ext)
    return wav_fp

# ---------------------- #
# Audio handling methods #
# ---------------------- #

def load_and_resample(fp: str, sr: int = SAMPLE_RATE) -> torch.Tensor:
    wav_orig, sr_orig = torchaudio.load(fp)
    wav = torchaudio.functional.resample(wav_orig, sr_orig, sr)
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
        segment,
    ) -> np.ndarray:
    """
    Takes torchaudio tensor and a pyannote segment,
    returns slice of tensor corresponding to segment endpoints.
    """
    start_idx = sec_to_samples(segment.start)
    end_idx = sec_to_samples(segment.end)
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

# ---- #
# main #
# ---- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser("Annotation runner")
    parser.add_argument("-i", "--input", help=".wav file or directory of .wav files to annotate")
    parser.add_argument(
        "-s",
        "--strategy",
        choices=["asr-only", "drz-only", "asr-first", "drz-first", "multitier"],
        default="asr-first",
        help="Specify what pipeline to use for annotation. "\
        +"`asr-first` (default) will run Whisper first then diarization with PyAnnote, "\
        +"and the PyAnnote diarization will be used to decide the speaker identity"\
        +"for each chunk output by Whisper. "\
        +"`drz-only` will run diarization with PyAnnote but not transcribe any audio."\
        +"`asr-only` will run Whisper without performing speaker diarization. "\
        +"`drz-first` will run diarization with PyAnnote first and then send each "\
        +"speaker turn to Whisper for transcription. This will likely result in poor "\
        +"turn boundaries, so `asr-first` should generally be preferred. "\
        +"`multitier` will output an .eaf file with Whisper annotations and PyAnnote "\
        +"speaker turns on separate tiers, and is mostly useful for debugging and understanding "\
        +"the decisions that ASR and diarization are making independently."
    )
    parser.add_argument(
        "-n",
        "--num_speakers",
        help="Number of speakers in file. Default 2.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-m", "--asr_model",
        help=f"ASR model path. Default is {ASR_URI}.",
        default=ASR_URI,
    )
    parser.add_argument(
        "-d", "--drz_model",
        help=f"DRZ model path. Default is {DIARIZE_URI}.",
        default=DIARIZE_URI,
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
        "--asr_batch_size",
        type=int, default=8,
        help="Inference batch size for ASR. Default 8."
    )
    parser.add_argument(
        '--file_extension', '-x', default='.mp3',
    )
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    return annotate(args)


def annotate(args) -> int:
    print(f"Initializing ASR pipeline from URI {args.asr_model}...")
    if args.strategy != "drz-only":
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=args.asr_model,
            device=args.device,
            chunk_length_s=args.chunk_length_s,
        )
        tokenizer = WhisperTokenizer.from_pretrained(args.asr_model)
        forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="english", task="transcribe")
    else:
        asr_pipe=None
        tokenizer=None
        forced_decoder_ids=None
    if args.strategy != "asr-only":
        print(f"Initializing diarization pipeline from URI {args.drz_model}...")
        drz_pipe = PyannotePipeline.from_pretrained(args.drz_model)
        drz_pipe.to(torch.device(args.device))
    else:
        drz_pipe=None

    if os.path.isdir(args.input):
        if args.recursive:
            glob_str = os.path.join(args.input, "**", "*" + args.file_extension.lower())
        else:
            glob_str = os.path.join(args.input, "*" + args.file_extension.lower())
        # search for both lower and upper cased extensions
        glob_str_upper = glob_str.replace(args.file_extension, args.file_extension.upper())
        audio_fps = glob(glob_str, recursive=args.recursive) + glob(glob_str_upper, recursive=args.recursive)
        for audio_fp in audio_fps:
            annotate_file(
                args,
                asr_pipe,
                drz_pipe,
                audio_fp,
                generate_kwargs={'forced_decoder_ids': forced_decoder_ids}
            )
        return 0

    annotate_file(
        args,
        asr_pipe,
        drz_pipe,
        args.input,
        generate_kwargs={'forced_decoder_ids': forced_decoder_ids}
    )
    return 0

def annotate_file(args, asr_pipe, drz_pipe, audio_fp, generate_kwargs):
    print("Annotating file", audio_fp)
    eaf = Elan.Eaf()

    # ELAN does not accept .mp3 media files
    wav_fp = change_file_suffix(audio_fp, '.wav')
    eaf.add_linked_file(wav_fp)
    wav = load_and_resample(audio_fp)
    if args.strategy=='drz-first':
        eaf = drz_first(
                wav=wav,
                eaf=eaf,
                num_speakers=args.num_speakers,
                drz_pipe=drz_pipe,
                asr_pipe=asr_pipe,
                generate_kwargs=generate_kwargs,
            )
    elif args.strategy=='drz-only':
        eaf = drz_only(
            wav=wav,
            eaf=eaf,
            num_speakers=args.num_speakers,
            drz_pipe=drz_pipe,
        )
    elif args.strategy=='asr-first':
        eaf = asr_first(
                wav=wav,
                eaf=eaf,
                num_speakers=args.num_speakers,
                drz_pipe=drz_pipe,
                asr_pipe=asr_pipe,
                generate_kwargs=generate_kwargs,
                return_timestamps='word' if args.return_word_timestamps else True,
            )
    elif args.strategy=='multitier':
        eaf = multitier(
                wav=wav,
                eaf=eaf,
                num_speakers=args.num_speakers,
                drz_pipe=drz_pipe,
                asr_pipe=asr_pipe,
                generate_kwargs=generate_kwargs,
                return_timestamps='word' if args.return_word_timestamps else True,
            )
    else:
        eaf = asr_only(
                wav=wav,
                eaf=eaf,
                asr_pipe=asr_pipe,
                generate_kwargs=generate_kwargs,
            )

    eaf_fp = change_file_suffix(audio_fp, '.eaf')
    eaf.to_file(eaf_fp)
    txt_fp = change_file_suffix(audio_fp, '.txt')
    write_script(
        eaf,
        txt_fp,
        merge_turns=args.strategy!='asr-only',
        keep_line_breaks=args.return_word_timestamps,
    )

    print("Saved ELAN annotations to", eaf_fp)
    print("Saved text annotations to", txt_fp)

def asr_only(
        wav: torch.Tensor,
        eaf: Elan.Eaf,
        asr_pipe: Pipeline,
        **kwargs,
    ):
    chunks = perform_asr(wav, pipe=asr_pipe, return_timestamps=True, **kwargs)["chunks"]
    for chunk in chunks:
        start, end = chunk['timestamp']
        start, end = fix_whisper_timestamps(start, end, wav)

        text = chunk['text']
        eaf.add_annotation("default", sec_to_ms(start), sec_to_ms(end), text)
    return eaf

def drz_only(
        wav: torch.Tensor,
        eaf: Elan.Eaf,
        num_speakers: int,
        drz_pipe: PyannotePipeline,
    ):

    diarization = diarize(wav, drz_pipe, num_speakers=num_speakers)
    speakers = diarization.labels()
    for speaker in speakers:
        eaf.add_tier(speaker)
        speaker_timeline = diarization.label_timeline(speaker)
        for segment in speaker_timeline:
            start_ms = sec_to_ms(segment.start)
            end_ms = sec_to_ms(segment.end)
            eaf.add_annotation(speaker, start_ms, end_ms)
    return eaf        

def asr_first(
        wav: torch.Tensor,
        eaf: Elan.Eaf,
        num_speakers: int,
        drz_pipe: PyannotePipeline,
        asr_pipe: Pipeline,
        **kwargs,
    ):
    chunks = perform_asr(wav, pipe=asr_pipe, **kwargs)["chunks"]
    diarization = diarize(wav, drz_pipe, num_speakers=num_speakers)

    speakers = diarization.labels()
    for speaker in speakers:
        eaf.add_tier(speaker)

    for chunk in chunks:
        start, end = chunk['timestamp']
        start, end = fix_whisper_timestamps(start, end, wav)

        text = chunk['text']
        speaker = diarization.argmax(Segment(start, end))
        if not speaker:
            speaker='default'
        eaf.add_annotation(speaker, sec_to_ms(start), sec_to_ms(end), text)
    return eaf

def multitier(
        wav: torch.Tensor,
        eaf: Elan.Eaf,
        num_speakers: int,
        drz_pipe: PyannotePipeline,
        asr_pipe: Pipeline,
        **kwargs,
    ):
    eaf = asr_only(wav, eaf, asr_pipe, **kwargs)
    eaf.rename_tier('default', 'asr')
    eaf = drz_only(wav, eaf, num_speakers, drz_pipe)
    return eaf

def drz_first(
        wav: torch.Tensor,
        eaf: Elan.Eaf,
        num_speakers: int,
        drz_pipe: PyannotePipeline,
        asr_pipe: Pipeline,
        **kwargs,
    ):

    diarization = diarize(wav, drz_pipe, num_speakers=num_speakers)

    speakers = diarization.labels()
    for speaker in speakers:
        eaf.add_tier(speaker)
        speaker_timeline = diarization.label_timeline(speaker)
        for segment in tqdm(
                speaker_timeline,
                desc=f"Performing ASR for speaker {speaker}",
                total=len(list(speaker_timeline
            ))):
            segment_wav = get_segment_slice(wav, segment)
            segment_text = perform_asr(segment_wav, asr_pipe, **kwargs)['text']
            start_ms = sec_to_ms(segment.start)
            end_ms = sec_to_ms(segment.end)
            eaf.add_annotation(speaker, start_ms, end_ms, segment_text)
    
    return eaf


if __name__ == '__main__':
    main()