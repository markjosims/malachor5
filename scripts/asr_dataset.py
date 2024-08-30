from glob import glob
from pympi import Elan
from typing import Optional, Sequence
from argparse import ArgumentParser
import pandas as pd
import os
import librosa
import numpy as np
import soundfile
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Audio
from transformers import pipeline
from pyannote.audio import Pipeline as pyannote_pipeline
import torch

GDRIVE_DIR = '/Users/markjos/Library/CloudStorage/GoogleDrive-mjsimmons@ucsd.edu/Shared drives/Tira/Recordings'
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
tqdm.pandas()

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', default=GDRIVE_DIR)
    parser.add_argument('--output', '-o')
    parser.add_argument('--logfile', '-l', default='error.txt')
    parser.set_defaults(func=empty_command)

    commands=parser.add_subparsers(help='Command to run')

    pool_eaf_data_parser=commands.add_parser(
        'pool_eaf_data',
        help="Glob all `.eaf` files in input directory recursively and create `.csv` file "+\
        "containing rows for each interval with columns `start`, `end`, `transcription`, "+\
        "`eaf_source` and `wav_source.` --output may be path to `.csv` file or directory to "+\
        "create `metadata.csv` in."
    )
    pool_eaf_data_parser.add_argument('--transcription_tier', '-t', default='IPA Transcription')
    pool_eaf_data_parser.set_defaults(func=pool_eaf_data)

    make_clips_parser=commands.add_parser(
        'make_clips',
        help=make_clips.__doc__
    )
    make_clips_parser.add_argument('--check_clips_exist', action='store_true')
    make_clips_parser.set_defaults(func=make_clips)

    hf_dataset_parser=commands.add_parser(
        'make_hf_dataset',
        help=make_hf_dataset.__doc__
    )
    hf_dataset_parser.set_defaults(func=make_hf_dataset)

    remove_clips_parser=commands.add_parser(
        'remove_extra_clips',
        help=remove_extra_clips.__doc__
    )
    remove_clips_parser.set_defaults(func=remove_extra_clips)

    infer_asr_parser = commands.add_parser('infer_asr', help=infer_asr.__doc__)
    infer_asr_parser.add_argument('--model', '-m', default='openai/whisper-large-v3')
    infer_asr_parser.add_argument(
        '--device',
        '-D',
        type=lambda s: int(s) if s!='cpu' else s,
        default=DEVICE
    )
    infer_asr_parser.add_argument('--batch_size', '-b', type=int, default=32)
    infer_asr_parser.set_defaults(func=infer_asr)

    infer_vad_parser = commands.add_parser('infer_vad', help=infer_vad.__doc__)
    infer_vad_parser.add_argument('--model', '-m', default='pyannote/speaker-diarization-3.1')
    infer_vad_parser.add_argument(
        '--device',
        '-D',
        type=lambda s: int(s) if s!='cpu' else s,
        default=DEVICE
    )
    infer_vad_parser.set_defaults(func=infer_vad)

    return parser

# -------------- #
# Helper methods #
# -------------- #

def get_media_path(eaf):
    media_paths = [x['MEDIA_URL'] for x in eaf.media_descriptors]
    media = media_paths[0]
    # trim prefix added by ELAN
    # have to keep initial / on posix systems
    # and remove on Windows
    if os.name == 'nt':
        media = media.replace('file:///', '')
    else:
        media = media.replace('file://', '')
    # computers why must you be so silly
    return media

def samples_to_ms(samples, sampling_rate=16_000) -> int:
    seconds = samples/sampling_rate
    ms=int(seconds*1_000)
    return ms

def ms_to_samples(ms, sampling_rate=16_000) -> int:
    seconds = ms/1_000
    samples=int(seconds*sampling_rate)
    return samples

def ms_to_humantime(ms: int) -> str:
    timestr = ''
    hours = ms//3_600_000
    if hours:
        timestr=f'h{hours:02}'
    minutes = (ms%3_600_000)//60_000
    timestr=timestr+f'm{minutes:02}'
    seconds = (ms%60_000)//1_000
    ms_remain = ms%1000
    timestr=timestr+f's{seconds:02}ms{ms_remain:03}'
    return timestr

def clip_segment(
        wav: Optional[np.ndarray],
        start_ms: int,
        end_ms: int,
        wav_basename: str,
        target_dir: str,
        sampling_rate: int=16_000,
    ) -> str:
    """
    Generate filepath from `wav_basename` corresponding to the start
    and end timestamps specified by `start_ms` and `end_ms`.
    If `wav` argument is provided, clip to respective timestamps
    and save to filepath. Either way, return filepath as str.
    """
    start_timestr=ms_to_humantime(start_ms)
    end_timestr=ms_to_humantime(end_ms)

    time_suffix = f"-{start_timestr}-{end_timestr}.wav"
    clip_basename = wav_basename.removesuffix('.wav')+time_suffix
    clip_fp = os.path.join(
        target_dir,
        clip_basename
    )

    if wav is not None:
        start_samples=ms_to_samples(start_ms, sampling_rate=sampling_rate)
        end_samples=ms_to_samples(end_ms, sampling_rate=sampling_rate)
        wav_clip = wav[start_samples:end_samples]
        soundfile.write(clip_fp, wav_clip, samplerate=sampling_rate)

    return clip_fp

def check_clips_exist(is_source: pd.Series, wav_source: str, clip_dir: str) -> bool:
    num_source = is_source.value_counts()[True]
    source_basename = os.path.basename(wav_source)
    glob_basename = source_basename.removesuffix('.wav')+'*.wav'
    glob_str = os.path.join(clip_dir, glob_basename)
    clip_paths = glob(glob_str)
    return len(clip_paths) == num_source

def infer_asr(args) -> int:
    """
    Run ASR using HF pipeline on `audio` column in input dataset.
    Save output csv with results in column named after model checkpoint specified.
    """
    ds = load_from_disk(args.input)
    pipe=pipeline('automatic-speech-recognition', args.model, device=args.device)
    def map_pipe(row):
        result = pipe([audio['array'] for audio in row['audio']])
        out={}
        model_col = args.model.split(sep='/')[-1]
        out[model_col] = [item['text'] for item in result]
        out['path'] = row['audio']['path']
        return out
    ds=ds.map(map_pipe, batched=True, batch_size=args.batch_size, remove_columns=ds.column_names)
    ds.to_csv(args.output)
    return 0

def infer_vad(args) -> int:
    """
    Run VAD using PyAnnote speaker diarization set to detect one speaker.
    Add column indicating number of ms of detected speech.
    """
    ds = load_from_disk(args.input)
    pipe=pyannote_pipeline.from_pretrained(args.model)
    pipe.to(torch.device(args.device))
    def map_pipe(row):
        result = pipe(
            {
                'waveform': torch.tensor(row['audio']['array']).unsqueeze(0).to(args.device).float(),
                'sample_rate': row['audio']['sampling_rate'],
            },
            num_speakers=1,
        )
        out={}
        # item.chart() returns list of shape [('SPEAKER_00', num_sec)]
        model_col = args.model.split(sep='/')[-1]
        out[model_col]=result.to_lab().replace('\n', ';')
        out['path'] = row['audio']['path']
        return out
    ds=ds.map(map_pipe, remove_columns=ds.column_names)
    ds.to_csv(args.output)
    return 0

# --------------- #
# Command methods #
# --------------- #

def empty_command(args) -> int:
    print("Please specify a command.")
    return 1

def pool_eaf_data(args) -> int:
    eafs = glob(os.path.join(args.input, '**/*.eaf'), recursive=True)
    start, end, val, eaf_source, wav_source = [], [], [], [], []
    for eaf_fp in eafs:
        eaf=Elan.Eaf(eaf_fp)
        tiers=eaf.get_tier_names()
        if args.transcription_tier not in tiers:
            print(f"{eaf_fp} missing tier {args.transcription_tier}, skipping")
            continue
        intervals=eaf.get_annotation_data_for_tier(args.transcription_tier)
        if not intervals:
            continue
        eaf_start, eaf_end, eaf_val = zip(*intervals)
        start.extend(eaf_start)
        end.extend(eaf_end)
        val.extend(eaf_val)

        wav_fp = get_media_path(eaf)
        eaf_source.extend(eaf_fp for _ in intervals)
        wav_source.extend(wav_fp for _ in intervals)


    df = pd.DataFrame({
        'start':start,
        'end':end,
        'transcription':val,
        'eaf_source': eaf_source,
        'wav_source': wav_source,
    })
    csv_path = args.output
    if os.path.isdir(csv_path):
        csv_path=os.path.join(csv_path, 'metadata.csv')
    df.to_csv(csv_path, index=False)
    return 0

def make_clips(args) -> int:
    """
    Open input `.csv` file, create `clips` dir in output dir which is populated with
    `.wav` files for each interval in the input csv and add column with relative path of clip.
    If --output is csv file, save to output path, else if output is dir, save to `clipdata.csv`
    in output dir.
    """
    df=pd.read_csv(args.input)
    # choosing colname `file_name` bc that's what HF AudioFolder expects
    df['file_name']=''
    output_dir=args.output if os.path.isdir(args.output) else os.path.dirname(args.output)
    csv_path=args.output if args.output.endswith('.csv') else os.path.join(output_dir, 'clipdata.csv')
    clip_dir=os.path.join(output_dir, 'clips')
    os.makedirs(clip_dir, exist_ok=True)
    for wav_source in tqdm(df['wav_source'].unique()):
        is_wav_source=df['wav_source']==wav_source
        if args.check_clips_exist and check_clips_exist(is_wav_source, wav_source, clip_dir):
            # if clips already exist, skip loading in wav but still save clip filepaths
            wav, sampling_rate = None, None
        else:
            try:
                wav, sampling_rate=librosa.load(wav_source, sr=16_000, mono=True)
            except:
                with open(args.logfile, 'a') as f:
                    f.write("Could not open wav_source "+wav_source)
                continue
        df.loc[is_wav_source, 'file_name']=df.loc[is_wav_source].progress_apply(
            lambda row: clip_segment(
                wav=wav,
                start_ms=row['start'],
                end_ms=row['end'],
                wav_basename=os.path.basename(wav_source),
                target_dir=clip_dir,
                sampling_rate=sampling_rate,
            ),
            axis=1
        )
    df.to_csv(csv_path, index=False)
    return 0

def make_hf_dataset(args) -> int:
    """
    Create an AudioFolder dataset from input dir and saves to output dir.
    Expects input dir to contain `metadata.csv` with column `clip` or `file_name.`
    """
    metadata_path=os.path.join(args.input, 'metadata.csv')
    df = pd.read_csv(metadata_path)
    if 'file_name' not in df.columns:
        # assume csv has column 'clip' which includes absolute path
        # or path relative from project directory for each clip
        # add new column 'file_name' which is relative to dataset directory
        df['file_name']=df['clip'].apply(
            lambda s: os.path.join(
                os.path.relpath(s, args.input)
            )
        )
        df.to_csv(metadata_path, index=False)
    del df
    ds = load_dataset("audiofolder", data_dir=args.input)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds.save_to_disk(args.output)
    return 0

def remove_extra_clips(args) -> int:
    """
    Looks for any clips not found in input `.csv` file and deletes from `clips` dir.
    """
    df = pd.read_csv(args.input)
    data_dir = os.path.dirname(args.input)
    clip_dir = args.output or os.path.join(
        data_dir,
        'clips',
    )
    clip_wavs = glob(os.path.join(clip_dir, '*.wav'))
    clip_wavs = [os.path.relpath(path, data_dir) for path in clip_wavs]
    df_clip_wavs = df['clip'].apply(lambda path: os.path.relpath(path, data_dir))
    for clip_wav in tqdm(clip_wavs):
        if clip_wav not in df_clip_wavs:
            os.remove(clip_wav)

# ---- #
# main #
# ---- #

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == '__main__':
    main()