from glob import glob
from pympi import Elan
from typing import Optional, Sequence
from argparse import ArgumentParser
import pandas as pd
import os
import librosa
import soundfile
from tqdm import tqdm

GDRIVE_DIR = '/Users/markjos/Library/CloudStorage/GoogleDrive-mjsimmons@ucsd.edu/Shared drives/Tira/Recordings'
tqdm.pandas()

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', default=GDRIVE_DIR)
    parser.add_argument('--output', '-o')
    parser.add_argument('--logfile', '-l', default='error.txt')
    parser.set_defaults(func=empty_command)

    commands=parser.add_subparsers(help='Command to run')

    pool_eaf_data_parser=commands.add_parser('pool_eaf_data')
    pool_eaf_data_parser.add_argument('--transcription_tier', '-t', default='IPA Transcription')
    pool_eaf_data_parser.set_defaults(func=pool_eaf_data)

    make_clips_parser=commands.add_parser('make_clips')
    make_clips_parser.set_defaults(func=make_clips)
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
        wav,
        start_ms,
        end_ms,
        wav_basename: str,
        target_dir: str,
        sampling_rate: int=16_000
    ):
    start_samples=ms_to_samples(start_ms, sampling_rate=sampling_rate)
    end_samples=ms_to_samples(end_ms, sampling_rate=sampling_rate)
    wav_clip = wav[start_samples:end_samples]

    start_timestr=ms_to_humantime(start_ms)
    end_timestr=ms_to_humantime(end_ms)

    time_suffix = f"-{start_timestr}-{end_timestr}.wav"
    clip_basename = wav_basename.removesuffix('.wav')+time_suffix
    clip_fp = os.path.join(
        target_dir,
        clip_basename
    )

    soundfile.write(clip_fp, wav_clip, samplerate=sampling_rate)
    return clip_fp

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
    df.to_csv(args.output, index=False)
    return 0

def make_clips(args) -> int:
    df=pd.read_csv(args.input)
    df['clip']=''
    output_dir=args.output if os.path.isdir(args.output) else os.path.dirname(args.output)
    csv_path=args.output if args.output.endswith('.csv') else os.path.join(output_dir, 'clipdata.csv')
    clip_dir=os.path.join(output_dir, 'clips')
    os.makedirs(clip_dir, exist_ok=True)
    for wav_source in tqdm(df['wav_source'].unique()):
        is_wav_source=df['wav_source']==wav_source
        try:
            wav, sampling_rate=librosa.load(wav_source, sr=16_000, mono=True)
        except:
            with open(args.logfile, 'a') as f:
                f.write("Could not open wav_source "+wav_source)
            continue
        df.loc[is_wav_source, 'clip']=df.loc[is_wav_source].progress_apply(
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

# ---- #
# main #
# ---- #

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == '__main__':
    main()