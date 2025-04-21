from datasets import load_from_disk, Dataset, DatasetDict, Audio
from typing import Optional, Sequence, Dict, Any
from argparse import ArgumentParser
import pandas as pd
import torchaudio
import torch
import os

# --------------- #
# dataset methods #
# --------------- #
def row_to_wav(row: Dict[str, Any], outdir: str) -> None:
    """
    Reads samples a Dataset Audio row (in the form of a dictionary)
    and save as a wav in the specified `outdir`.
    """
    basepath = row['audio']['path']
    path = os.path.join(outdir, basepath)
    wav_array = row['audio']['array']
    sr = row['audio']['sampling_rate']
    # cast to torch tensor and make 2D
    wav_tensor = torch.Tensor(wav_array).unsqueeze(0)
    torchaudio.save(path, wav_tensor, sr)

def audio_as_path(row: Dict[str, Any], outdir: str) -> Dict[str, str]:
    """
    Get basename for audio path from Dataset Audio row (in the form of a dictionary),
    join basename with `outdir` and return as a dict.
    Allows audio paths to be moved to a new directory for an entire Dataset using e.g.
    `Dataset.map`
    """
    basepath = row['audio']['path']
    path = os.path.join(outdir, basepath)
    return {'path': path}

# --------------- #
# command methods #
# --------------- #

def empty_command(args) -> int:
    print("Please specifiy a command.")
    return 1

def arrow_to_wav(args) -> int:
    """
    Load in a HugggingFace Audio Dataset and save a directory of wavs at `args.outdir`.
    """
    outdir = args.outdir
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    ds = load_from_disk(args.input)
    print(f"Saving audio dataset ")
    # ds.map(lambda row: row_to_wav(row, outdir))

    ds = ds.map(lambda row: audio_as_path(row, outdir), remove_columns=['audio'])
    if type(ds) is DatasetDict:
        for splitname, split in ds.items():
            csv_path = os.path.join(outdir, f'{splitname}_metadata.csv')
            df=split.to_pandas()
            df.to_csv(csv_path, index=False)
    else:
        csv_path = os.path.join(outdir, 'metadata.csv')
        df=split.to_pandas()
        df.to_csv(csv_path, index=False)

    return 0

def make_audio_dataset(args) -> int:
    """
    Read in a csv at `args.input` where one column is a list of paths for
    wav files with the column name specified by `args.wav_col`.
    Creates Audio Dataset from .csv file and saves in PyArrow format at `args.output`.
    """
    csv_path = os.path.join(args.input, args.csv_basename)
    df = pd.read_csv(csv_path)
    df = df.rename({args.wav_col: 'audio'}, axis=1)
    ds = Dataset.from_pandas(df).cast_column('audio', Audio(sampling_rate=16_000))
    ds.save_to_disk(args.output)
    return 0

# ---- #
# main #
# ---- #

def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    parser.set_defaults(func=empty_command)
    subparsers=parser.add_subparsers(help='Command to run.')

    arrow_to_wav_parser = subparsers.add_parser('arrow_to_wav')
    arrow_to_wav_parser.set_defaults(func=arrow_to_wav)

    make_audio_dataset_parser = subparsers.add_parser('make_audio_ds')
    make_audio_dataset_parser.add_argument(
        "--csv_basename", '-c', default='eaf_data.csv'
    )
    make_audio_dataset_parser.add_argument(
        "--wav_col", '-w', default='wav_clip'
    )
    make_audio_dataset_parser.set_defaults(func=make_audio_dataset)

    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_argparser()
    args = parser.parse_args(argv)
    return args.func(args)



if __name__ == '__main__':
    main()