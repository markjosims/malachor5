from datasets import load_from_disk, Dataset, DatasetDict
from typing import Optional, Sequence, Dict, Any
from argparse import ArgumentParser
import torchaudio
import torch
import os

# --------------- #
# dataset methods #
# --------------- #
def row_to_wav(row: Dict[str, Any], outdir: str) -> None:
    basepath = row['audio']['path']
    path = os.path.join(outdir, basepath)
    wav_array = row['audio']['array']
    sr = row['audio']['sampling_rate']
    # cast to torch tensor and make 2D
    wav_tensor = torch.Tensor(wav_array).unsqueeze(0)
    torchaudio.save(path, wav_tensor, sr)

def audio_as_path(row: Dict[str, Any], outdir: str) -> Dict[str, str]:
    basepath = row['audio']['path']
    path = os.path.join(outdir, basepath)
    return {'path': path}

# --------------- #
# command methods #
# --------------- #

def arrow_to_wav(args) -> int:
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

# ---- #
# main #
# ---- #

def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('COMMAND', choices=['ARROW_TO_WAV'])
    parser.add_argument('--input', '-i')
    parser.add_argument('--outdir', '-o')
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_argparser()
    args = parser.parse_args(argv)
    if args.COMMAND == 'ARROW_TO_WAV':
        return arrow_to_wav(args)
    return 1



if __name__ == '__main__':
    main()