import sys
sys.path.append('scripts')
import pandas as pd
from typing import Sequence, Optional
from argparse import ArgumentParser
from glob import glob
import os
from longform import load_and_resample, SAMPLE_RATE
import torchaudio
from tqdm import tqdm

def init_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Mask timestamps in eliciation recordings corresponding to annotations from ASR dataset")
    parser.add_argument('--metadata', '-d')
    parser.add_argument('--elicitation_wavs', '-w')
    parser.add_argument('--output', '-o')
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    wav_files = glob(os.path.join(args.elicitation_wavs, '*.wav'))
    df = pd.read_csv(args.metadata)
    # for each record in the dataset, we need to find the corresponding wav file
    # then mask the timestamps for that annotation in the original wav file
    # by setting the samples to zero
    df['wav_basename']=df['wav_source'].apply(lambda x: os.path.basename(x))
    for wav_path in tqdm(wav_files):
        wav = load_and_resample(wav_path)
        wav_basename = os.path.basename(wav_path)
        has_basename = df['wav_basename']==wav_basename
        # get the start and end sample indices from annotation timestamps
        start_idcs = df.loc[has_basename, 'start'].apply(lambda x: int(x*SAMPLE_RATE))
        end_idcs = df.loc[has_basename, 'end'].apply(lambda x: int(x*SAMPLE_RATE))
        for start, end in zip(start_idcs, end_idcs):
            wav[start:end] = 0 # mask the samples
        # save the masked wav file
        torchaudio.save(os.path.join(args.output, wav_basename), wav, SAMPLE_RATE)
    return 0

if __name__ == '__main__':
    main()