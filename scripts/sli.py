from transformers import pipeline
from datasets import load_from_disk, Audio
from typing import Optional, Sequence
from argparse import ArgumentParser
import json

MMS_LID_256 = 'facebook/mms-lid-256'
DEFAULT_SR = 16_000
DEVICE = 0 if torch.cuda.is_available() else -1

def init_argparser() -> ArgumentParser:
    parser = ArgumentParser("Script for running SLI experiment")
    parser.add_argument(
        "--model", '-m',
    )
    parser.add_argument(
        "--dataset", '-d',
    )
    parser.add_argument(
        "--device", '-D', type=int, default=DEVICE,
    )
    parser.add_argument(
        '--batch_size', '-b', type=int,
    )
    parser.add_argument(
        "--output", '-o',
    )

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_argparser()
    args = parser.parse_args(argv)

    pipe = pipeline(
        'audio-classification', 
        args.model,
        device=(torch.device(args.device)),
    )
    dataset = load_from_disk(args.dataset)

    dataset = dataset.cast_column('audio', Audio(sampling_rate=DEFAULT_SR))

    output=pipe(dataset['audio'], batch_size=args.batch_size)

    with open(args.output, 'w') as f:
        json.dump(output, f)

    return 0

if __name__ == '__main__':
    main()