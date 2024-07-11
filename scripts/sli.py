from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from datasets import load_from_disk
from typing import Optional, Sequence
from argparse import ArgumentParser

MMS_LID_256 = 'facebook/mms-lid-256'

def init_argparser() -> ArgumentParser:
    parser = ArgumentParser("Script for running SLI experiment")
    parser.add_argument(
        "--model", '-m'
    )
    parser.add_argument(
        "--dataset", '-d'
    )
    parser.add_argument(
        "--device", '-D', type=int,
    )

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_argparser()
    args = parser.parse_args(argv)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model)
    proc = Wav2Vec2FeatureExtractor.from_pretrained(args.model)

    dataset = load_from_disk(args.dataset)

    

    return 0

if __name__ == '__main__':
    main()