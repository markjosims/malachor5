import sys
sys.path.append('scripts')
from dataset_utils import load_dataset_safe
from typing import Sequence, Optional
from argparse import ArgumentParser

def init_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Mask timestamps in eliciation recordings corresponding to annotations from ASR dataset")
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--elicitation_wavs', '-w')
    parser.add_argument('--output,' '-o')
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    
    return 0

if __name__ == '__main__':
    main()