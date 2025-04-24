from argparse import ArgumentParser
from typing import Sequence, Optional
import os

AUDIO_DIR = os.environ.get("TIRA_ELICIATION_WAVS")
TIRA_ASR_DIR = os.environ.get("TIRA_ASR_DS")
VERSION = "0.1.0"

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = ArgumentParser()
    parser.add_argument()

if __name__ == '__main__':
    main()