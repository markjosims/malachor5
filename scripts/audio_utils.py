from datasets import load_from_disk
from typing import Optional, Sequence
from argparse import ArgumentParser

def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('COMMAND', choices=['ARROW_TO_WAV'])

def main(argv: Optional[Sequence[str]]=None) -> int:
    return 0



if __name__ == '__main__':
    main()