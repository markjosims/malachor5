from glob import glob
import torch
from argparse import ArgumentParser
from typing import Optional, Sequence
from tqdm import tqdm

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_arg('--input', '-i', help='globstr for matching pytorch files.')
    parser.add_arg('--output', '-o')
    return parser

def main(argv: Optional[Sequence[str]]) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    embeds_dict = {}
    for pt_file in tqdm(glob(args.input)):
        embed_matrix = torch.load(pt_file)
        embed_avg = torch.mean(embed_matrix, dim=0)
        embeds_dict[pt_file]=embed_avg
    torch.save(embed_matrix, args.output)

if __name__ == '__main__':
    main()