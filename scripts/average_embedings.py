from glob import glob
import torch
from argparse import ArgumentParser
from typing import Optional, Sequence
from tqdm import tqdm

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', help='globstr for matching pytorch files.')
    parser.add_argument('--output', '-o')
    parser.add_argument('--device', '-d', type=int)
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    embeds_dict = {}
    for pt_file in tqdm(glob(args.input)):
        embed_matrix = torch.load(pt_file, map_location=torch.device(args.device))
        embed_avg = torch.mean(embed_matrix, dim=0).to('cpu')
        del embed_matrix
        embeds_dict[pt_file]=embed_avg
    torch.save(embeds_dict, args.output)

if __name__ == '__main__':
    main()