from typing import Optional, Sequence
from argparse import ArgumentParser
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers import WhisperProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import json
import os

DEVICE = 0 if torch.cuda.is_available() else "cpu"

with open('meta/language_codes.json') as f:
    LANGUAGE_CODES = json.load(f)

def collate_hf_dataset(batch, proc, device):
    return proc(
        [row['audio']['array'] for row in batch],
        return_tensors='pt',
        sampling_rate=16_000,
    ).to(device)

def whisper_embeddings(args, language: str, model: Optional[WhisperEncoder]=None) -> torch.Tensor:
    print("Calculating embeddings for language", language, "from dataset", args.dataset)
    if not model:
        model = WhisperEncoder.from_pretrained(args.model)
        model = model.to(args.device)
    proc = WhisperProcessor.from_pretrained(args.model, language=language)
    ds = load_dataset(
        args.dataset,
        LANGUAGE_CODES[language],
        split=args.split,
        streaming=True,
    )
    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_hf_dataset(batch, proc, args.device),
    )
    embeds = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_embeds = model(batch['input_features'])['last_hidden_state']
            embeds.append(batch_embeds.to('cpu'))
    
    embeds = torch.concat(embeds, dim=0)
    if args.average:
        embeds = torch.mean(embeds, dim=0)
    return embeds
        

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', default="openai/whisper-large-v3")
    # TODO: implement choosing Encoder vs Decoder
    parser.add_argument('--dataset', '-d', default='google/fleurs')
    parser.add_argument('--language', '-l', nargs='+')
    parser.add_argument('--split', '-s', default='test')
    # parser.add_argument('--sample_num', '-n', default=500)
    parser.add_argument('--output', '-o')
    parser.add_argument('--average', '-a', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--device', '-D', default=DEVICE)
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    if args.language == ['all']:
        args.language = LANGUAGE_CODES.keys()
    model = WhisperEncoder.from_pretrained(args.model)
    model = model.to(args.device)

    for language in args.language:
        embeds = whisper_embeddings(args, model=model, language=language)
        embeds_path = f"{args.dataset.split('/')[-1]}-{LANGUAGE_CODES[language]}-{args.split}.pt"
        if args.output:
            embeds_path = os.path.join(args.output, embeds_path)
        torch.save(embeds, embeds_path)
    return 0

if __name__ == '__main__':
    main()