from typing import Optional, Sequence
from argparse import ArgumentParser
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers import WhisperProcessor
from datasets import load_dataset
import torch
import json
import os

DEVICE = 0 if torch.cuda.is_available() else "cpu"

with open('meta/language_codes.json') as f:
    LANGUAGE_CODES = json.load(f)


def whisper_embeddings(args, language: str, model: Optional[WhisperEncoder]=None) -> torch.Tensor:
    print("Calculating embeddings for language", language, "from dataset", args.dataset)
    if not model:
        model = WhisperEncoder.from_pretrained(args.model)
    proc = WhisperProcessor.from_pretrained(args.model, language=language)
    ds = load_dataset(
        args.dataset,
        LANGUAGE_CODES[language],
        split=args.split
    )
    ds = ds.map(
        lambda batch: embed_record(batch, proc, model),
        remove_columns=ds.column_names,
        batched=True,
        batch_size=args.batch_size,
    )
    embeds = torch.tensor(ds['embed'])
    if args.average:
        embeds = torch.mean(embeds, dim=0)
    return embeds
        

def embed_record(batch, proc, model):
    input_dict = proc(
        [row["array"] for row in batch['audio']],
        sampling_rate=batch['audio'][0]['sampling_rate'],
        return_tensors='pt',    
    )
    hidden_states = model(input_dict['input_features'])['last_hidden_state']
    embed = torch.mean(hidden_states, 1)
    return {'embed': embed}

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
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    if args.language == ['all']:
        args.language = LANGUAGE_CODES.keys()
    model = WhisperEncoder.from_pretrained(args.model)

    for language in args.language:
        embeds = whisper_embeddings(args, model=model, language=language)
        embeds_path = f"{args.dataset.split('/')[-1]}-{LANGUAGE_CODES[language]}-{args.split}.pt"
        if args.output:
            embeds_path = os.path.join(args.output, embeds_path)
        torch.save(embeds, embeds_path)
    return 0

if __name__ == '__main__':
    main()