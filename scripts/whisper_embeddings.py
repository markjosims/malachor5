from typing import Optional, Sequence
from argparse import ArgumentParser
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers import WhisperProcessor
from datasets import load_dataset
import torch
from fleurs import _FLEURS_LONG_TO_LANG


def whisper_embeddings(args) -> torch.Tensor:
    model = WhisperEncoder.from_pretrained(args.model)
    proc = WhisperProcessor.from_pretrained(args.model, language=args.language)
    ds = load_dataset(
        args.dataset,
        _FLEURS_LONG_TO_LANG[args.language],
        split=args.split
    )
    ds = ds.map(
        lambda batch: embed_record(batch, proc, model),
        remove_columns=ds.column_names,
        batched=True,
        batch_size=8,
    )
    embeds = torch.tensor(ds['embeds'])
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
    parser.add_argument('--language', '-l')
    parser.add_argument('--split', '-s', default='test')
    # parser.add_argument('--sample_num', '-n', default=500)
    parser.add_argument('--output', '-o')
    parser.add_argument('--average', '-a', action='store_true')
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    embeds = whisper_embeddings(args)
    embeds_name = args.output or \
        f"{args.dataset.split('/')[-1]}-{_FLEURS_LONG_TO_LANG[args.language]}-{args.split}.pt"
    torch.save(embeds, embeds_name)
    return 0

if __name__ == '__main__':
    main()