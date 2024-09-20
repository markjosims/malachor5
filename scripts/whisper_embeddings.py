from typing import Optional, Sequence, Generator
from argparse import ArgumentParser
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers import WhisperProcessor
from speechbrain.inference.classifiers import EncoderClassifier
from datasets import load_dataset, load_from_disk, Audio, IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sli import sb_model, sb_embeddings, collate_sb
import torch
import json
import os

DEVICE = 0 if torch.cuda.is_available() else "cpu"

with open('meta/language_codes.json') as f:
    LANGUAGE_CODES = json.load(f)

# --------------- #
# Dataset methods #
# --------------- #

def collate_whisper(batch, proc, device):
    return proc(
        [row['audio']['array'] for row in batch],
        return_tensors='pt',
        sampling_rate=16_000,
    ).to(device)

class DatasetGenerator(IterableDataset):
    def __init__(self, dataset, num_records=0):
        self.dataset=dataset
        self.num_records=num_records or len(dataset)
    
    def __iter__(self):
        for i, row in enumerate(self.dataset):
            if i==self.num_records:
                break
            yield row

    def __len__(self):
        return self.num_records


def get_dataloader(args, language: Optional[str]=None) -> DataLoader:
    local = os.path.exists(args.dataset)
    proc = WhisperProcessor.from_pretrained(args.model, language=language)

    # load dataset
    if language:
        ds = load_dataset(
            args.dataset,
            language,
            split=args.split,
            streaming=True,
        )
    elif local:
        try:
            ds = load_from_disk(args.dataset)[args.split]
        except FileNotFoundError:
            ds = load_dataset(args.dataset, split=args.split)
    else:
        ds = load_dataset(
            args.dataset,
            split=args.split,
            streaming=True,
        )

    # resample even if not needed, since checking a streamed dataset is complicated
    print("Resampling to 16_000Hz")
    ds = ds.cast_column('audio', Audio(sampling_rate=16_000))

    # wrap in generator
    ds_gen = DatasetGenerator(ds, args.num_records)

    return DataLoader(
        ds_gen,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_whisper(batch, proc, args.device),
    )

# --------------------- #
# Model loading methods #
# --------------------- #

def load_model(args):
    if ('openai' in args.model) or ('whisper' in args.model):
        model = WhisperEncoder.from_pretrained(args.model)
        model = model.to(args.device)
        return model
    if ('speechbrain' in args.model):
        return sb_model(args)
    raise ValueError("Model type not recognized.")

# ----------------- #
# Embedding methods #
# ----------------- #

def whisper_embeddings(args, language: Optional[str]=None, model: Optional[WhisperEncoder]=None) -> torch.Tensor:
    if not model:
        model = WhisperEncoder.from_pretrained(args.model)
        model = model.to(args.device)
    dataloader = get_dataloader(args, language)
    hidden_states = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_embeds = model(batch['input_features'])['last_hidden_state']
            hidden_states.append(batch_embeds.to('cpu'))
    
    hidden_states = torch.concat(hidden_states, dim=0)
    # make embedding by averaging activations across timestamps
    embeds = torch.mean(hidden_states, dim=1)
    del hidden_states
    # optionally average across all records
    if args.average:
        embeds = torch.mean(embeds, dim=0)
    return embeds

def sb_embeddings(args, language: Optional[str]=None, model: Optional[EncoderClassifier]=None) -> torch.Tensor:
    if not model:
        model=sb_model(args)

# ---- #
# Main #
# ---- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', default="openai/whisper-large-v3")
    # TODO: implement choosing Encoder vs Decoder
    parser.add_argument('--dataset', '-d', default='google/fleurs')
    parser.add_argument('--language', '-l', nargs='+')
    parser.add_argument('--split', '-s', default='test')
    parser.add_argument('--num_records', '-n', type=int)
    parser.add_argument('--output', '-o')
    parser.add_argument('--average', '-a', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--device', '-D', default=DEVICE, type=int)
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    args.device = torch.device(args.device)
    if args.language == ['all'] and 'fleurs' in args.dataset:
        args.language = [lang['fleurs'] for lang in LANGUAGE_CODES if 'fleurs' in lang]
    elif args.language == ['all'] and 'common_voice' in args.dataset:
        args.language = [lang['commonvoice_code'] for lang in LANGUAGE_CODES if 'commonvoice_code' in lang]

    model = load_model(args)

    # for multilingual dataset load each language individually
    if args.language:
        for language in tqdm(args.language):
            print("Calculating embeddings for language", language, "from dataset", args.dataset)
            embeds = whisper_embeddings(args, model=model, language=language)
            embeds_path = f"{args.dataset.split('/')[-1]}-{language}-{args.split}.pt"
            if args.output:
                embeds_path = os.path.join(args.output, embeds_path)
            torch.save(embeds, embeds_path)
    # otherwise assume monolingual dataset, e.g. Tira ASR corpus
    else:
        embeds = whisper_embeddings(args, model=model)
        embeds_path = f"{args.dataset.split('/')[-1]}-{args.split}.pt"
        if args.output:
            embeds_path = os.path.join(args.output, embeds_path)
        torch.save(embeds, embeds_path)
    return 0

if __name__ == '__main__':
    main()