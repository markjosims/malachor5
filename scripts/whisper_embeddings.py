from typing import Optional, Sequence, Generator
from argparse import ArgumentParser
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers import WhisperProcessor
from speechbrain.inference.classifiers import EncoderClassifier
from datasets import load_dataset, load_from_disk, Audio, IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_utils import collate_dataset
from model_utils import sb_model
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

    # wrap in generator if iterable
    if type(ds) is IterableDataset:
        ds = DatasetGenerator(ds, args.num_records)
    # otherwise, if num_records is specified, slice
    elif args.num_records:
        ds=ds[:args.num_records]

    if args.model_type == 'whisper':
        proc = WhisperProcessor.from_pretrained(args.model, language=language)
        collate_fn = lambda batch: collate_whisper(batch, proc, args.device)
    else:
        collate_fn = collate_dataset

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

# --------------------- #
# Model loading methods #
# --------------------- #

def load_model(args):
    if ('openai' in args.model) or ('whisper' in args.model):
        args.model_type='whisper'
        model = WhisperEncoder.from_pretrained(args.model)
        model = model.to(args.device)
        return model
    if ('speechbrain' in args.model):
        args.model_type='sb'
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
    embeds_tensor = torch.mean(hidden_states, dim=1)
    del hidden_states
    # optionally average across all records
    if args.average:
        embeds_tensor = torch.mean(embeds_tensor, dim=0)
    return embeds_tensor

def sb_embeddings(args, language: Optional[str]=None, model: Optional[EncoderClassifier]=None) -> torch.Tensor:
    if not model:
        model=sb_model(args)
    dataloader = get_dataloader(args, language)
    embeds = []
    for batch in tqdm(dataloader):
        batch_embeds = model.encode_batch(batch).cpu()
        embeds.append(batch_embeds)
    embeds_tensor = torch.concat(embeds)
    # optionally average across all records
    if args.average:
        embeds_tensor = torch.mean(embeds_tensor, dim=0)
    return embeds_tensor

# ---- #
# Main #
# ---- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', default="openai/whisper-large-v3")
    parser.add_argument('--sb_savedir')
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
    if args.model_type=='whisper':
        embed_funct=whisper_embeddings
    else: # args.model_type=='sb'
        embed_funct=sb_embeddings

    # for multilingual dataset load each language individually
    if args.language:
        for language in tqdm(args.language):
            embeds_path = f"{args.dataset.split('/')[-1]}-{language}-{args.split}.pt"
            if os.path.exists(embeds_path):
                tqdm.write(f"PyTorch file already found at {embeds_path} for language {language}, skipping")
                continue
            
            try:
                tqdm.write(f"Calculating embeddings for language {language}, from dataset, {args.dataset}")
                embeds = embed_funct(args, model=model, language=language)
                if args.output:
                    embeds_path = os.path.join(args.output, embeds_path)
                torch.save(embeds, embeds_path)
            except Exception as e:
                tqdm.write(f"Error when calculating embeddings for language {language}, skipping")
                tqdm.write(str(e))
    # otherwise assume monolingual dataset, e.g. Tira ASR corpus
    else:
        embeds = embed_funct(args, model=model)
        embeds_path = f"{args.dataset.split('/')[-1]}-{args.split}.pt"
        if args.output:
            embeds_path = os.path.join(args.output, embeds_path)
        torch.save(embeds, embeds_path)
    return 0

if __name__ == '__main__':
    main()