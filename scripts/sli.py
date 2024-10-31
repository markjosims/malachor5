import torch.utils
from transformers import pipeline, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from sklearn.linear_model import LogisticRegression
from datasets import load_from_disk, Audio, DatasetDict
from typing import Optional, Sequence, Dict, Any, List, List, Union
from argparse import ArgumentParser
import torch
from tqdm import tqdm
import pandas as pd
import pickle
from dataset_utils import build_dataloader, dataset_generator
from model_utils import load_lr, sb_model, DEVICE
from speechbrain.inference.classifiers import EncoderClassifier

MMS_LID_256 = 'facebook/mms-lid-256'
DEFAULT_SR = 16_000

# --------- #
# argparser #
# --------- #

def empty_command(args) -> int:
    print("Specify a command to run.")
    return 1

def add_sli_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--sli_model')
    parser.add_argument('--sli_model_type', choices=['hf', 'sb'], default='sb')
    parser.add_argument('--sb_savedir', default='speechbrain')
    return parser

def init_argparser() -> ArgumentParser:
    parser = ArgumentParser("Script for running SLI experiment")
    parser.add_argument('--dataset', '-D')
    parser.add_argument('--output', '-o')
    parser.add_argument('--device', '-d')
    parser=add_sli_args(parser)
    parser.set_defaults(func=empty_command)
    commands=parser.add_subparsers('command')

    train_lr_parser=commands.add_parser('train_lr', help=train_logreg.__doc__)
    train_lr_parser.add_argument('--embeds_path')
    train_lr_parser.set_defaults(func=train_logreg)
    
    return parser

# ----------------- #
# Inference methods #
# ----------------- #

def infer_lr(args, dataset):
    embeds = load_embeddings(args, dataset)
    lr = load_lr(args)
    outputs = lr.predict(embeds)

    return outputs

# ----------------- #
# Embedding methods #
# ----------------- #

def sb_embeddings(
        args,
        dataset,
        model: Optional[EncoderClassifier]=None
    ) -> torch.Tensor:
    sli_model=getattr(args, 'sli_model', 'speechbrain/lang-id-voxlingua107-ecapa')
    if model is None:
        print(f"Loading SpeechBrain model {sli_model} for extracting embeddings.")
        model = sb_model(args)
    if type(dataset) is DatasetDict:
        embedding_dict={}
        for split in dataset:
            embedding_dict[split]=sb_embeddings(args, dataset[split], model)
        return embedding_dict

    dataloader = build_dataloader(dataset, args.sli_batch_size)

    embeddings = []
    for batch in tqdm(dataloader):
        batch_embeds = model.encode_batch(batch).cpu()
        embeddings.append(batch_embeds)

    embedding_tensor = torch.concat(embeddings).squeeze()
    return embedding_tensor

def hf_embeddings(args, dataset, model=None) -> torch.Tensor:
    if model is None:
        print(f"Loading HuggingFace model {args.sli_model} for extracting embeddings.")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(args.sli_model)
        proc = Wav2Vec2FeatureExtractor.from_pretrained(args.sli_model)
        model.to(torch.device(args.device))

    if type(dataset) is DatasetDict:
        embedding_dict={}
        for split in dataset:
            embedding_dict[split]=hf_embeddings(args, dataset[split], model)
        return embedding_dict

    dataloader = build_dataloader(dataset, args.sli_batch_size)

    logits = []
    # [
    # tensor([logit, logit, logit,...]),  logits for example 1
    # tensor([logit, logit, logit,...]),  logits for example 2
    # ...
    #]
    hidden_states = []
    # [
    #   tensor([                                    activations for example 1
    #     [activation, activation, ...] * frames,   activations for layer 1 example 1
    #     [activation, activation, ...] * frames,   activations for layer 2 example 1
    #   ... ]),
    #   tensor([                                    activations for example 2
    #     [activation, activation, ...] * frames,   activations for layer 1 example 2
    #     [activation, activation, ...] * frames,   activations for layer 2 example 2
    #   ... ]),
    # ]

    for batch in tqdm(dataloader):
        processed_batch = proc(batch.numpy(), return_tensors='pt', sampling_rate=DEFAULT_SR)
        processed_batch = processed_batch.to(torch.device(args.device))
        with torch.no_grad():
            output = model(**processed_batch, output_hidden_states=True)
        output_hs = output['hidden_states']
        output_lgts = output['logits']

        # logits are simple, just append
        logits.append(output_lgts)
        
        # hidden states are the problem child
        # first make a list where each item is a tensor of hidden states for a single example
        batch_hidden_states = [None for _ in range(len(batch))]
        for layer in output_hs:
            for i, record_activations in enumerate(layer):
                # `record_activations` is a 2D tensor of hidden units by frame
                # remove any padded zeros
                not_padded = record_activations.abs().sum(dim=1)!=0
                record_activations=record_activations[not_padded]
                record_activations=record_activations.unsqueeze(0)
                # add this layer's activations to the tensor corresponding
                # to the current record
                if batch_hidden_states[i] is None:
                    batch_hidden_states[i] = record_activations
                else:
                    batch_hidden_states[i] = torch.concat([
                        batch_hidden_states[i],
                        record_activations
                    ])
        batch_hidden_states = [hs.cpu() for hs in batch_hidden_states]
        hidden_states.append(batch_hidden_states)

    # logits are simple
    logits = torch.concat(logits).cpu()

    return logits, hidden_states

def load_embeddings(args, dataset):
    if getattr(args, 'embeds_path', None):
        embeds = torch.load(args.embeds_path)
    elif getattr(args, 'embed_api', 'sb') == 'hf':
        embeds = hf_embeddings(args, dataset)
    else:
        embeds = sb_embeddings(args, dataset)
    return embeds

# ------------------- #
# logistic regression #
# ------------------- #

def train_logreg(args, dataset) -> int:
    embeds = load_embeddings(args, dataset)

    train_labels=dataset['train']['label']
    test_labels=dataset['test']['label']

    train_X=embeds['train']
    test_X=embeds['test']

    logreg=LogisticRegression().fit(train_X, train_labels)
    scores=logreg.score(test_X, test_labels)
    print(scores)
    
    with open(args.output, 'wb') as f:
        pickle.dump(logreg, f)
    
    return 0 

# ---- #
# Main #
# ---- #

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_argparser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()