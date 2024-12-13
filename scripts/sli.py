import torch.utils
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from sklearn.linear_model import LogisticRegression
from datasets import DatasetDict, Dataset
from typing import Optional, Sequence, Union, Dict, Literal
from argparse import ArgumentParser
import torch
from tqdm import tqdm
import pickle
from dataset_utils import build_sb_dataloader, load_sli_dataset, add_dataset_args
from model_utils import load_lr, sb_model, DEVICE
from speechbrain.inference.classifiers import EncoderClassifier
from model_utils import add_sli_args

MMS_LID_256 = 'facebook/mms-lid-256'
DEFAULT_SR = 16_000

# --------- #
# argparser #
# --------- #

def empty_command(args) -> int:
    print("Specify a command to run.")
    return 1

def init_argparser() -> ArgumentParser:
    parser = ArgumentParser("Script for running SLI experiment")
    parser.add_argument('--output', '-o')
    parser.add_argument('--device', '-D', default=DEVICE)
    parser.add_argument('--batch_size', type=int, default=8)
    parser=add_sli_args(parser)
    parser=add_dataset_args(parser)
    parser.set_defaults(func=empty_command)
    commands=parser.add_subparsers(title='command')

    train_lr_parser=commands.add_parser('train_lr', help=train_logreg.__doc__)
    train_lr_parser.add_argument('--embeds_path')
    train_lr_parser.set_defaults(func=train_logreg)
    
    return parser

# ----------------- #
# Embedding methods #
# ----------------- #

def sb_embeddings(
        args,
        dataset,
        model: Optional[EncoderClassifier]=None
    ) -> torch.Tensor:
    if getattr(args, 'sli_embed_model', None) is None:
        args.sli_embed_model='speechbrain/lang-id-voxlingua107-ecapa'
    if model is None:
        print(f"Loading SpeechBrain model {args.sli_embed_model} for extracting embeddings.")
        model = sb_model(args)
    if type(dataset) is DatasetDict:
        embedding_dict={}
        for split in dataset:
            embedding_dict[split]=sb_embeddings(args, dataset[split], model)
        return embedding_dict

    dataloader = build_sb_dataloader(dataset, args.batch_size, getattr(args, 'dataset_type', 'hf_dataset'))

    embeddings = []
    for batch in tqdm(dataloader):
        batch_embeds = model.encode_batch(batch).cpu()
        embeddings.append(batch_embeds)

    embedding_tensor = torch.concat(embeddings).squeeze()
    return embedding_tensor

def hf_embeddings(args, dataset, model=None) -> torch.Tensor:
    if getattr(args, 'sli_embed_model', None) is None:
        args.sli_embed_model='facebook/mms-lid-256'
    if model is None:
        print(f"Loading HuggingFace model {args.sli_embed_model} for extracting embeddings.")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(args.sli_embed_model)
        proc = Wav2Vec2FeatureExtractor.from_pretrained(args.sli_embed_model)
        model.to(torch.device(args.device))

    if type(dataset) is DatasetDict:
        embedding_dict={}
        for split in dataset:
            embedding_dict[split]=hf_embeddings(args, dataset[split], model)
        return embedding_dict

    dataloader = build_sb_dataloader(dataset, args.batch_size, getattr(args, 'dataset_type', 'hf_dataset'))

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

def load_embeddings(args, dataset=None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if dataset is None:
        dataset, args = load_sli_dataset(args)
    if getattr(args, 'embeds_path', None):
        embeds = torch.load(args.embeds_path)
    elif getattr(args, 'embed_api', None) == 'hf':
        embeds = hf_embeddings(args, dataset)
    else:
        embeds = sb_embeddings(args, dataset)
    return embeds

# ------------------- #
# logistic regression #
# ------------------- #

def train_logreg(args, dataset=None) -> int:
    if dataset is None:
        dataset, args = load_sli_dataset(args)
    embeds = load_embeddings(args, dataset)

    train_labels=dataset['train']['label']
    test_labels=dataset['test']['label']

    train_X=embeds['train']
    test_X=embeds['test']

    lr=LogisticRegression().fit(train_X, train_labels)
    scores=lr.score(test_X, test_labels)
    print(scores)
    
    lr_dict = {
        'lr_model': lr,
        'embed_model': args.sli_embed_model,
        'embed_api': args.embed_api,
        'scores': scores,
        'sli_map': args.sli_map,
        'sli_label2id': args.sli_label2id,
        'sli_id2label': args.sli_id2label,
    }

    with open(args.output, 'wb') as f:
        pickle.dump(lr_dict, f)
    
    return 0 

def infer_lr(args=None, dataset=None, lr_model:Optional[str]=None, **kwargs):
    lr, args = load_lr(args=args, lr_model=lr_model, **kwargs)
    if dataset is None:
        dataset, _ = load_sli_dataset(args)
    embeds = load_embeddings(args, dataset)
    # using list of chunks
    if getattr(args, 'dataset_type', None) == 'chunk_list':
        outputs = lr.predict(embeds)
        labels = [args.sli_id2label[out] for out in outputs]
        for chunk, label in zip(dataset, labels):
            chunk['sli_pred']=label
    # using either HF Dataset or DatasetDict
    elif type(embeds) is dict:
        for split_name, split_embeds in embeds.items():
            outputs = lr.predict(split_embeds)
            labels = [args.sli_id2label[out] for out in outputs]
            dataset[split_name] = dataset[split_name].add_column("sli_preds", labels)
    else:
        outputs = lr.predict(embeds)
        labels = [args.sli_id2label[out] for out in outputs]
        dataset = dataset.add_column("sli_preds", labels) 

    return dataset, args

# ---- #
# Main #
# ---- #

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_argparser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()