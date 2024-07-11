from transformers import pipeline
from datasets import load_from_disk, Audio, Dataset
from typing import Optional, Sequence, Dict, Any, Union, List, Generator
from argparse import ArgumentParser
import torch
import os
from tqdm import tqdm

MMS_LID_256 = 'facebook/mms-lid-256'
DEFAULT_SR = 16_000
DEVICE = 0 if torch.cuda.is_available() else -1

def dataset_generator(dataset: Dataset) -> Generator:
    for row in dataset:
        yield row['audio']

def init_argparser() -> ArgumentParser:
    parser = ArgumentParser("Script for running SLI experiment")
    parser.add_argument(
        "--model", '-m',
    )
    parser.add_argument(
        "--dataset", '-d',
    )
    parser.add_argument(
        "--device", '-D', type=int, default=DEVICE,
    )
    parser.add_argument(
        '--batch_size', '-b', type=int,
    )
    parser.add_argument(
        "--output", '-o',
    )
    parser.add_argument(
        "--split", '-s', choices=['train', 'test', 'validation'], default='test',
    )
    return parser

def compare_predictions(row: Dict[str, Any]):
    label_col = 'lang'
    pred_col = 'output'

    label = row[label_col].lower()
    pred = row[pred_col]
    pred.sort(key=lambda x:x['score'], reverse=True)

    pred_label = pred[0]['label']
    pred_score = pred[0]['score']

    eng_score = [score for score in pred if score['label'].lower()=='eng'][0]['score']

    # TODO: allow using decision threshold for P(ENG) to determine accuracy
    # TODO: allow metalang to be set dynamically
    # correct if model predicts same label
    # in practice this will be when model predicts 'eng' for English
    if label==pred_label:
        acc=1
    # correct if model predicts non-English for non-English
    elif (label!='eng') and (pred_label!='eng'):
        acc=1
    # incorrect otherwise
    else:
        acc=0

    return {
        "label": label,
        "pred": pred_label,
        "pred_score": pred_score,
        "eng_score": eng_score,
        "acc": acc,
    }

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_argparser()
    args = parser.parse_args(argv)

    # load model and dataset
    pipe = pipeline(
        'audio-classification', 
        args.model,
        device=(torch.device(args.device)),
    )
    split_path = os.path.join(args.dataset, args.split)
    dataset = load_from_disk(split_path)
    dataset = dataset.cast_column('audio', Audio(sampling_rate=DEFAULT_SR))

    # run inference on dataset using pipeline
    ds_gen = dataset_generator(dataset)
    output = []
    for batch_output in tqdm(
        pipe(ds_gen, batch_size=args.batch_size),
        total=len(dataset),
        desc='SLI pipeline'
    ):
        output.append(batch_output)

    # calculate accuracy on output
    dataset = dataset.add_column("output", output)
    output_metrics = dataset.map(compare_predictions, remove_columns=dataset.column_names)
    output_metrics = output_metrics.to_pandas()

    # save result
    output_metrics.to_csv(args.output)

    return 0

if __name__ == '__main__':
    main()