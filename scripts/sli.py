from transformers import pipeline
from datasets import load_from_disk, Audio
from typing import Optional, Sequence, Dict, Any, Union, List
from argparse import ArgumentParser
import torch
import os

MMS_LID_256 = 'facebook/mms-lid-256'
DEFAULT_SR = 16_000
DEVICE = 0 if torch.cuda.is_available() else -1

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
    # TODO: allow using decision threshold for P(ENG)
    pred = row[pred_col]
    pred_max = get_max_score(pred)
    pred_label = pred_max['label']

    # correct if model predicts same label
    # in practice this will be when model predicts 'eng' for English
    if label==pred_label:
        score=1
    # correct if model predicts non-English for non-English
    # TODO: allow metalang to be set dynamically
    elif (label!='eng') and (pred_label!='eng'):
        score=1
    # incorrect otherwise
    else:
        score=0

    return {
        "label": label,
        "pred": pred_label,
        "score": score,
    }

def get_max_score(score_list: List[Dict[str, Union[str, float]]]):
    scores_sorted = sorted(score_list, key=lambda x:x['score'], reverse=True)
    return scores_sorted[0]

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
    # output=pipe(dataset['audio'], batch_size=args.batch_size)
    output = dataset.map(lambda r: pipe(r['audio']), batch_size=args.batch_size)

    # calculate accuracy on output
    dataset = dataset.add_column("output", output)
    output_metrics = dataset.map(compare_predictions)
    output_metrics = output_metrics.to_pandas()

    # save result
    output_metrics.to_csv(args.output)

    return 0

if __name__ == '__main__':
    main()