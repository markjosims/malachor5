import torch.utils
from transformers import pipeline
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.dataio.batch import PaddedBatch
from datasets import load_from_disk, Audio, Dataset
from datasets.combine import concatenate_datasets
from typing import Optional, Sequence, Dict, Any, List, List, Generator
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import pandas as pd
import json

MMS_LID_256 = 'facebook/mms-lid-256'
DEFAULT_SR = 16_000
DEVICE = 0 if torch.cuda.is_available() else -1

# ----------------------- #
# Data processing methods #
# ----------------------- #

def dataset_generator(dataset: Dataset) -> Generator:
    """
    For progress bars to work with the HuggingFace pipeline,
    the dataset must be wrapped in an iterable class,
    with the Pipeline object handling batching.
    """
    for row in dataset:
        yield row['audio']


# ----------------- #
# Inference methods #
# ----------------- #

def infer_hf(args, dataset) -> List[Dict[str, Any]]:
    pipe = pipeline(
        'audio-classification', 
        args.model,
        device=(torch.device(args.device)),
    )

    ds_gen = dataset_generator(dataset)
    output = []
    for batch_output in tqdm(
        pipe(ds_gen, batch_size=args.batch_size),
        total=len(dataset),
        desc='SLI pipeline'
    ):
        output.append(batch_output)
    return output

def infer_sb(args, dataset) -> List[Dict[str, Any]]:
    model = EncoderClassifier.from_hparams(
        source=args.model,
        savedir=args.sb_savedir,
        run_opts={"device":torch.device(args.device)},
    )

    # create a dataloader that returns batches of wav objs
    # dataset = dataset.map(lambda row: {'wav': row['audio']['array']})
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: PaddedBatch([row['wav'] for row in b])
    )

    label_encoder = model.hparams.label_encoder
    assert label_encoder.is_continuous()

    long_labels = label_encoder.decode_ndim(range(len(label_encoder)))
    iso_codes = [label.split(':')[0] for label in long_labels]

    outputs = []
    for batch in tqdm(dataloader):
        prediction = model.classify_batch(batch)
        probs = prediction[0]

        # save label probabilities as a list of dicts
        # to match output returned by HF pipeline
        for row_probs in probs:
            row_obj = []
            for i, log_prob in row_probs:
                label = iso_codes[i]
                long_label = long_labels[i]
                prob = log_prob.exp().item()
                row_obj.append()
            outputs.append({'label': label, 'score': prob, 'long_label': long_label})
    return outputs

# ------------------ #
# Evaluation methods #
# ------------------ #

def compare_predictions(row: Dict[str, Any]):
    label_col = 'lang'
    pred_col = 'output'

    label = row[label_col].lower()
    pred = row[pred_col]
    pred.sort(key=lambda x:x['score'], reverse=True)

    pred_label = pred[0]['label']
    pred_score = pred[0]['score']

    eng = [score for score in pred if score['label'].lower()=='eng']
    eng_score = eng[0]['score'] if len(eng)>0 else 0

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
        "pred_score": round(pred_score, 4),
        "eng_score": round(eng_score, 4),
        "acc": acc,
    }

def get_metric_summary(metrics: pd.DataFrame) -> Dict[str, float]:
    summary_obj = {}

    label_tic = metrics['label']=='tic'
    label_eng = metrics['label']=='eng'

    summary_obj['tic_mean_acc'] = round(metrics[label_tic]['acc'].mean(), 4)
    summary_obj['tic_mean_eng_score'] = round(metrics[label_tic]['eng_score'].mean(), 4)
    summary_obj['tic_lang_counts'] = metrics[label_tic]['pred'].value_counts().to_dict()
    
    summary_obj['eng_mean_acc'] = round(metrics[label_eng]['acc'].mean(), 4)
    summary_obj['eng_mean_eng_score'] = round(metrics[label_eng]['eng_score'].mean(), 4)
    summary_obj['eng_lang_counts'] = metrics[label_eng]['pred'].value_counts().to_dict()

    tic_langs = set(metrics[label_tic]['pred'])
    eng_langs = set(metrics[label_eng]['pred'])
    overlap_langs = set.intersection(tic_langs, eng_langs)
    summary_obj['overlap'] = list(overlap_langs)

    return summary_obj

# ---- #
# Main #
# ---- #

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
        "--split", '-s', choices=['train', 'test', 'validation', 'all'], default='test',
    )
    parser.add_argument(
        '--inference_api', '-a', choices=['hf', 'sb'], default='hf',
    )
    parser.add_argument(
        '--sb_savedir', help='Path to save SpeechBrain model to, if not saved locally already.'
    )
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_argparser()
    args = parser.parse_args(argv)

    # load dataset
    if args.split == 'all':
        dataset = load_from_disk(args.dataset)
        dataset = concatenate_datasets(dataset.values())
    else:
        split_path = os.path.join(args.dataset, args.split)
        dataset = load_from_disk(split_path)
    dataset = dataset.cast_column('audio', Audio(sampling_rate=DEFAULT_SR))

    # run inference on dataset using pipeline
    if args.inference_api == 'hf':
        output = infer_hf(args, dataset)
    else:
        output = infer_sb(args, dataset)

    # calculate accuracy on output
    dataset = dataset.add_column("output", output)
    output_metrics = dataset.map(compare_predictions, remove_columns=dataset.column_names)
    output_metrics = output_metrics.to_pandas()
    summary = get_metric_summary(output_metrics)

    # save result
    output_metrics.to_csv(args.output+'.csv')
    with open(args.output+'.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return 0

if __name__ == '__main__':
    main()