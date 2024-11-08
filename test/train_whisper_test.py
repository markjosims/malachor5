from argparse import Namespace
from transformers import Seq2SeqTrainer

import sys
sys.path.append('scripts')
from train_whisper import evaluate_dataset, init_parser, get_metrics, get_training_args
from dataset_utils import load_and_prepare_dataset, FLEURS, SPECIAL_TOKENS, load_data_collator
from model_utils import load_whisper_model_for_training_or_eval

def test_lang_col_generate(tmpdir):
    """
    Test that setting `--language` arg correctly
    ensures forced decoding to the given language during decoding
    with generate.
    """
    args = init_parser().parse_args([])
    args.output=str(tmpdir)
    args.dataset = FLEURS
    args.language = ['sw']
    args.num_records = 50
    args.predict_with_generate=True
    args.model = 'openai/whisper-tiny'
    args.action = 'evaluate'

    ds, processor = load_and_prepare_dataset(args)
    compute_metrics = get_metrics(args, processor)
    training_args = get_training_args(args)
    model = load_whisper_model_for_training_or_eval(args)
    data_collator = load_data_collator(model, processor)
    trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )
    predictions = evaluate_dataset(args, ds['validation'], trainer, processor, save_results_to_disk=False)
    predictions = predictions.predictions
    swahili_token = SPECIAL_TOKENS['sw']['id']
    for pred in predictions:
        assert swahili_token in pred


