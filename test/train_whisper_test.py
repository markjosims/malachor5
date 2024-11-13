from argparse import Namespace

import sys
sys.path.append('scripts')
from train_whisper import evaluate_dataset, init_parser, get_metrics, get_training_args, WhisperTrainer
from dataset_utils import load_and_prepare_dataset, load_data_collator, FLEURS, SPECIAL_TOKENS, TIRA_BILING, TIRA_ASR_DS
from model_utils import load_whisper_model_for_training_or_eval

def test_lang_col_generate(tmpdir):
    """
    Test that setting `--language` arg correctly
    ensures forced decoding to the given language during decoding
    with generate.
    """
    args = init_parser().parse_args([])
    args.output=str(tmpdir)
    args.dataset = TIRA_ASR_DS
    args.language = ['sw']
    args.num_records = 10
    args.predict_with_generate=True
    args.model = 'openai/whisper-tiny'
    args.action = 'evaluate'
    args.eval_datasets=[FLEURS, TIRA_BILING]
    args.eval_dataset_languages=['en', 'sw+en']

    ds, processor = load_and_prepare_dataset(args)
    compute_metrics = get_metrics(args, processor)
    training_args = get_training_args(args)
    model = load_whisper_model_for_training_or_eval(args)
    data_collator = load_data_collator(model, processor)
    trainer = WhisperTrainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )
    predictions_dict = evaluate_dataset(args, ds['validation'], trainer, processor, save_results_to_disk=False)
    swahili_token = SPECIAL_TOKENS['sw']['id']
    en_token = SPECIAL_TOKENS['en']['id']

    for pred in predictions_dict[FLEURS.split('/')[-1]].predictions:
        assert en_token in pred

    for pred in predictions_dict[TIRA_ASR_DS.split('/')[-1]].predictions:
        assert swahili_token in pred
    
    for pred in predictions_dict[TIRA_BILING.split('/')[-1]].predictions:
        assert en_token in pred
        assert swahili_token in pred



