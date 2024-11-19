import torch
import sys
import os
sys.path.append('scripts')
from train_whisper import evaluate_dataset, init_parser, get_metrics, get_training_args, preprocess_logits_for_metrics, calculate_fisher_matrix
from dataset_utils import load_and_prepare_dataset, load_data_collator, FLEURS, SPECIAL_TOKENS, TIRA_BILING, TIRA_ASR_DS
from model_utils import WhisperTrainer, load_whisper_model_for_training_or_eval, prepare_trainer_for_peft

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
    model = load_whisper_model_for_training_or_eval(args, processor)
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


def test_lang_token_peft(tmpdir):
    """
    Test that setting `--peft_type language_token` freezes gradients
    for all parameters except embedding weights for given language ID.
    """
    args = init_parser().parse_args([])
    args.output = str(tmpdir)
    args.dataset = TIRA_ASR_DS
    args.language = ['sw']
    args.num_records = 10
    args.model = 'openai/whisper-tiny'
    args.num_train_epochs = 1
    args.peft_type = 'lang_token'

    ds, processor = load_and_prepare_dataset(args)
    compute_metrics = get_metrics(args, processor)
    training_args = get_training_args(args)
    model = load_whisper_model_for_training_or_eval(args)
    data_collator = load_data_collator(model, processor)
    token_embedding_matrix = model.model.decoder.embed_tokens.weight.detach().clone()
    trainer = WhisperTrainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
            train_dataset=ds['train'],
            eval_dataset=ds['validation'],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
    trainer = prepare_trainer_for_peft(args, trainer, processor)
    trainer.train()
    swahili_token = SPECIAL_TOKENS['sw']['id']
    for name, param in model.named_parameters():
        if name=='model.decoder.embed_tokens.weight':
            assert param.requires_grad
        else:
            assert not param.requires_grad

    token_embedding_matrix_trained = model.model.decoder.embed_tokens.weight
    for i, embedding_vector in enumerate(token_embedding_matrix):
        embedding_vector_trained=token_embedding_matrix_trained[i]
        if i==swahili_token:
            assert not torch.equal(embedding_vector, embedding_vector_trained)
        else:
            assert torch.equal(embedding_vector, embedding_vector_trained)

def test_save_fisher_matrix(tmpdir):
    parser = init_parser()
    args = parser.parse_args([])
    args.output = str(tmpdir)
    args.dataset = TIRA_ASR_DS
    args.language = ['sw']
    args.num_records = 10
    args.model = 'openai/whisper-tiny'
    args.num_train_epochs = 1
    args.action = 'calculate_fisher'

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
            train_dataset=ds['train'],
            eval_dataset=ds['validation'],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    calculate_fisher_matrix(args, trainer)
    fisher_matrix_path = os.path.join(
        args.output,
        os.path.basename(TIRA_ASR_DS)+'_fisher.pt' 
    )
    assert os.path.exists(fisher_matrix_path)
    assert type(torch.load(args.output)) is torch.Tensor