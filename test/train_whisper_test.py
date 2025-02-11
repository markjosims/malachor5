import torch
import sys
import os
sys.path.append('scripts')
from train_whisper import evaluate_dataset, init_parser, get_metrics, get_training_args, argmax_logits, calculate_fisher_matrix, get_lid_probs
from dataset_utils import load_and_prepare_dataset, load_data_collator, FLEURS, LANG_TOKENS, TIRA_BILING, TIRA_ASR_DS
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
    swahili_token = LANG_TOKENS['sw']['id']
    en_token = LANG_TOKENS['en']['id']

    for pred in predictions_dict[FLEURS.split('/')[-1]+'-en'].predictions:
        assert en_token in pred
        assert swahili_token not in pred

    for pred in predictions_dict[TIRA_ASR_DS.split('/')[-1]+'-sw'].predictions:
        assert swahili_token in pred
        assert en_token not in pred
    
    for pred in predictions_dict[TIRA_BILING.split('/')[-1]+'-sw+en'].predictions:
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
    param_dict = {name:param.detach().clone() for name,param in model.named_parameters()}
    trainer = WhisperTrainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
            train_dataset=ds['train'],
            eval_dataset=ds['validation'],
            preprocess_logits_for_metrics=argmax_logits,
        )
    trainer = prepare_trainer_for_peft(args, trainer, processor)
    trainer.train()
    swahili_token = LANG_TOKENS['sw']['id']
    for name, param in model.named_parameters():
        if name=='model.decoder.embed_tokens.weight':
            assert param.requires_grad
            for i, embedding_vector_trained in enumerate(param):
                embedding_vector = param_dict[name][i]
                if i==swahili_token:
                    assert not torch.equal(embedding_vector, embedding_vector_trained)
                else:
                    assert torch.equal(embedding_vector, embedding_vector_trained)
            # sanity check, we checked the swahili token embedding
            assert i>=swahili_token

        else:
            assert not param.requires_grad
            assert torch.equal(param, param_dict[name])

def test_lang_token_regularization(tmpdir):
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
    trainer = WhisperTrainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
            train_dataset=ds['train'],
            eval_dataset=ds['validation'],
            preprocess_logits_for_metrics=argmax_logits,
    )
    trainer = prepare_trainer_for_peft(args, trainer, processor)
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    loss1 = trainer.training_step(model, batch)
    
    toy_embed = torch.zeros(384)
    toy_embed_path = os.path.join(tmpdir, 'embed_center.pt')
    torch.save(toy_embed, toy_embed_path)
    trainer = WhisperTrainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
            train_dataset=ds['train'],
            eval_dataset=ds['validation'],
            preprocess_logits_for_metrics=argmax_logits,
            mean_embed_path=toy_embed_path,
            embed_dist_lambda=1,
    )
    # check euclidean distance
    trainer = prepare_trainer_for_peft(args, trainer, processor)
    trainer.embed_dist_type='euclidean'
    trainer.embed_dist_lambda = 1
    loss2 = trainer.training_step(model, batch)
    assert loss1.item() < loss2.item()

    trainer.embed_dist_lambda = 100
    loss3 = trainer.training_step(model, batch)
    assert loss2.item() < loss3.item()

    # check cosine distance
    trainer.embed_dist_type='cosine'
    trainer.embed_dist_lambda = 1
    loss4 = trainer.training_step(model, batch)
    assert not torch.equal(loss4, loss2)

    trainer.embed_dist_lambda = 100
    loss5 = trainer.training_step(model, batch)
    assert loss4.item() < loss5.item()


def test_save_fisher_matrix(tmpdir):
    """
    Run `calculate_fisher_matrix` and check it outputs
    a dict of Torch tensors.
    """
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
            preprocess_logits_for_metrics=argmax_logits,
        )

    calculate_fisher_matrix(args, trainer, model)
    fisher_matrix_path = os.path.join(
        args.output,
        os.path.basename(TIRA_ASR_DS)+'_fisher.pt' 
    )
    assert os.path.exists(fisher_matrix_path)
    fisher_matrix = torch.load(fisher_matrix_path)
    assert type(fisher_matrix) is dict
    for val in fisher_matrix.values():
        assert type(val) is torch.Tensor

def test_train_w_ewc(tmpdir):
    """
    Train model using EWC, check that setting `ewc_lambda` to a higher value
    returns a higher loss.
    """
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
            preprocess_logits_for_metrics=argmax_logits,
    )
    fisher_matrix_path = calculate_fisher_matrix(args, trainer, model)

    ewc_trainer1 = WhisperTrainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
            train_dataset=ds['train'],
            eval_dataset=ds['validation'],
            preprocess_logits_for_metrics=argmax_logits,
            fisher_matrix_path=fisher_matrix_path,
            ewc_lambda=0.1
    )
    ewc_trainer2 = WhisperTrainer(
            args=training_args,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
            train_dataset=ds['train'],
            eval_dataset=ds['validation'],
            preprocess_logits_for_metrics=argmax_logits,
            fisher_matrix_path=fisher_matrix_path,
            ewc_lambda=1_000
    )

    assert ewc_trainer1.ewc_lambda==0.1
    assert ewc_trainer1.fisher_matrix_path==fisher_matrix_path
    assert type(ewc_trainer1.previous_params) is dict
    assert all(type(v) is torch.Tensor for v in ewc_trainer1.previous_params.values())
    assert type(ewc_trainer1.fisher_matrix) is dict
    assert all(type(v) is torch.Tensor for v in ewc_trainer1.fisher_matrix.values())
    assert ewc_trainer2.ewc_lambda==1_000
    assert all(type(v) is torch.Tensor for v in ewc_trainer2.previous_params.values())
    assert type(ewc_trainer2.fisher_matrix) is dict
    assert all(type(v) is torch.Tensor for v in ewc_trainer2.fisher_matrix.values())

    ewc_trainer1.train()
    ewc_trainer2.train()

    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    loss1 = ewc_trainer1.training_step(model, batch)
    loss2 = ewc_trainer2.training_step(model, batch)

    assert loss1.item() < loss2.item()

def test_get_lid_probs(tmpdir):
    """
    Run `get_lid_probs` and check it outputs
    a dict of Torch tensors.
    """
    parser = init_parser()
    args = parser.parse_args([])
    args.output = str(tmpdir)
    args.dataset = TIRA_ASR_DS
    args.language = ['sw']
    args.num_records = 10
    args.model = 'openai/whisper-tiny'
    args.num_train_epochs = 1
    args.action = 'get_lid_probs'

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
            preprocess_logits_for_metrics=argmax_logits,
    )

    get_lid_probs(args, trainer, model)
    lid_logits_path = os.path.join(
        args.output,
        os.path.basename(TIRA_ASR_DS)+'_lid_logits.pt' 
    )
    assert os.path.exists(lid_logits_path)
    lid_logits = torch.load(lid_logits_path)
    assert type(lid_logits) is dict
    assert len(lid_logits)==99
    for val in lid_logits.values():
        assert type(val) is torch.Tensor
        assert len(val)==args.num_records

def test_lid_loss(tmpdir):
    """
    Compute loss with LID joint task and check different than ASR loss alone.
    """
    parser = init_parser()
    args = parser.parse_args([])
    args.output = str(tmpdir)
    args.dataset = TIRA_ASR_DS
    args.language = ['sw']
    args.num_records = 10
    args.model = 'openai/whisper-tiny'
    args.num_train_epochs = 1

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
            preprocess_logits_for_metrics=argmax_logits,
            lid_loss_alpha=0.2,
    )
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    loss_w_lid = trainer.training_step(model, batch)
    trainer.lid_loss_alpha = None
    loss_base = trainer.training_step(model, batch)
    assert not torch.equal(loss_w_lid, loss_base)

def test_get_lid_labels():
    labels = torch.tensor([
        [5001, 5002,  121,   90,   76  ],
        [5010, 5009,  298,   12,   432 ],
        [5002, 2981,  87,    974,  87  ],
        [5002, 5021,  294,   596,  31  ],
    ], dtype=torch.int)
    lang_ids = [5000+i for i in range(11)]
    lid_logits = torch.randn(labels.shape[0], labels.shape[1], labels.max().item()+1)
    lid_labels = WhisperTrainer.get_lid_labels(labels, lang_ids, num_classes=lid_logits.shape[-1])
    expected_labels = torch.zeros_like(lid_logits[:,0,:]).float()
    for i, row in enumerate(labels):
        for val in row:
            if val in lang_ids:
                expected_labels[i,val]=1
        expected_labels[i]/=expected_labels[i].sum()
    assert torch.equal(lid_labels, expected_labels)
