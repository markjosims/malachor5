import torch
import numpy as np
import json
import sys
import os
sys.path.append('scripts')
from train_whisper import evaluate_dataset, init_parser, get_metrics, get_training_args, argmax_logits, calculate_fisher_matrix, get_lid_probs, train
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

def test_get_lid_labels_colwise():
    labels = torch.tensor([
        [5001, 5002,  121,   90,   76  ],
        [5010, 5009,  298,   12,   432 ],
        [5002, 2981,  87,    974,  87  ],
        [5002, 5021,  294,   596,  31  ],
    ], dtype=torch.int)
    lang_ids = [5000+i for i in range(11)]
    lid_logits = torch.randn(labels.shape[0], labels.shape[1], labels.max().item()+1)
    lid_labels = WhisperTrainer.get_lid_labels(
        labels,
        lang_ids,
        num_classes=lid_logits.shape[-1],
        colwise=True,
    )
    expected_labels = torch.zeros_like(lid_logits[:,:2,:])
    for i, row in enumerate(labels):
        for j, val in enumerate(row):
            if val in lang_ids:
                expected_labels[i,j,val]=1 
    assert torch.equal(lid_labels, expected_labels)

def test_compute_lid_loss():
    labels = torch.tensor([
        [4, 2, 2],
        [4, 4, 2]
    ], dtype=torch.long)
    perfect_logits = torch.nn.functional.one_hot(labels).float()
    perfect_logits[perfect_logits==0]=-np.inf
    perfect_logits.requires_grad=True
    lang_ids = [3,4]
    perfect_loss = WhisperTrainer.compute_lid_loss(perfect_logits, labels, lang_ids)
    assert perfect_loss == 0

    perfect_loss_colwise = WhisperTrainer.compute_lid_loss(perfect_logits, labels, lang_ids, colwise=True)
    assert perfect_loss_colwise == 0

    # change only non-lang token element to be an incorrect prediction
    # (predicts vocab element 0 instead of 2)
    perfect_lang_token_logits = perfect_logits.clone().detach()
    perfect_lang_token_logits[0,1,:]=-np.inf
    perfect_lang_token_logits[0,1,0]=1

    perfect_lang_token_loss = WhisperTrainer.compute_lid_loss(perfect_lang_token_logits, labels, lang_ids)
    assert perfect_lang_token_loss == 0

    perfect_lang_token_loss_colwise = WhisperTrainer.compute_lid_loss(perfect_lang_token_logits, labels, lang_ids, colwise=True)
    assert perfect_lang_token_loss_colwise == 0

    # change lang token in row 1 col 1
    wrong_lang_token_col1_row1 = perfect_logits.clone().detach()
    wrong_lang_token_col1_row1[0,0,:]=-np.inf
    wrong_lang_token_col1_row1[0,0,3]=1
    wrong_lang_token_col1_row1_loss = WhisperTrainer.compute_lid_loss(wrong_lang_token_col1_row1, labels, lang_ids)
    assert wrong_lang_token_col1_row1_loss > 0

    # change lang token in row 2 col 2
    wrong_lang_token_col2_row2 = perfect_logits.clone().detach()
    wrong_lang_token_col2_row2[1,1,:]=-np.inf
    wrong_lang_token_col2_row2[1,1,3]=1
    wrong_lang_token_col2_row2_loss = WhisperTrainer.compute_lid_loss(wrong_lang_token_col2_row2, labels, lang_ids)
    # should return zero loss as only first col is checked
    assert wrong_lang_token_col2_row2_loss == 0
    # same logits should return error if we compute lid loss colwise
    wrong_lang_token_col2_loss_colwise = WhisperTrainer.compute_lid_loss(wrong_lang_token_col2_row2, labels, lang_ids, colwise=True)
    assert wrong_lang_token_col2_loss_colwise > 0
    
    # changing all three lang tokens should give greater loss than just one
    wrong_lang_token_col1 = perfect_logits.clone().detach()
    wrong_lang_token_col1[:,0,:]=-np.inf
    wrong_lang_token_col1[:,0,3]=1
    wrong_lang_token_col1_loss = WhisperTrainer.compute_lid_loss(wrong_lang_token_col1, labels, lang_ids)
    assert wrong_lang_token_col1_loss > 0
    assert wrong_lang_token_col1_loss > wrong_lang_token_col1_row1_loss
    # colwise loss should also be less than first col only
    wrong_lang_token_loss_col1_colwise = WhisperTrainer.compute_lid_loss(wrong_lang_token_col1, labels, lang_ids, colwise=True)
    assert wrong_lang_token_loss_col1_colwise < wrong_lang_token_col1_loss

    # # second row contains codeswitching
    labels = torch.tensor([
        [4, 2, 2],
        [4, 3, 2]
    ], dtype=torch.long)
    perfect_logits = torch.nn.functional.one_hot(labels).float()
    perfect_logits[perfect_logits==0]=-np.inf
    perfect_logits.requires_grad=True
    lang_ids = [3,4]
    
    # should give zero loss for colwise case only
    perfect_loss = WhisperTrainer.compute_lid_loss(perfect_logits, labels, lang_ids)
    assert perfect_loss > 0 

    perfect_loss_colwise = WhisperTrainer.compute_lid_loss(perfect_logits, labels, lang_ids, colwise=True)
    assert perfect_loss_colwise == 0

    # if we set equal likelihood to both lang tokens in col1
    # colwise will return non-zero loss, first col will not
    equal_prob_cs = perfect_logits.clone().detach()
    equal_prob_cs[1,0,:]=-np.inf
    equal_prob_cs[1,0,3]=1
    equal_prob_cs[1,0,4]=1
    equal_prob_cs_loss = WhisperTrainer.compute_lid_loss(equal_prob_cs, labels, lang_ids)
    # assert equal_prob_cs_loss == 0
    equal_prob_cs_loss_colwise = WhisperTrainer.compute_lid_loss(equal_prob_cs, labels, lang_ids, colwise=True)
    assert equal_prob_cs_loss_colwise > equal_prob_cs_loss
    # nvm I don't understand cross entropy

def test_experiment_json(tmpdir):
    """
    Train model and check that `experiment.json` is saved successfully
    """
    parser = init_parser()
    args = parser.parse_args([])
    args.output = str(tmpdir)
    args.dataset = TIRA_ASR_DS
    args.language = ['sw']
    args.num_records = 10
    args.model = 'openai/whisper-tiny'
    args.num_train_epochs = 2

    train(args)
    json_path = str(tmpdir/'experiment.json')
    assert os.path.exists(json_path)
    with open(json_path) as f:
        exp_json = json.load(f)
    assert exp_json['experiment_name'] == tmpdir.basename
    assert exp_json['experiment_path'] == str(tmpdir)
    assert exp_json['num_train_epochs'] == 2
    assert exp_json['base_checkpoint'] == 'openai/whisper-tiny'

    execution_list = exp_json['executions']
    assert type(execution_list) is list
    assert len(execution_list) == 1
    assert type(execution_list[0]) is dict
    assert type(execution_list[0]['uuid']) is str
    assert type(execution_list[0]['start_time']) is str
    assert type(execution_list[0]['argv']) is str

    train_data = exp_json['train_data']
    assert type(train_data) is list
    assert len(train_data) == 1
    assert type(train_data[0]) is dict
    assert train_data[0]['dataset'] == os.path.basename(TIRA_ASR_DS)
    assert train_data[0]['dataset_path'] == TIRA_ASR_DS
    assert train_data[0]['num_records'] == 10

    train_events = exp_json['train_events']
    assert type(train_events) is list
    assert len(train_events) > 1
    for event in train_events:
        assert 'tag' in event
        assert 'value' in event
        assert 'step' in event
        assert 'uuid' in event
        assert 'start_time' in event

    val_data = exp_json['val_data']
    assert type(val_data) is list
    assert len(val_data) == 1
    assert val_data[0]['dataset'] == os.path.basename(TIRA_ASR_DS)
    assert val_data[0]['dataset_path'] == TIRA_ASR_DS
    assert val_data[0]['num_records'] == 10

    val_events = val_data[0]['events']
    assert type(val_events) is list
    assert len(val_events) > 1
    for event in val_events:
        assert 'tag' in event
        assert 'value' in event
        assert 'step' in event
        assert 'uuid' in event
        assert 'start_time' in event