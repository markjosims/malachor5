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