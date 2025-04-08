# weights for loss regularization functions
LOSS_REGULARIZATION_HYPERPARAMS = {
    'lid_loss_alpha': {'type': float},
    'ewc_lambda': {'type': float}
}

# parameters associated with language prompt tuning
PROMPT_TUNING_HYPERPARAMS = {
    'mean_embed_path': {'type': str},
    'embed_dist_lambda': {'type': float, 'default': 1},
    'embed_dist_type': {'choices': ['euclidean', 'cosine'], 'default': 'euclidean'},
}

# output paths for actions other than train/evaluate/test
EXTRA_OUTPUT_ARGS = {
    'lid_logits_path': {'type': str, 'help': "For args.action=='get_lid_probs'"},
    'fisher_matrix_path': {'type': str, 'help': "Outpu path for fisher matrix for args.action=='calculate_fisher, read path for ewc regularization"},
}

EVAL_ARGS = {
    'eval_checkpoints': {
        'nargs':'+',
        'help': "For evaluating multiple checkpoints. To evaluate all checkpoints, pass 'all', else pass a list of checkpoint numbers (NOT paths)."
    },
    'eval_output': {'type': str, 'help': 'Path to save evaluation output and predictions to. If training, evaluate last checkpoint.'},
}

# args for LM-boosted decoding
LM_ARGS = {
    'lm': {'nargs':'+', 'type': str},
    'lm_betas': {'nargs':'+', 'type': float},
    'lm_alpha': {'type':float, 'default': 0.5},
    'lm_input': {'choices':['text', 'tokens'], 'default': 'text'},
}

TRAIN_PROG_ARGS = {
    'output': {'abbreviation': 'o', 'help': 'Directory to save model to'},
    'ft_peft_model': {'action': 'store_true'},
    'resume_from_checkpoint': {'action': 'store_true'},
    'checkpoint': {'type': int},
    'action': {'choices': ['train', 'evaluate', 'test', 'calculate_fisher', 'get_lid_probs'], 'default': 'train'},
}

# hyperparams associated with `transformers.Seq2SeqTrainingArguments`
DEFAULT_TRAINER_HYPERPARAMS = {
    'group_by_length': False,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_accumulation_steps': 16,
    'eval_strategy': "epoch",
    'save_strategy': "epoch",
    'num_train_epochs': 4,
    'gradient_checkpointing': False,
    'fp16': False,
    'save_steps': 5000,
    'eval_steps': 5000,
    'logging_steps': 1000,
    'learning_rate': 3e-4,
    'warmup_steps': 500,
    'report_to': 'tensorboard',
    'predict_with_generate': False,
    'generation_num_beams': 1,
    'remove_unused_columns': False,
    'eval_on_start': False,
    'use_cpu': False,
}
# stored again as these impact eval in a way other args don't
GENERATE_ARGS = ['generation_num_beams', 'predict_with_generate', 'prompt_file']
TRAINER_HYPERPARAM_ABBREVIATIONS = {
    'per_device_train_batch_size': 'b',
    'per_device_eval_batch_size': 'B',
    'num_train_epochs': 'e',
    'gradient_accumulation_steps': 'g',
}

def get_hyperparam_argdict():
    """
    Return a dictioanry of shape {
        `arg_name`: {
            'abbreviation': str,
            'default', val,
            ?'action': str,
            ?'type': type,
        }
    }
    for trainer hyperparams
    """
    argdict = {}
    for k, v in DEFAULT_TRAINER_HYPERPARAMS.items():
        k_dict = {'default': v}

        if k in TRAINER_HYPERPARAM_ABBREVIATIONS:
            k_dict['abbreviation']=TRAINER_HYPERPARAM_ABBREVIATIONS[k]
        if type(v) is bool:
            k_dict['action']='store_true'
        else:
            k_dict['type']=type(v)
        argdict[k]=k_dict
    return argdict