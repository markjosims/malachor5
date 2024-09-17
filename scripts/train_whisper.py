from argparse import ArgumentParser
from typing import Sequence, Optional, Dict, Any

DEFAULT_HYPERPARAMS = {
    'group_by_length': True,
    'per_device_train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'evaluation_strategy': "epoch",
    'save_strategy': "epoch",
    'num_train_epochs': 4,
    'gradient_checkpointing': True,
    'fp16': False,
    'save_steps': 5000,
    'eval_steps': 5000,
    'logging_steps': 1000,
    'learning_rate': 3e-4,
    'warmup_steps': 500,
    'report_to': 'tensorboard',
    'debug': 'underflow_overflow',
}

# ---------------- #
# Argparse methods #
# ---------------- #

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--model', '-m')
    add_hyperparameter_args(parser)
    return parser

def add_hyperparameter_args(parser: ArgumentParser) -> None:
    hyper_args = parser.add_argument_group(
        'hyperparameters',
        description='Hyperparameter values for training'
    )
    add_args_from_dict(DEFAULT_HYPERPARAMS, hyper_args)

def add_args_from_dict(args_dict: Dict[str, Any], arg_group: ArgumentParser) -> None:
    add_arg = lambda *args, **kwargs: arg_group.add_argument(*args, **kwargs)
    for k, v in args_dict.items():
        if type(v) is bool:
            add_arg('--'+k, default=v, action='store_true')
        else:
            add_arg('--'+k, type=type(v), default=v)

# ---- #
# main #
# ---- #

def main(argv: Sequence[Optional[str]]=None) -> int:
    return 0

if __name__ == '__main__':
    main()