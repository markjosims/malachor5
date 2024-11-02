from argparse import ArgumentParser
import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from sklearn.linear_model import LogisticRegression
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import AutomaticSpeechRecognitionPipeline, WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer
import pickle

DEVICE = 0 if torch.cuda.is_available() else -1
device_type = lambda s: int(s) if s!='cpu' else s

# ----------------- #
# model preparation #
# ----------------- #

def load_whisper_peft(args) -> WhisperForConditionalGeneration:
    model_path = args.checkpoint or args.model
    peft_config = PeftConfig.from_pretrained(model_path)
    model_basename = peft_config.base_model_name_or_path
    print(f"Loading adapters from {model_path} for model {model_basename}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_basename,
    )
    model = PeftModel.from_pretrained(model, model_path)
    return model


def load_whisper_pipeline(args) -> AutomaticSpeechRecognitionPipeline:
    if args.peft:
        model = load_whisper_peft(args)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model)
    tokenizer = WhisperTokenizer.from_pretrained(args.model)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        batch_size=args.batch_size,
        device=args.device,
        chunk_length_s=getattr(args, 'chunk_length_s', None),
    )
    return pipe


def load_peft_model_for_finetuning(args):
    model = load_whisper_peft(args)
    print("Merging PEFT model for further finetuning...")
    model = model.merge_and_unload()
    return model


def get_forced_decoder_ids(args, tokenizer):
    forced_decoder_ids=set()
    for language in args.language or [None]:
        forced_decoder_ids.update(
                tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
            )
    forced_decoder_ids=list(forced_decoder_ids)
    return forced_decoder_ids


def set_generation_config(args, model, tokenizer):
    forced_decoder_ids=get_forced_decoder_ids(args, tokenizer)
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    return model


def load_whisper_model_for_training_or_eval(args) -> WhisperForConditionalGeneration:
    if args.ft_peft_model:
        model = load_peft_model_for_finetuning(args)
    elif args.action in ('evaluate', 'test') and args.peft_type:
        return load_whisper_peft(args)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model)
    if args.peft_type == 'LoRA':
        print("Wrapping model with LoRA...")
        # TODO add LoRA args to CLI
        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    return model


def sb_model(args):
    model = EncoderClassifier.from_hparams(
        source=args.sli_model,
        savedir=args.sb_savedir,
        run_opts={"device":torch.device(args.device)},
    )
    return model


def load_lr(args) -> LogisticRegression:
    with open(args.lr_model, 'rb') as f:
        lr_dict = pickle.load(f)
    args.sli_model=lr_dict['embed_model']
    args.embed_api=lr_dict['embed_api']
    lr_model=lr_dict['lr_model']
    return args, lr_model


# ---------------- #
# Argparse methods #
# ---------------- #

def add_processor_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--processor')
    parser.add_argument('--g2p', action='store_true')
    parser.add_argument('--transcription_ids', action='store_true')
    parser.add_argument('--label_key', default='transcription')
    parser.add_argument('--language', '-l', nargs='+')
    parser.add_argument('--load_ds_cache', '-c', action='store_true')
    parser.add_argument('--char_vocab')
    parser.add_argument('--condense_tones', action='store_true')
    return parser

def add_whisper_model_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--model', '-m')
    parser.add_argument('--device', '-D', default=DEVICE, type=device_type)
    parser.add_argument('--peft_type', choices=['LoRA'])
    return parser


def add_sli_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--sli_model')
    parser.add_argument('--sli_model_type', choices=['hf', 'sb'], default='sb')
    parser.add_argument('--sb_savedir', default='speechbrain')
    return parser