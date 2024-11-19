from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Optional, Union
import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from sklearn.linear_model import LogisticRegression
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import AutomaticSpeechRecognitionPipeline, Seq2SeqTrainer, WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
import pickle

DEVICE = 0 if torch.cuda.is_available() else -1
device_type = lambda s: int(s) if s!='cpu' else s

# ---------------------- #
# custom trainer objects #
# ---------------------- #

class WhisperTrainer(Seq2SeqTrainer):
    def __init__(self, *args, token_id_to_train=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_id_to_train = token_id_to_train  # ID of the embedding vector to train

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if 'forced_decoder_ids' in inputs:
            # need to set one array of ids as the decoder prompt for whole batch
            # expected shape is [(1, ID), (2, ID), ...]
            forced_decoder_ids = [(i+1, tok_id) for i, tok_id in enumerate(inputs.pop('forced_decoder_ids')[0])]
            gen_kwargs['forced_decoder_ids']=forced_decoder_ids
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )

    def training_step(
            self,
            model: torch.nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]]
        ) -> torch.Tensor:
        # don't pass forced_decoder_ids during training
        inputs.pop('forced_decoder_ids', None)
        loss = super().training_step(model, inputs)
        # Apply gradient masking for the decoder embeddings
        if self.token_id_to_train is not None:
            decoder_input_embeddings = model.model.decoder.embed_tokens.weight
            # Create a mask with gradients for only the token to be trained
            mask = torch.zeros_like(decoder_input_embeddings)
            mask[self.token_id_to_train] = 1

            with torch.no_grad():
                decoder_input_embeddings.grad *= mask  # Mask gradients
        return loss

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


def get_forced_decoder_ids(args, tokenizer, ids_only=False):
    """
    Get task and language prompt tokens for languages specified by `args.language`
    and task 'transcribe'. By default returns a list of tuples, [(i, token_id), ...].
    If `ids_only`, pass a list of token ids sorted by `i`.
    """
    forced_decoder_ids=set()
    for language in args.language or [None]:
        forced_decoder_ids.update(
                tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
        )
    forced_decoder_ids=list(forced_decoder_ids)
    if ids_only:
        forced_decoder_ids.sort(key=lambda t:t[0])
        forced_decoder_ids=[t[1] for t in forced_decoder_ids]
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
    if args.peft_type and args.peft_type.lower() == 'lora':
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
    elif args.peft_type and args.peft_type.lower() == 'lang_token':
        assert len(args.language)==1, "Exactly one language must be passed when fine-tuning language token."
        print(f"Freezing all parameters except for {args.language[0]} language token.")
        for name, param in model.named_parameters():
            if name == 'model.decoder.embed_tokens.weight':
                param.requires_grad=True
            else:
                param.requires_grad=False
        
    return model

def prepare_trainer_for_peft(args, trainer: WhisperTrainer, processor: WhisperProcessor):
    if args.peft_type == 'lang_token':
        lang = args.language[0]
        decoder_ids = processor.get_decoder_prompt_ids(language=lang)
        # returns list [(1, LANG_ID), (2, TRANSCRIBE_ID), (3, NOTIMESTAMPS_ID)]
        lang_id = decoder_ids[0][1]
        trainer.token_id_to_train = lang_id
        return trainer
    return trainer


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
