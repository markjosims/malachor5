from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple, Optional, Union, Literal
import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from sklearn.linear_model import LogisticRegression
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
)
from transformers.modeling_outputs import BaseModelOutput
import pickle
from tokenization_utils import get_forced_decoder_ids, LANG_TOKEN_IDS
import numpy as np
import kenlm
from datasets import Dataset


DEVICE = 0 if torch.cuda.is_available() else -1
LOGREG_PATH = 'models/voxlingua_logreg.pkl'
LR_ARG_DEFAULTS = {
    'batch_size': 8,
    'device': DEVICE,
    'sb_savedir': 'models/speechbrain',
}
device_type = lambda s: int(s) if s!='cpu' else s

# ---------------------- #
# custom trainer objects #
# ---------------------- #

class LanguageModelRescorer(LogitsProcessor):
    def __init__(self, tokenizer, lm_path_list, lm_betas=None, alpha=0.5, lm_input: Literal['text', 'tokens']='text'):
        super().__init__()
        self.alpha = alpha  # Weight for LM fusion
        self.tokenizer = tokenizer
        self.lm_list = [kenlm.LanguageModel(lm_path) for lm_path in lm_path_list]
        if len(self.lm_list)>1:
            if len(lm_betas)!=len(self.lm_list)-1:
                raise ValueError("Must return one beta value less than number of lms")
            lm_betas.append(1-sum(lm_betas))
            self.lm_betas = lm_betas
        self.lm_input = lm_input

    def score_str(self, hyp_str, eos):
        if len(self.lm_list) == 1:
            return self.lm_list[0].score(hyp_str)
        unweighted_scores = [lm.score(hyp_str, eos=eos) for lm in self.lm_list]
        weighted_scores = [score*beta for score,beta in zip(unweighted_scores, self.lm_betas)]
        return sum(weighted_scores)

    def __call__(self, input_ids, scores):
        """Modify logits using LM-based rescoring."""
        eos_list = [self.tokenizer.eos_token_id in hyp for hyp in input_ids]
        if self.lm_input == 'text':
            text_hypotheses = self.tokenizer.batch_decode(input_ids)
            lm_scores = [self.score_str(hyp, eos) for hyp, eos in zip(text_hypotheses, eos_list)]
        else:
            token_hypotheses = [
                ' '.join([str(id) for id in hyp if id not in self.tokenizer.all_special_ids])
                for hyp in input_ids.tolist()
            ]
            lm_scores = [self.score_str(hyp, eos) for hyp, eos in zip(token_hypotheses, eos_list)]
            
        lm_adjustment = torch.tensor(lm_scores, device=scores.device).unsqueeze(1) * self.alpha
        scores = scores + lm_adjustment

        return scores

class WhisperTrainer(Seq2SeqTrainer):
    def __init__(
            self,
            *args,
            token_id_to_train=None,
            fisher_matrix_path=None,
            ewc_lambda=1,
            mean_embed_path=None,
            embed_dist_lambda=1,
            embed_dist_type: Literal['euclidean', 'cosine']='euclidean',
            lid_loss_alpha=None,
            lm_path=None,
            lm_betas=None,
            lm_alpha=0.5,
            lm_input: Literal['text', 'tokens']='text',
            tokenizer=None,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.token_id_to_train = token_id_to_train  # ID of the embedding vector to train
        self.fisher_matrix_path = fisher_matrix_path
        self.ewc_lambda = ewc_lambda
        self.embed_dist_lambda = embed_dist_lambda
        self.embed_dist_type = embed_dist_type
        self.lid_loss_alpha = lid_loss_alpha
        self.lm_path = lm_path
        self.lm_alpha = lm_alpha
        
        if fisher_matrix_path is not None:
            self.fisher_matrix = torch.load(fisher_matrix_path, map_location=kwargs['args'].device)
            print(f"Loading fisher matrix from path {fisher_matrix_path}...")
            self.previous_params = {name: param.clone().detach() for name, param in self.model.named_parameters() if param.requires_grad}
        else:
            self.fisher_matrix = None
            self.previous_params = None
        if mean_embed_path is not None:
            print(f"Loading mean embedding from path {mean_embed_path} for {embed_dist_type} regularization...")
            self.mean_embed = torch.load(mean_embed_path, map_location=kwargs['args'].device)
        else:
            self.mean_embed = None
        if lm_path is not None:
            self.lm_rescorer = LanguageModelRescorer(tokenizer, lm_path_list=lm_path, alpha=lm_alpha, lm_betas=lm_betas, lm_input=lm_input)
        else:
            self.lm_rescorer = None
        

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
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
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        if self.lm_rescorer is not None:
            logits_processor = LogitsProcessorList([self.lm_rescorer])
            gen_kwargs['logits_processor'] = logits_processor
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **gen_kwargs,
        )

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ):
        if self.lm_rescorer is not None:
            logits_processor = LogitsProcessorList([self.lm_rescorer])
            gen_kwargs['logits_processor'] = logits_processor
        return super().predict(
            test_dataset=test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **gen_kwargs,
        )

    def training_step(
            self,
            model: torch.nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            num_items_in_batch=None,
        ) -> torch.Tensor:
        # don't pass forced_decoder_ids during training
        inputs.pop('forced_decoder_ids', None)
        loss = super().training_step(model, inputs, num_items_in_batch)
        # Apply gradient masking for the decoder embeddings
        if self.token_id_to_train is not None:
            decoder_input_embeddings = model.model.decoder.embed_tokens.weight
            # Create a mask with gradients for only the token to be trained
            mask = torch.zeros_like(decoder_input_embeddings)
            mask[self.token_id_to_train] = 1

            with torch.no_grad():
                decoder_input_embeddings.grad *= mask  # Mask gradients
        return loss
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if return_outputs or (self.lid_loss_alpha is not None):
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs, **kwargs)
        if self.fisher_matrix is not None and self.previous_params is not None:
            # EWC Regularization Term
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if name in self.fisher_matrix:
                    fisher = self.fisher_matrix[name]
                    prev_param = self.previous_params[name]
                    ewc_loss += (fisher * (param - prev_param).pow(2)).sum()
            
            loss = loss + (self.ewc_lambda * ewc_loss)
        
        if self.mean_embed is not None:
            token_embed = model.model.decoder.embed_tokens.weight[self.token_id_to_train]
            if self.embed_dist_type == 'cosine':
                y = torch.tensor([1], device=token_embed.device) # maximize cosine similarity
                embed_dist_loss = torch.nn.functional.cosine_embedding_loss(token_embed.unsqueeze(0), self.mean_embed.unsqueeze(0), y)
            else:
                # self.embed_dist_type == 'euclidean'
                embed_dist_loss = torch.subtract(token_embed, self.mean_embed).pow(2).sum().sqrt()
            loss = loss + (self.embed_dist_lambda * embed_dist_loss.pow(2))

        if self.lid_loss_alpha is not None:
            lang_ids = list(self.model.generation_config.lang_to_id.values())
            lid_loss = self.compute_lid_loss(outputs.logits, inputs.labels, lang_ids)
            loss = (1-self.lid_loss_alpha)*loss + self.lid_loss_alpha*lid_loss
        if return_outputs:
            return loss, outputs
        return loss

    @staticmethod
    def compute_lid_loss(logits, labels, lang_ids, colwise=True):
        lid_label_mask, lang_prefix_mask, max_lang_token_ct = WhisperTrainer.get_logit_masks_for_lid(labels, lang_ids)
        ground_truth_lid_mat = WhisperTrainer.get_lid_labels(
            labels=labels,
            lang_ids=lang_ids,
            lid_label_mask=lid_label_mask,
            lang_prefix_mask=lang_prefix_mask,
            num_classes=logits.shape[-1],
            colwise=colwise,
        )
        if colwise:
            lid_probs = torch.nn.functional.softmax(
                logits[lang_prefix_mask].view(
                    lang_prefix_mask.shape[0], 
                    max_lang_token_ct,
                    logits.shape[-1],
                )[lid_label_mask][:,lang_ids],
                dim=1
            )
        else:
            lid_probs = torch.nn.functional.softmax(
                logits[:,0,lang_ids],
                dim=1,
            )
        # breakpoint()
        lid_loss = torch.nn.functional.cross_entropy(
            lid_probs,
            ground_truth_lid_mat
        )
        return lid_loss
    
    @staticmethod
    def get_lid_labels(
        labels: torch.Tensor,
        num_classes: int,
        lang_ids,
        lang_prefix_mask=None,
        lid_label_mask=None,
        colwise=False
    ):
        if (lid_label_mask is None) or (lang_prefix_mask is None):
            lid_label_mask, lang_prefix_mask, _ = WhisperTrainer.get_logit_masks_for_lid(labels, lang_ids)
        one_hot_tensors = [
            torch.nn.functional.one_hot(
                label[lang_prefix_mask[i]].long(),
                num_classes=num_classes,
            ).float()
            for i, label in enumerate(labels)
        ]
        ground_truth_lid_mat = torch.stack(one_hot_tensors)
        # zero out positions in possible lang token prefix search space
        # tha't dont have a language token
        ground_truth_lid_mat[~lid_label_mask]=0
        if colwise:
            return ground_truth_lid_mat[lid_label_mask][:,lang_ids]
        ground_truth_lid_mat=ground_truth_lid_mat.sum(dim=1)
        langs_per_row = ground_truth_lid_mat.sum(dim=1).unsqueeze(dim=1)
        ground_truth_lid_mat/=langs_per_row
    
        return ground_truth_lid_mat[:,lang_ids]

    @staticmethod
    def get_logit_masks_for_lid(labels, lang_ids):
        lid_label_mask = torch.stack(
            [labels==lang_id for lang_id in lang_ids]
        ).sum(dim=0).bool()
        # there may be some rows with multiple lang tokens
        # and some rows with one
        # to make sure tensors are right shape, make new mask over the first N
        # tokens of each row for N=max_[batch](num lang tokens)
        max_lang_token_ct = lid_label_mask.sum(dim=1).max().item()
        lang_prefix_mask = torch.zeros_like(lid_label_mask)
        lang_prefix_mask[:,:max_lang_token_ct]=1
        lang_prefix_mask=lang_prefix_mask.bool()
        # now trim lid_label_mask to size of search space for language tokens
        lid_label_mask = lid_label_mask[:,:max_lang_token_ct]
        return lid_label_mask, lang_prefix_mask, max_lang_token_ct
    
    def get_lid_logits(
            self,
            logits: torch.FloatTensor,
            generation_config: Optional[GenerationConfig] = None,
        ) -> torch.Tensor:
        """
        Same as `WhisperGenerationMixin.detect_language` from `generation_whisper.py`
        except returns a tensor of LID logits for the input batch rather than the argmax.
        """
        generation_config = generation_config or self.model.generation_config
        non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
        non_lang_mask[list(generation_config.lang_to_id.values())] = False

        logits[:, non_lang_mask] = -np.inf
        return logits

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
    if getattr(args, 'peft_type', None) or getattr(args, 'peft', None):
        model = load_whisper_peft(args)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model)
    tokenizer = WhisperTokenizer.from_pretrained(args.model)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        batch_size=getattr(args, 'batch_size', 8),
        device=getattr(args, 'device', DEVICE),
        chunk_length_s=getattr(args, 'chunk_length_s', None),
    )
    return pipe


def load_peft_model_for_finetuning(args):
    model = load_whisper_peft(args)
    print("Merging PEFT model for further finetuning...")
    model = model.merge_and_unload()
    return model


def set_generation_config(args, model, tokenizer):
    forced_decoder_ids=get_forced_decoder_ids(tokenizer, args.language)
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    return model


def load_whisper_model_for_training_or_eval(args) -> WhisperForConditionalGeneration:
    if args.ft_peft_model:
        model = load_peft_model_for_finetuning(args)
    elif args.action in ('evaluate', 'test') and args.peft_type:
        return load_whisper_peft(args)
    elif args.checkpoint and not args.peft_type:
        model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint)
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
    if args.peft_type and args.peft_type.lower() == 'lang_token':
        lang = args.language[0]
        decoder_ids = processor.get_decoder_prompt_ids(language=lang)
        # returns list [(1, LANG_ID), (2, TRANSCRIBE_ID), (3, NOTIMESTAMPS_ID)]
        lang_id = decoder_ids[0][1]
        trainer.token_id_to_train = lang_id
        return trainer
    return trainer


def sb_model(args):
    if args.device==-1:
        # torch uses -1 to represent CPU, SpeechBrain uses 'cpu'
        args.device='cpu'
    model = EncoderClassifier.from_hparams(
        source=args.sli_embed_model,
        savedir=args.sb_savedir,
        run_opts={"device":torch.device(args.device)},
    )
    return model


def load_lr(lr_model: Optional[str]=None, args: Optional[Namespace]=None, **kwargs) -> Tuple[LogisticRegression, Namespace]:
    if lr_model is None:
        lr_model = args.lr_model
    with open(lr_model, 'rb') as f:
        lr_dict = pickle.load(f)
    lr_obj=lr_dict['lr_model']
    if args is None:
        for k, v in LR_ARG_DEFAULTS.items():
            if k not in kwargs:
                kwargs[k]=v
        args=Namespace(**kwargs)
    args.embed_model=lr_dict['embed_model']
    args.embed_api=lr_dict['embed_api']
    args.sli_map=lr_dict['sli_map']
    args.sli_id2label=lr_dict['sli_id2label']
    args.sli_label2id=lr_dict['sli_label2id']
    return lr_obj, args


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
    parser.add_argument('--peft_type', choices=['LoRA', 'lang_token'])
    return parser


def add_sli_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--sli_embed_model')
    parser.add_argument('--embed_api', choices=['hf', 'sb'], default='sb')
    parser.add_argument('--sb_savedir', default='models/speechbrain')
    return parser
