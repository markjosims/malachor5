from transformers import WhisperProcessor, WhisperForConditionalGeneration, LogitsProcessorList, LogitsProcessor
import torch
import sys
sys.path.append('scripts')
from longform import load_and_resample
import kenlm

class LanguageModelRescorer(LogitsProcessor):
    def __init__(self, tokenizer, lm_path, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # Weight for LM fusion
        self.tokenizer = tokenizer
        self.lm = kenlm.LanguageModel(lm_path)

    def __call__(self, input_ids, scores):
        """Modify logits using LM-based rescoring."""
        hypotheses = self.tokenizer.batch_decode(input_ids)

        lm_scores = [self.lm.score(hyp) for hyp in hypotheses]
        lm_adjustment = torch.tensor(lm_scores, device=scores.device).unsqueeze(1) * self.alpha
        scores = scores + lm_adjustment

        return scores

if __name__ == '__main__':
    # Load Whisper model and processor
    model_id = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Prepare input
    audio_path = '/Users/markjos/projects/malachor5/data/SASOC/audio/AKHONA_12-10-31_24.wav'
    audio_input = load_and_resample(audio_path, flatten=True)
    input_features = processor(audio_input, return_tensors="pt", sampling_rate=16_000).input_features

    # prompt=processor.tokenizer.get_decoder_prompt_ids()
    # prompt=torch.tensor(prompt)
    # output = model.generate(input_features, num_beams=2)
    # decoded_text = processor.batch_decode(output, skip_special_tokens=False)
    
    sasoc_3gram_path = '/Users/markjos/projects/malachor5/data/SASOC/sasoc_3gram.arpa'
    alpha = 0.5
    lm_rescorer = LanguageModelRescorer(processor.tokenizer, sasoc_3gram_path, alpha=alpha)
    logits_processor = LogitsProcessorList([lm_rescorer])

    num_beams=4
    output = model.generate(input_features, logits_processor=logits_processor, num_beams=num_beams)
    decoded_text = processor.batch_decode(output, skip_special_tokens=False)
    print(f"LM rescoring {alpha=}, {num_beams=}")
    print(decoded_text)

    output = model.generate(input_features, num_beams=num_beams)
    decoded_text = processor.batch_decode(output, skip_special_tokens=False)
    print(f"No LM, {num_beams=}")
    print(decoded_text)