from string_norm import remove_punct
from jiwer.process import process_words, process_characters
from typing import Union, List, Dict, Literal
from collections import defaultdict
from lid_utils import get_word_language

def get_metrics_by_language(
        reference: Union[str, List[str]],
        hypothesis: Union[str, List[str]],
        metric: Literal['cer', 'wer']='wer',
        ignore_punct: bool = False,
        langs: List[str] = ['tira', 'eng', 'misc'],
        average: bool = False,
    ) -> List[Dict[str, int]]:
    """
    Returns dictionary with language-specific edit metrics.
    Return keys are as follows:
    - 'num_tira': number of tira words in reference
    - 'num_tira_hyp': number of tira words in hypothesis
    - 'pct_tira': proportion of tira words in reference
    - 'tira_insertions': number of tira words inserted
    - 'tira_deletions': number of tira words deleted
    - 'tira2eng_substitutions': number of tira words substituted with English words
    - 'tira2misc_substitutions': number of tira words substituted with non-English words
    - 'tira2tira_substitutions': number of tira words substituted with other tira words
    - 'tira_hits': number of tira words that are correct
    - 'num_eng': number of English words in reference
    - 'num_eng_hyp': number of English words in hypothesis
    - 'pct_eng': proportion of English words in reference
    - 'eng_insertions': number of English words inserted
    - 'eng_deletions': number of English words deleted
    - 'eng2tira_substitutions': number of English words substituted with tira words
    - 'eng2misc_substitutions': number of English words substituted with non-English
    - 'eng2eng_substitutions': number of English words substituted with other English words
    - 'eng_hits': number of English words that are correct
    - 'num_misc_hyp': number of non-English non-Tira words in hypothesis
    
    Plus, for each string edit key, there is an additional key with the same name but with '_rate' appended,
    giving the number of edits as a portion of the total number of words for that language in the reference.
    """
    if 'misc' not in langs:
        # for now, keeping 'misc' category by default
        # to handle cases where LID cannot be established
        langs.append('misc')

    if type(reference) is str:
        reference = [reference,]
        hypothesis = [hypothesis,]
    if ignore_punct:
        reference = [remove_punct(ref) for ref in reference]
        hypothesis = [remove_punct(hyp) for hyp in hypothesis]
    metric_list = []
    output = process_words(reference, hypothesis) if metric=='wer' else\
        process_characters(reference, hypothesis)
    alignments = output.alignments
    for ref, hyp, align in zip(reference, hypothesis, alignments):
        metrics = metric_factory(output, metric=metric, langs=langs)

        for aligned_word in align:
            alignment_metrics = get_metrics_from_alignment(aligned_word, ref, hyp, metric=metric, langs=langs)
            for key, value in alignment_metrics.items():
                metrics[key] += value
        # calculate rates
        for k, v in metrics.copy().items():
            if k.startswith('num_'):
                lang = 'eng' if k.endswith('eng') else k[-4:]
                total = len(ref)
                metrics[f'pct_{lang}'] = v / total if v!=0 else v
            elif any(k.endswith(name) for name in ['substitutions', 'deletions', 'hits']):
                lang = 'eng' if k.startswith('eng') else k[:4]
                total = metrics[f'num_{lang}']
                metrics[f"{k.removesuffix('s')}_rate"] = v / total if v!=0 else v
            elif k.endswith('insertions'):
                lang = 'eng' if k.startswith('eng') else k[:4]
                total = metrics[f'num_{lang}_hyp']
                metrics[f"{k.removesuffix('s')}_rate"] = v / total if v!=0 else v
        # Calculate lang-specific WER
        for lang in langs:
            num_lang = metrics[f'num_{lang}']
            if num_lang == 0:
                metrics[f'{lang}_{metric}'] = 0
            else:
                num_insert = metrics[f"{lang}{'_char' if metric=='cer' else ''}_insertions"]
                num_delete = metrics[f"{lang}{'_char' if metric=='cer' else ''}_deletions"]
                num_sub = sum(metrics[f"{lang}2{lang2}{'_char' if metric=='cer' else ''}_substitutions"] for lang2 in langs)
                metrics[f'{lang}_{metric}'] = (num_insert + num_delete + num_sub) / num_lang
        metric_list.append(metrics)
    if average:
        if len(metric_list)==1:
            return metric_list[0]
        avg_metrics = metric_list[0]
        for metric_obj in metric_list[1:]:
            for k, v in metric_obj.items():
                avg_metrics[k]+=v
        for k, v in avg_metrics:
            avg_metrics[k]=v/len(metric_list)
        return avg_metrics
        
    return metric_list

def get_metrics_from_alignment(
        align,
        ref,
        hyp,
        metric: Literal['cer', 'wer']='wer',
        langs=['tira', 'eng', 'misc'],
    ) -> Dict[str, int]:
    alignment_metrics = defaultdict(lambda:0)
    align_type = align.type
    if align_type == 'substitute':
        for ref_idx, hyp_idx in zip(
            range(align.ref_start_idx, align.ref_end_idx),
            range(align.hyp_start_idx, align.hyp_end_idx),
        ):
            ref_word = get_word_for_alignment(ref, ref_idx, metric)
            hyp_word = get_word_for_alignment(hyp, hyp_idx, metric)
            ref_lang = get_word_language(ref_word, langs=langs)
            hyp_lang = get_word_language(hyp_word, langs=langs)
            alignment_metrics[f'num_{ref_lang}'] += 1
            alignment_metrics[f'num_{hyp_lang}_hyp'] += 1
            alignment_metrics[f"{ref_lang}2{hyp_lang}{'_char' if metric=='cer' else ''}_substitutions"] += 1
    elif align_type == 'equal':
        for ref_idx in range(align.ref_start_idx, align.ref_end_idx):
            ref_word = get_word_for_alignment(ref, ref_idx, metric)
            ref_lang = get_word_language(ref_word, langs=langs)
            alignment_metrics[f'num_{ref_lang}'] += 1
            alignment_metrics[f'num_{ref_lang}_hyp'] += 1
            alignment_metrics[f"{ref_lang}{'_char' if metric=='cer' else ''}_hits"] += 1
    elif align_type == 'insert':
        for hyp_idx in range(align.hyp_start_idx, align.hyp_end_idx):
            hyp_word = get_word_for_alignment(hyp, hyp_idx, metric)
            hyp_lang = get_word_language(hyp_word, langs=langs)
            alignment_metrics[f'num_{hyp_lang}_hyp'] += 1
            alignment_metrics[f"{hyp_lang}{'_char' if metric=='cer' else ''}_insertions"] += 1
    else: # align_type == 'deletion'
        for ref_idx in range(align.ref_start_idx, align.ref_end_idx):
            ref_word = get_word_for_alignment(ref, ref_idx, metric)
            ref_lang = get_word_language(ref_word, langs=langs)
            alignment_metrics[f'num_{ref_lang}'] += 1
            alignment_metrics[f"{ref_lang}{'_char' if metric=='cer' else ''}_deletions"] += 1
    return alignment_metrics

def get_word_for_alignment(s: str, i: int, metric: Literal['cer', 'wer']='wer') -> str:
    if metric == 'cer':
        return get_word_from_char_i(s, i)
    return s.split()[i]

def get_word_from_char_i(s: str, i: int) -> str:
    word = s[i]
    if word.isspace():
        return word
    j=i-1
    while j >= 0 and not s[j].isspace():
        word = s[j] + word
        j -= 1
    j=i+1
    while j < len(s) and not s[j].isspace():
        word += s[j]
        j += 1
    return word

def metric_factory(
        jiwer_output,
        metric: Literal['cer', 'wer']='wer',
        langs: List[str]=['tira', 'eng', 'misc']
    ) -> Dict[str, int]:
    edits = ['insertions', 'deletions', 'hits']
    if metric == 'cer':
        edits = ['char_' + edit for edit in edits]
    if metric == 'cer':
        metrics = {'cer': jiwer_output.cer}
    else:
        metrics = {
            'wer': jiwer_output.wer,
            'mer': jiwer_output.mer,
            'wil': jiwer_output.wil,
            'wip': jiwer_output.wip,
        }
    for lang in langs:
        for edit in edits:
            metrics[f'{lang}_{edit}'] = 0
        for lang2 in langs:
            metrics[f"{lang}2{lang2}{'_char' if metric=='cer' else ''}_substitutions"] = 0
        metrics[f'num_{lang}'] = 0
        metrics[f'num_{lang}_hyp'] = 0
    
    return metrics