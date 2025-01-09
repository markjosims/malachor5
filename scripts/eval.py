from string_norm import has_unicode, is_en_word, get_remove_oov_char_funct
from jiwer.process import process_words
from typing import Union, List, Dict
from collections import defaultdict

remove_nontira_chars = get_remove_oov_char_funct('meta/tira_asr_unique_chars.txt')

def has_tira_chars(s: str) -> bool:
    return s==remove_nontira_chars(s)

def get_word_language(word: str) -> str:
    if has_unicode(word) and has_tira_chars(word):
        return 'tira'
    elif is_en_word(word):
        return 'eng'
    else:
        return 'misc'

def get_wer_by_language(reference: Union[str, List[str]], hypothesis: Union[str, List[str]]):
    """
    Returns dictionary with language-specific edit metrics.
    Return keys are as follows:
    - 'num_tira': number of tira words in reference
    - 'pct_tira': proportion of tira words in reference
    - 'tira_insertions': number of tira words inserted
    - 'tira_deletions': number of tira words deleted
    - 'tira2eng_substitutions': number of tira words substituted with English words
    - 'tira2misc_substitutions': number of tira words substituted with non-English words
    - 'tira2tira_substitutions': number of tira words substituted with other tira words
    - 'tira_hits': number of tira words that are correct
    - 'num_eng': number of English words in reference
    - 'pct_eng': proportion of English words in reference
    - 'eng_insertions': number of English words inserted
    - 'eng_deletions': number of English words deleted
    - 'eng2tira_substitutions': number of English words substituted with tira words
    - 'eng2misc_substitutions': number of English words substituted with non-English
    - 'eng2eng_substitutions': number of English words substituted with other English words
    - 'eng_hits': number of English words that are correct
    
    Plus, for each string edit key, there is an additional key with the same name but with '_rate' appended,
    giving the number of edits as a portion of the total number of words for that language in the reference.
    """
    if type(reference) is str:
        reference = [reference,]
        hypothesis = [hypothesis,]
    metric_list = []
    output = process_words(reference, hypothesis)
    alignments = output.alignments
    for ref, hyp, align in zip(reference, hypothesis, alignments):
        ref_words = ref.split()
        hyp_words = hyp.split()
        metrics = metric_factory(output)

        for aligned_word in align:
            alignment_metrics = get_metrics_from_alignment(aligned_word, ref_words, hyp_words)
            for key, value in alignment_metrics.items():
                metrics[key] += value
            # calculate rates
            for k, v in metrics.copy().items():
                if k.startswith('num_'):
                    lang = 'tira' if k.endswith('tira') else 'eng'
                    total = len(ref)
                    metrics[f'pct_{lang}'] = v / total
                elif any(k.endswith(name) for name in ['substitutions', 'deletions', 'hits']):
                    lang = 'tira' if k.startswith('tira') else 'eng'
                    total = metrics[f'num_{lang}']
                    metrics[f'{k}_rate'] = v / total
                elif k.endswith('insertion'):
                    lang = 'tira' if k.startswith('tira') else 'eng'
                    total = len(hyp)
                    metrics[f'{k}_rate'] = v / total
        metric_list.append(metrics)
    return metric_list

def get_metrics_from_alignment(align, ref_words, hyp_words) -> Dict[str, int]:
    alignment_metrics = defaultdict(lambda:0)
    align_type = align.type
    if align_type == 'substitute':
        for ref_idx, hyp_idx in zip(
            range(align.ref_start_idx, align.ref_end_idx),
            range(align.hyp_start_idx, align.hyp_end_idx),
        ):
            ref_word = ref_words[ref_idx]
            hyp_word = hyp_words[hyp_idx]
            ref_lang = get_word_language(ref_word)
            hyp_lang = get_word_language(hyp_word)
            alignment_metrics[f'num_{ref_lang}'] += 1
            alignment_metrics[f'{ref_lang}2{hyp_lang}_substitutions'] += 1
    elif align_type == 'equal':
        for ref_idx in range(align.ref_start_idx, align.ref_end_idx):
            ref_word = ref_words[ref_idx]
            ref_lang = get_word_language(ref_word)
            alignment_metrics[f'num_{ref_lang}'] += 1
            alignment_metrics[f'{ref_lang}_hits'] += 1
    elif align_type == 'insert':
        for hyp_idx in range(align.hyp_start_idx, align.hyp_end_idx):
            hyp_word = hyp_words[hyp_idx]
            hyp_lang = get_word_language(hyp_word)
            alignment_metrics[f'num_{hyp_lang}'] += 1
            alignment_metrics[f'{hyp_lang}_insertions'] += 1
    else: # align_type == 'deletion'
        for ref_idx in range(align.ref_start_idx, align.ref_end_idx):
            ref_word = ref_words[ref_idx]
            ref_lang = get_word_language(ref_word)
            alignment_metrics[f'num_{ref_lang}'] += 1
            alignment_metrics[f'{ref_lang}_deletions'] += 1

    return alignment_metrics


def metric_factory(jiwer_output):
    metrics = {
            'num_tira':                   0,
            'tira_insertions':            0,
            'tira_deletions':             0,
            'tira2eng_substitutions':     0,
            'tira2misc_substitutions':    0,
            'tira2tira_substitutions':    0,
            'tira_hits':                  0,
            'num_eng':                    0,
            'eng_deletions':              0,
            'eng_insertions':             0,
            'eng2tira_substitutions':     0,
            'eng2misc_substitutions':     0,
            'eng2eng_substitutions':      0,
            'eng_hits':                   0,
            'wer':                        jiwer_output.wer,
            'mer':                        jiwer_output.mer,
            'wil':                        jiwer_output.wil,
            'wip':                        jiwer_output.wip,
        }
    
    return metrics