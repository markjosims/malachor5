from typing import Dict, Sequence, Any, Literal, List, Tuple, Callable, Optional
from string import punctuation
import unicodedata
import json
import epitran
import enchant
from unidecode import unidecode

"""
Copied from zugubul
"""

DIACS = ['grave', 'macrn', 'acute', 'circm', 'caron', 'tilde',]

en_dict = enchant.request_dict('en_US')

COMBINING = {
    'grave': "\u0300",
    'macrn': "\u0304",
    'acute': "\u0301",
    'circm': "\u0302",
    'caron': "\u030C",
    'tilde': "\u0303",
}
COMBINING_TO_NAME = {
    "\u0300": 'grave',
    "\u0304": 'macrn',
    "\u0301": 'acute',
    "\u0302": 'circm',
    "\u030C": 'caron',
    "\u0303": 'tilde',
}
TONE_LETTERS = {
    'grave': "L",
    'macrn': "M",
    'acute': "H",
    'circm': "HL",
    'caron': "LH",
}
COMPOSITE = {
    "a": {"acute": "á", "macrn": "ā", "grave": "à", "caron": "ǎ", "circm": "â", "tilde": "ã",},
    "e": {"acute": "é", "macrn": "ē", "grave": "è", "caron": "ě", "circm": "ê", "tilde": "ẽ",},
    "i": {"acute": "í", "macrn": "ī", "grave": "ì", "caron": "ǐ", "circm": "î", "tilde": "ĩ",},
    "o": {"acute": "ó", "macrn": "ō", "grave": "ò", "caron": "ǒ", "circm": "ô", "tilde": "õ",},
    "u": {"acute": "ú", "macrn": "ū", "grave": "ù", "caron": "ǔ", "circm": "û", "tilde": "ũ",},
}

# ------------------- #
# IPA transliteration #
# ------------------- #

def get_epitran(lang_tag, lang_key='fleurs', script: Optional[str]=None):
    """
    Instantiate and return an Epitran transliteration object
    for the given `fleurs_lang`.
    """
    with open('meta/language_codes.json') as f:
        lang_codes = json.load(f)
    if type(lang_tag) is list and len(lang_tag)==1:
        lang_tag=lang_tag[0]
    elif type(lang_tag) is list:
        return {tag: get_epitran(lang_key=lang_key, script=script) for tag in lang_tag}
    lang_dict = [d for d in lang_codes if d.get(lang_key, None)==lang_tag][0]
    iso3 = lang_dict['iso3']
    if not script:
        script=lang_dict['fleurs_script']
    
    return epitran.Epitran(f"{iso3}-{script}")

# --------------------------- #
# ASR post-processing methods #
# --------------------------- #

def get_remove_oov_char_funct(vocab_file: str) -> Callable[str, str]:
    """
    Load a text file containing every unique char in an ASR dataset.
    Returns a function that takes a str and outputs the same str with any characters
    not in the vocab file removed.
    """
    with open(vocab_file) as f:
        chars=f.read()
    return lambda s: ''.join(c for c in s if c in chars)

def condense_tones(s: str) -> str:
    """
    Return a string such that for each sequence of multiple tone chars in `s`,
    only the first is returned.
    """
    in_tone_seq = False
    out=''
    for c in s:
        if c in COMBINING_TO_NAME and not in_tone_seq:
            # first tone in sequence
            out+=c
            in_tone_seq=True
        elif c in COMBINING_TO_NAME:
            pass
        else:
            out+=c
            in_tone_seq=False

    return out

# ---------------- #
# Text LID methods #
# ---------------- #

def is_en_word(w: str) -> bool:
    return en_dict.check(w)

def has_unicode(s):
    return unidecode(s) != s

# ---------------------------- #
# String normalization helpers #
# ---------------------------- #

def unicode_normalize(
        text: str,
        unicode_format: Literal['NFC', 'NFKC', 'NFD', 'NFKD'] = 'NFKD',
    ) -> str:
    """
    wraps unicodedata.normalize with default format set to NFKD
    """
    return unicodedata.normalize(unicode_format, text)

def unicode_description(char: str):
    unicode_name = unicodedata.name(char, 'No unicode name found')
    unicode_point = str(hex(ord(char)))
    return {
        'unicode_name': unicode_name,
        'unicode_point': unicode_point,
    }

def get_char_metadata(texts: Sequence[str]) -> List[Dict[str, str]]:
    unique_chars = set()
    for t in texts:
        unique_chars.update(t)
    char_objs = []
    for c in unique_chars:
        char_obj = dict()
        char_obj['character'] = c
        char_obj.update(unicode_description(c))
        char_obj['replace'] = False
        char_objs.append(char_obj)
    return char_objs

def get_reps_from_chardata(chardata: List[Dict[str, str]]) -> Dict[str, str]:
    reps = {}
    for char_obj in chardata:
        intab = char_obj['character']
        outtab = char_obj['replace']
        if outtab is False:
            continue
        if not outtab:
            outtab = ''
        reps[intab] = outtab
    return reps

def max_ord_in_str(text: str) -> int:
    return max(ord(c) for c in text)

def make_replacements(text: str, reps: Dict[str, str]) -> str:
    """
    Makes all replacements specified by `reps`, a dict whose keys are intabs
    and values are outtabs to replace them.
    Avoids transitivity by first replacing intabs to a unique char not found in the original string.
    """
    max_ord = max_ord_in_str(text)
    intab2unique = {
        k: chr(max_ord+i+1) for i, k in enumerate(reps.keys())
    }
    unique2outtab = {
        intab2unique[k]: v for k, v in reps.items()
    }

    # sort intabs so that longest sequences come first
    intabs = sorted(reps.keys(), key=len, reverse=True)

    for intab in intabs:
        sentinel = intab2unique[intab]
        text = text.replace(intab, sentinel)
    for sentinel, outtab in unique2outtab.items():
        text = text.replace(sentinel, outtab)

    return text

def remove_punct(text: str) -> str:
    for p in punctuation:
        text = text.replace(p, '')
    return text

def report_unique_chars(texts: Sequence[str]) -> Dict[str, Any]:
    unique = set()
    (unique.update(text) for text in texts)
    # find some way to get Unicode metadata for each character

def strip_diacs(text: str, tone_only: bool = False) -> str:
    text = unicode_normalize(text)
    for diac in COMBINING.values():
        if tone_only and diac == COMBINING['tilde']:
            continue
        text = text.replace(diac, '')
    return text

def get_tone_as_letters(text: str) -> str:
    tone_words = []
    for word in text.split():
        tone_diacs = ''.join(c for c in word if c in COMBINING.values())
        tone_word = '-'.join(tone_diacs)
        for diac_name, letter in TONE_LETTERS.items():
            tone_word = tone_word.replace(COMBINING[diac_name], letter)
        tone_words.append(tone_word)

    return ' '.join(tone_words)

def split_segs_and_tone(text: str) -> Tuple[str, str]:
    return strip_diacs(text, tone_only=True), get_tone_as_letters(text)