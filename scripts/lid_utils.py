from unidecode import unidecode
import string
import nltk
from nltk.corpus import cmudict
from string_norm import has_tira_chars, remove_punct

"""
Helper methods for determining language identity of text.
For English, use pyenchant if available, if not use nltk.words.
For Tira, check if word is in list of Tira words (stored in `meta` folder)
or if word has non-ascii characters AND only has characters used for transcribing Tira.
For Zulu, check if word is in list of Zulu words (also stored in `meta`).
"""

try:
    CMU_DICT = cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    CMU_DICT = cmudict.dict()
CMU_WORDS = set(strip_punct(w.lower()) for w in CMU_DICT.keys())

tira_words_path = 'meta/tira_words.txt'
with open(tira_words_path, encoding='utf8') as f:
    TIRA_WORDS = [word.strip() for word in f.readlines()]
zulu_words_path = 'meta/zulu_words.txt'
with open(zulu_words_path, encoding='utf8') as f:
    ZULU_WORDS = [word.strip() for word in f.readlines()]


# ---------------- #
# Text LID methods #
# ---------------- #


def strip_punct(f):
    def g(s):
        return f(s.strip(string.punctuation))
    return g

@strip_punct
def is_en_word(w: str) -> bool:
    return w.lower() in CMU_WORDS

@strip_punct
def has_unicode(s):
    return unidecode(s) != s

@strip_punct
def is_tira_word(w: str) -> bool:
    return w in TIRA_WORDS or (has_unicode(w) and has_tira_chars(w))

@strip_punct
def is_zulu_word(w: str) -> bool:
    return w in ZULU_WORDS

def get_word_language(word: str, langs=None) -> str:
    if langs is None:
        langs = ['misc', 'tira', 'eng', 'zulu']
    word=remove_punct(word).strip()
    lang='misc'
    if len(word)<=1 and not word.isalpha():
        pass
    elif ('tira' in langs) and is_tira_word(word):
        lang='tira'
    elif ('zulu' in langs) and is_zulu_word(word):
        lang='zulu'
    elif ('eng') in langs and is_en_word(word):
        lang='eng'
    return lang

