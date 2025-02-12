import enchant
from unidecode import unidecode
import string
import importlib.util

from string_norm import has_tira_chars, remove_punct
if importlib.util.find_spec('enchant') is not None:
    import enchant

tira_words_path = 'meta/tira_words.txt'
with open(tira_words_path) as f:
    TIRA_WORDS = [word.strip() for word in f.readlines()]
zulu_words_path = 'meta/zulu_words.txt'
with open(zulu_words_path) as f:
    ZULU_WORDS = [word.strip() for word in f.readlines()]


# ---------------- #
# Text LID methods #
# ---------------- #

en_dict = enchant.request_dict('en_US')

def strip_punct(f):
    def g(s):
        return f(s.strip(string.punctuation))
    return g

@strip_punct
def is_en_word(w: str) -> bool:
    return en_dict.check(w)

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
    word=remove_punct(word).strip()
    lang='misc'
    if len(word)<=1 and not word.isalpha():
        pass
    elif is_tira_word(word):
        lang='tira'
    elif is_zulu_word(word):
        lang='zulu'
    elif is_en_word(word):
        lang='eng'
    if langs and (lang not in langs):
        return 'misc'
    return lang

