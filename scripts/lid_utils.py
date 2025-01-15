import enchant
from unidecode import unidecode
import importlib.util
if importlib.util.find_spec('enchant') is not None:
    import enchant


# ---------------- #
# Text LID methods #
# ---------------- #

en_dict = enchant.request_dict('en_US')

def is_en_word(w: str) -> bool:
    return en_dict.check(w)

def has_unicode(s):
    return unidecode(s) != s

