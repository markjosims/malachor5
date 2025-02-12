import sys
sys.path.append('scripts')
from lid_utils import get_word_language

def test_get_word_language():
    assert get_word_language('"àpɾí"???!') == 'tira'
    assert get_word_language('&Hello;;;') == 'eng'
    assert get_word_language('!:;"ubeqinisile"??') == 'zulu'