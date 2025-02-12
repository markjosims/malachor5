import sys
sys.path.append('scripts')
from tokenization_utils import normalize_eng_words_only

def test_normalize_tira_eng_str():
    source = "Yeah, 'àn ápɾí jícə̀lò!' means 'is the boy good?' "
    norm = 'yeah àn ápɾí jícə̀lò means is the boy good'
    out = normalize_eng_words_only(source)
    assert norm == out