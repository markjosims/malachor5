import sys
sys.path.append('scripts')
from tokenization_utils import normalize_multiling

def test_normalize_tira_eng_str():
    source = "Yeah, 'àn ápɾí jícə̀lò!' means 'is the boy good?' "
    norm = 'yeah àn ápɾí jícə̀lò means is the boy good'
    out = normalize_multiling(source)
    assert norm == out