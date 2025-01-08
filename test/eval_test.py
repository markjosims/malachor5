import sys
sys.path.append('scripts')
from eval import get_wer_by_language

def test_get_wer_by_language_1():
    ref = "yeah     àpɾí    jícə̀lò      is      right"
    #     ENG>TIC   H TIC   TIC>MISC    H ENG   H ENG
    hyp = "jâ       àpɾí    yichelow    is      right"

    result = get_wer_by_language(ref, hyp)[0]

    assert result['tira_insertions'] ==             0
    assert result['tira_deletions'] ==              0
    assert result['tira2eng_substitutions'] ==      0
    assert result['tira2misc_substitutions'] ==     1
    assert result['tira2tira_substitutions'] ==     0
    assert result['tira_hits'] ==                   1
    assert result['eng_insertions'] ==              0
    assert result['eng_deletions'] ==               0
    assert result['eng2tira_substitutions'] ==      1
    assert result['eng2misc_substitutions'] ==      0
    assert result['eng2eng_substitutions'] ==       0
    assert result['eng_hits'] ==                    2

def test_get_wer_by_language_2():
    ref = "àpɾí             means boy   àpɾí"
    #     TIC>TIC   I TIC   H ENG D ENG H TIC
    hyp = "àp       pɾí     means       àpɾí"

    result = get_wer_by_language(ref, hyp)[0]

    assert result['tira_insertions'] ==             1
    assert result['tira_deletions'] ==              0
    assert result['tira2eng_substitutions'] ==      0
    assert result['tira2misc_substitutions'] ==     0
    assert result['tira2tira_substitutions'] ==     1
    assert result['tira_hits'] ==                   0
    assert result['eng_insertions'] ==              0
    assert result['eng_deletions'] ==               1
    assert result['eng2tira_substitutions'] ==      0
    assert result['eng2misc_substitutions'] ==      0
    assert result['eng2eng_substitutions'] ==       0
    assert result['eng_hits'] ==                    1