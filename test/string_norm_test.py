import sys
sys.path.append('scripts')
from string_norm import *

def test_tira2arpabet():
    tira_str = 'ápɾí jícə̀lò jɛ̀ jìt̪ɔ̀t̪ɔ́ ŋávɛ̀ ðɛ̀ ðàŋàl nə̀ lə̀vɛ́r. ŋòɽón ŋáŋít̪ɔ̀.'
    arpabet_str = 'AA P DX IY Y IY CH AX L OW Y EH Y IY T AO T AO NG AA V EH DH EH DH AA NG AA L N AX L AX V EH R NG OW L OW N NG AA NG IY T AO'
    out_str = tira2arpabet(tira_str)
    assert out_str==arpabet_str