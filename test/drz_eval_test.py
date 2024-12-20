from glob import glob
import os
from pyannote.core import *
from pympi import Elan
import pandas as pd
import numpy as np

import sys
sys.path.append('scripts')
from drz_eval import elan_to_pyannote, get_diarization_metrics, evaluate_diarization, init_parser
from longform import perform_vad, perform_sli, load_and_resample, pipeout_to_eaf
from tokenization_utils import TIRA_LONGFORM
from model_utils import LOGREG_PATH
SAMPLE_WAVPATH = 'test/data/sample_biling.wav'

def test_elan_to_pyannote_tiers():
    """
    Make pyannote objects out of all .eaf files in TIRA_LONGFORM dataset.
    Ensure that `elan_to_pyannote` returns a dict containing keys for each speaker
    each mapping to an `Annotation` object.
    """
    eaf_paths = glob(os.path.join(TIRA_LONGFORM, 'eaf', '*.eaf'))
    assert len(eaf_paths)==8
    tier_names = ['HIM', 'SHA', 'NIN', 'PET', 'MAR', 'combined', 'verbose']
    for eaf in eaf_paths:
        pyannote_dict = elan_to_pyannote(eaf)
        for tier in tier_names:
            assert tier in pyannote_dict
            assert type(pyannote_dict[tier]) is Annotation

def test_elan_to_pyannote_data():
    """
    Create a toy Elan object and test that the corresponding pyannote `Annotation`
    objects have the right data.
    """
    eaf = Elan.Eaf()
    eaf.add_tier('HIM')
    eaf.add_annotation('HIM', 0, 500, 'TIC')
    eaf.add_annotation('HIM', 1000, 1500, "yeah that's right àpɾí")
    eaf.add_annotation('HIM', 3000, 3500, 'ENG')
    eaf.add_tier('MAR')
    eaf.add_annotation('MAR', 600, 900, 'TIC')
    eaf.add_annotation('MAR', 1600, 2200, 'ENG')

    pyannote_dict = elan_to_pyannote(eaf)

    him_annotation = Annotation()
    him_annotation[Segment(0, 0.5)] = 'TIC'
    him_annotation[Segment(1, 1.5)] = 'ENG'
    him_annotation[Segment(3, 3.5)] = 'ENG'

    mar_annotation = Annotation()
    mar_annotation[Segment(0.6, 0.9)] = 'TIC'
    mar_annotation[Segment(1.6, 2.2)] = 'ENG'

    assert pyannote_dict['HIM'] == him_annotation
    assert pyannote_dict['MAR'] == mar_annotation

def test_diarization_metrics():
    ref = {
        'HIM': Annotation(),
        'MAR': Annotation(),
        'combined': Annotation(),
    }
    ref['HIM'][Segment(0, 1)] = 'TIC'
    ref['HIM'][Segment(3.5, 4.5)] = 'ENG'
    ref['MAR'][Segment(2, 2.5)] = 'ENG'

    ref['combined'][Segment(0, 1)] = 'TIC'
    ref['combined'][Segment(3.5, 4.5)] = 'ENG'
    ref['combined'][Segment(2, 2.5)] = 'ENG'

    hyp = Annotation()
    # missed detection of Tira from 0.5 to 1.0
    hyp[Segment(0, 0.5)] = 'TIC'
    # false alarm of English from 2.5 to 3.0
    hyp[Segment(2, 3)] = 'ENG'
    # confusion of English for Tira from 3.5 to 4.5
    hyp[Segment(3.5, 4.5)] = 'TIC'

    metrics = get_diarization_metrics(ref, hyp, return_pct=False)

    assert metrics['combined'] == {
        'total': 2.5,                   # 2.5 seconds of speech overall
        'missed detection': 0.5,        # 0.5 seconds of Tira missed
        'false alarm': 0.5,             # 0.5 seconds of English falsely detected
        'confusion': 1.0,               # 1.0 seconds of confusion between English and Tira
        'correct': 1.0,                 # 1.0 seconds of correct detection
        'identification error rate': 0.8,  # (0.5 + 0.5 + 1.0) / 2.5 = 0.8
        'tira false alarm': 0.0,        # no false alarms for Tira
        'tira missed detection': 0.5,   # 0.5 seconds of Tira missed
        'tira confusion': 0.0,          # no Tira-English confusion
        'tira correct': 0.5,            # 0.5 seconds of correct Tira detection
        'tira total': 1.0,              # 1.0 second of Tira speech
        'eng false alarm': 0.5,         # 0.5 seconds of English falsely detected
        'eng missed detection': 0.0,    # no missed detection for English
        'eng confusion': 1.0,           # 1.0 seconds of English-Tira confusion
        'eng correct': 0.5,             # 0.5 second of correct English detection
        'eng total': 1.5,               # 1.5 seconds of English speech
    }
    assert metrics['MAR'] == {
        'total': 0.5,                   # 0.5 seconds of speech for MAR
        'missed detection': 0.0,        # no missed detections
        'confusion': 0.0,               # no language confusion
        'correct': 0.5,                 # 0.5 seconds of correct detection
        'tira missed detection': 0.0,   # no Tira missed
        'tira confusion': 0.0,          # no Tira-English confusion
        'tira correct': 0.0,            # no correct Tira detection
        'tira total': 0.0,              # no Tira speech
        'eng missed detection': 0.0,    # no missed detection for English
        'eng confusion': 0.0,           # no English-Tira confusion
        'eng correct': 0.5,             # 0.5 seconds of correct English detection
        'eng total': 0.5,               # 0.5 seconds of English speech
    }
    assert metrics['HIM'] == {
        'total': 2.0,                   # 2.0 seconds of speech for HIM
        'missed detection': 0.5,        # 0.5 seconds of missed detection for HIM
        'confusion': 1.0,               # 1.0 second language confusion
        'correct': 0.5,                 # 0.5 seconds of correct detection
        'tira missed detection': 0.5,   # 0.5 seconds of missed detection for Tira
        'tira confusion': 0.0,          # no Tira-English confusion
        'tira correct': 0.5,            # no correct Tira detection
        'tira total': 1.0,              # 1.0 second of Tira speech
        'eng missed detection': 0.0,    # no missed detection for English
        'eng confusion': 1.0,           # 1.0 second of English-Tira confusion
        'eng correct': 0.0,             # no correct English detection
        'eng total': 1.0,               # 1.0 second of English speech
    }

    metrics_df = get_diarization_metrics(ref, hyp, return_df=True, return_pct=False)
    assert metrics_df.shape == (3, 16)
    comp_df = pd.DataFrame({
        'total': [2.5, 0.5, 2.0],
        'missed detection': [0.5, 0.0, 0.5],
        'false alarm': [0.5, np.nan, np.nan],
        'confusion': [1.0, 0.0, 1.0],
        'correct': [1.0, 0.5, 0.5],
        'identification error rate': [0.8, np.nan, np.nan],
        'eng false alarm': [0.5, np.nan, np.nan],
        'eng missed detection': [0.0, 0.0, 0.0],
        'eng confusion': [1.0, 0.0, 1.0],
        'eng correct': [0.5, 0.5, 0.0],
        'eng total': [1.5, 0.5, 1.0],
        'tira false alarm': [0.0, np.nan, np.nan],
        'tira missed detection': [0.5, 0.0, 0.5],
        'tira confusion': [0.0, 0.0, 0.0],
        'tira correct': [0.5, 0.0, 0.5],
        'tira total': [1.0, 0.0, 1.0],
    }, index=['combined', 'MAR', 'HIM'])
    metrics_df=metrics_df.sort_index(axis=0).sort_index(axis=1)
    comp_df=comp_df.sort_index(axis=0).sort_index(axis=1)
    for col in comp_df.columns:
        assert pd.Series.equals(metrics_df[col], comp_df[col])

def test_perform_sli_and_evaluate():
    """
    Not checking any outputs, just making sure that the pipeline runs without errors.
    """
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav, return_wav_slices=True)
    assert type(vad_out) is dict
    vad_chunks = vad_out['vad_chunks']
    sli_out, _ = perform_sli(vad_chunks, lr_model=LOGREG_PATH)
    assert type(sli_out) is list
    eaf = pipeout_to_eaf(sli_out, chunk_key='sli_pred', tier_name='sli')
    assert type(eaf) is Elan.Eaf
    pyannote_dict = elan_to_pyannote(eaf, tgt_tiers=['sli'])
    assert type(pyannote_dict) is dict
    metrics = get_diarization_metrics(pyannote_dict, pyannote_dict['combined'], return_df=True)
    assert type(metrics) is pd.DataFrame

def test_perform_sli_directory(tmpdir):
    parser = init_parser()
    args = parser.parse_args([])
    args.wav = os.path.join(TIRA_LONGFORM, 'wav')
    args.ref = os.path.join(TIRA_LONGFORM, 'eaf')
    args.logreg = LOGREG_PATH
    args.output = os.path.join(tmpdir, 'output.csv')
    result = evaluate_diarization(args)
    assert result == 0
    assert os.path.exists(args.output)
    df = pd.read_csv(args.output)
    assert type(df) is pd.DataFrame
    assert df.shape == (8, 17)
    assert 'average' in df['file'].values

def test_diarization_metrics_vad():
    wav = load_and_resample(SAMPLE_WAVPATH)
    vad_out = perform_vad(wav, return_wav_slices=True)
    vad_eaf = pipeout_to_eaf(vad_out['vad_chunks'], label='vad', tier_name='vad')
    sli_out, _ = perform_sli(vad_out['vad_chunks'], lr_model=LOGREG_PATH)
    sli_eaf = pipeout_to_eaf(sli_out, chunk_key='sli_pred', tier_name='HIM')
    vad_metrics = get_diarization_metrics(ref=sli_eaf, hyp=vad_eaf, task='vad')
    assert type(vad_metrics) is dict
    keys = [
        'total', 'missed detection', 'false alarm', 'correct', 'diarization error rate',
        'tira missed detection', 'eng missed detection',
    ]
    for key in keys:
        assert key in vad_metrics['combined']
        assert type(vad_metrics['combined'][key]) in [float, int]

    if key not in ('false alarm', 'diarization error rate'):
        assert key in vad_metrics['HIM']
        assert type(vad_metrics['HIM'][key]) in [float, int]