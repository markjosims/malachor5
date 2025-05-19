from matlab import engine
import os

SNREVAL_DIR = os.environ.get('SNREVAL_DIR', "../snreval")

def start_engine():
    print("Loading matlab engine...")
    eng = engine.start_matlab()
    eng.addpath(SNREVAL_DIR)
    return eng

def map_snr(eng, audio, sampling_rate):
    """
    Calculate WADA SNR and NIST STNR for input audio array.
    Uses matlab code from Labrosa at https://labrosa.ee.columbia.edu/projects/snreval/
    Downloaded 9 Sep 2024
    """
    wada_snr = eng.wada_snr(audio, sampling_rate)
    nist_snr = eng.nist_stnr_m(audio, sampling_rate)
    return {'wada_snr': wada_snr, 'nist_snr': nist_snr}