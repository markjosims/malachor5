from test_utils import SAMPLE_BILING_PATH, NYEN_PATH, XDDERE_PATH, ALBRRIZO_PATH

import os
import pandas as pd
from datasets import load_from_disk, DatasetDict, Dataset
import sys
sys.path.append('scripts')
sys.path.append('scripts/dataset_builders')

from dataset_builder_utils import load_clips_to_ds
from longform import load_and_resample, SAMPLE_RATE

def test_load_clips_to_ds(tmpdir):
    rows = []
    for wav_path in [SAMPLE_BILING_PATH, NYEN_PATH, XDDERE_PATH, ALBRRIZO_PATH]:
        wav = load_and_resample(wav_path)
        num_samples = wav.shape[-1]
        num_s = num_samples/SAMPLE_RATE
        num_ms = int(num_s*1_000)
        annotation_len = min(num_ms//10, 100)
        for i in range(0, num_ms, num_ms//10):
            rows.append({
                "audio_basename": os.path.basename(wav_path),
                "start": i,
                "end": i+annotation_len,
            })
    df = pd.DataFrame(rows)
    audio_dir = os.path.dirname(SAMPLE_BILING_PATH)
    ds_path = os.path.join(tmpdir, "dataset")
    ds = load_clips_to_ds(df, audio_dir, ds_path)
    assert os.path.isdir(ds_path)
    ds = load_from_disk(ds_path)

    assert type(ds) is DatasetDict
    ds = ds['train']
    assert type(ds) is Dataset

    assert 'index' in ds.column_names
    assert (ds['index']==df.index).all()

    assert 'clip_name' in ds.column_names
    clip_idcs = [int(clip_name.split('_')[-1].removesuffix('.wav')) for clip_name in ds['clip_name']]
    assert (clip_idcs == df.index).all()
    df_basenames = df['audio_basename'].str.removesuffix('.wav')
    for df_basename, ds_basename in zip(df_basenames, ds['clip_name']):
        assert df_basename in ds_basename