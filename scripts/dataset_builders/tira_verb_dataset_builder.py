from typing import *
from tira_asr_dataset_builder import perform_textnorm
import os
import pandas as pd
from string import Template

import sys
sys.path.append('scripts')

AUDIO_DIR = os.environ.get("TIRA_ELICITATION_WAVS")
TIRA_VERB_DIR = os.environ.get("TIRA_VERBS")
VERSION = "0.1.0"
EXCEL_PATH = "meta/tira_verbs_raw.xlsx"

README_HEADER = Template(
"""
# tira_morph
Dataset of analyzed inflectional paradigms of Tira verbs.
Uses same textnorm steps as `tira_asr`. Contains $num_verbs
verb roots with $num_forms total unique inflected forms,
averaging $mean_form_ct forms per verb across $num_features
inflectional values.
"""
)
PREPROCESSING_STEPS = []

def main() -> int:
    df = pd.read_excel(EXCEL_PATH)
    print(len(df))
    

    PREPROCESSING_STEPS.append("## Drop rows not in TAMD dataset and melt dataframe")
    tamd_mask = df['Part of TAMD dataset?'].isin(["yes", "yes (extended)", "yes (has extension suffix)"])
    total_rows = len(df)
    df = df[tamd_mask]
    tamd_rows = len(df)
    tamd_str = f"- Dropped {total_rows-tamd_rows} verbs not in completed TAMD dataset, {tamd_rows} remaining verbs."
    PREPROCESSING_STEPS.append(tamd_str)
    print(tamd_str)

    feature_cols = [
       'IMP.AND',
       'IMP.VEN',
       'PFV.VEN',
       'PFV.AND',
       'IPFV.AND',
       'IPFV.VEN',
       'PRS.PROG.AND',
       'PRS.PROG.VEN',
       'Infinitive (NOM)',
       'Infinitive (ACC)',
       'PROH.AND',
       'PROH.VEN',
       'Dependent AND',
       'Dependent VEN',
       'Hort.ITV',
       'Hort.VEN',
    ]
    df = df.melt(id_vars=["Gloss", "Stem"], value_vars=feature_cols, value_name="form", var_name="feature")
    breakpoint()
    
    sentence_df, _ = perform_textnorm(df, PREPROCESSING_STEPS)

    readme_header_str = README_HEADER.substitute(

    )
    readme_out = os.path.join(TIRA_VERB_DIR, 'README.md')
    os.makedirs(TIRA_VERB_DIR, exist_ok=True)
    with open(readme_out, 'w', encoding='utf8') as f:
        f.write(readme_header_str+'\n')
        f.write('\n'.join(PREPROCESSING_STEPS))

    morph_out = os.path.join(TIRA_VERB_DIR, 'tira-segmentations.txt')
    with open(morph_out, mode='w', encoding='utf8') as f:
        for analysis in all_words_analyzed:
            morphs = analysis.split('-')
            f.write(f"{''.join(morphs)} {' '.join(morphs)}\n")
if __name__ == '__main__':
    main()