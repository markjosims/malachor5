import pandas as pd
import numpy as np
from typing import Literal

"""
Unittests for functions used in `generate_long_labels.ipynb`
"""

def test_associate_machine_labels():
    # make a toy example to test `associate_machine_labels`
    toy_df = pd.DataFrame({
        'start':        [0,         10,         20,             30,         40,         50,             60,         70     ],
        'end':          [5,         15,         25,             35,         45,         55,             65,         75     ],
        'asr_index':    [np.nan,    np.nan,     1,              np.nan,     np.nan,     2,              np.nan,     np.nan ],
        'tier_name':    ['asr',     'asr',      'human_label',  'asr',      'asr',      'human_label',   'asr',      'asr' ],
    })

    def associate_machine_labels_toy(df: pd.DataFrame):
        df=df.sort_values('start')
        def map_asr_indices(row: pd.Series):
            start = row['start']
            end = row['end']
            
            add_direction: Literal['prev', 'next'] = 'prev'
            last_index_added = None
            # while there are still rows in the asr labels that have not been associated with a human label
            # and the time window is less than 30 seconds
            # associate last index added with human label
            while (end-start) < 30:
                if last_index_added is not None:
                    df.at[last_index_added, 'asr_index'] = row['asr_index']

                prev_rows = df[df['end']<=start]
                next_rows = df[df['start']>=end]
                if len(prev_rows)==0 or not np.isnan(prev_rows.iloc[-1]['asr_index']):
                    prev_rows=None
                if len(next_rows)==0 or not np.isnan(next_rows.iloc[0]['asr_index']):
                    next_rows=None
                if (prev_rows is None) and (next_rows is None):
                    break
                if next_rows is not None and (add_direction == 'next' or prev_rows is None):
                    end = next_rows.iloc[0]['end']
                    add_direction = 'prev'
                    last_index_added = next_rows.iloc[0].name
                else: # next_rows is None or add_direction == 'prev':
                    start = prev_rows.iloc[-1]['start']
                    add_direction = 'next'
                    last_index_added = prev_rows.iloc[-1].name
        df[df['tier_name']=='human_label'].apply(map_asr_indices, axis=1)
        return df
    toy_df=associate_machine_labels_toy(toy_df)
    assert np.array_equal(toy_df['asr_index'].values, np.array([np.nan, 1, 1, 1, 2, 2, 2, np.nan]), equal_nan=True)

    toy_df['start'] = toy_df['start']/10
    toy_df['end'] = toy_df['end']/10
    toy_df['asr_index']=[np.nan, np.nan, 1, np.nan, np.nan, 2, np.nan, np.nan]
    toy_df=associate_machine_labels_toy(toy_df)
    assert np.array_equal(toy_df['asr_index'].values, np.array([1, 1, 1, 1, 1, 2, 2, 2]), equal_nan=True)