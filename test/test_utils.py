import sys
sys.path.append('scripts')
from dataset_utils import SPECIAL_TOKENS_FLAT

def assert_chunk_dict_shape(chunk_dict, chunks_key='chunks'):
    assert type(chunk_dict) is dict
    assert chunks_key in chunk_dict
    assert type(chunk_dict[chunks_key]) is list
    for chunk in chunk_dict[chunks_key]:
        assert type(chunk) is dict
        assert 'timestamp' in chunk
        assert type(chunk['timestamp']) is tuple
        assert len(chunk['timestamp'])==2
        start = chunk['timestamp'][0]
        end = chunk['timestamp'][1]
        assert type(start) is float
        assert type(end) is float
        assert end>start

def assert_tokens_in_row(row, token_names, col='labels'):
    language_ids = [SPECIAL_TOKENS_FLAT[token]['id'] for token in token_names]
    labels = row[col]
    for lang_id in language_ids:
        assert lang_id in labels