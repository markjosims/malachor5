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
    tok_ids = [SPECIAL_TOKENS_FLAT[token]['id'] for token in token_names]
    labels = row[col]
    for tok_id in tok_ids:
        assert tok_id in labels

def assert_tokens_not_in_row(row, token_names, col='labels'):
    tok_ids = [SPECIAL_TOKENS_FLAT[token]['id'] for token in token_names]
    labels = row[col]
    for tok_id in tok_ids:
        assert tok_id not in labels

def assert_labels_begin_with(row, token_names, col='labels'):
    tok_ids = [SPECIAL_TOKENS_FLAT[token]['id'] for token in token_names]
    labels = row[col]
    assert labels[:len(tok_ids)]==tok_ids, labels

def assert_labels_end_with(row, token_names, col='labels'):
    tok_ids = [SPECIAL_TOKENS_FLAT[token]['id'] for token in token_names]
    labels = row[col]
    assert labels[-len(tok_ids):]==tok_ids, labels

def assert_tokens_appear_once(row, token_names, col='labels'):
    tok_ids = [SPECIAL_TOKENS_FLAT[token]['id'] for token in token_names]
    labels = row[col]
    for tok_id in tok_ids:
            assert labels.count(tok_id)==1, tok_id