
def assert_chunk_dict_shape(chunk_dict):
    assert type(chunk_dict) is dict
    assert 'chunks' in chunk_dict
    assert type(chunk_dict['chunks']) is list
    for chunk in chunk_dict['chunks']:
        assert type(chunk) is dict
        assert 'timestamp' in chunk
        assert type(chunk['timestamp']) is tuple
        assert len(chunk['timestamp'])==2
        start = chunk['timestamp'][0]
        end = chunk['timestamp'][1]
        assert type(start) is float
        assert type(end) is float
        assert end>start