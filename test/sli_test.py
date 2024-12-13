import sys
sys.path.append('scripts')
import os
from sli import train_logreg, infer_lr, init_argparser, load_embeddings
from model_utils import load_lr, LOGREG_PATH
from tokenization_utils import TIRA_SLI, SB_VOXLINGUA
from datasets import Dataset
from dataset_utils import load_sli_dataset, load_dataset_safe
import torch
import numbers
import pickle
from sklearn.linear_model import LogisticRegression

def test_load_embeddings():
    """
    `load_embeddings` should return a torch tensor with embeddings
    for each row in the dataset
    """
    args = init_argparser().parse_args([])
    args.dataset = TIRA_SLI
    args.num_records = 3
    args.split='train'
    args.embed_api = 'sb'
    sb_embeddings = load_embeddings(args)
    assert type(sb_embeddings) is torch.Tensor
    assert sb_embeddings.shape[0] == 3
    assert sb_embeddings.shape[-1] == 256
    
def test_load_sli_dataset():
    """
    Should load a HuggingFace dataset as well as a json object
    for mapping SLI labels and ids
    """
    args = init_argparser().parse_args([])
    args.dataset = TIRA_SLI
    args.num_records = 1
    args.split='train'
    dataset, args = load_sli_dataset(args)
    assert type(dataset) is Dataset
    assert len(dataset)==1
    assert args.sli_map == [
        {"language": "Tira", "label": "TIC", "id": 0},
        {"language": "English", "label": "ENG", "id": 1}
    ]
    assert args.sli_id2label == {
        0: 'TIC',
        1: 'ENG',
    }
    assert args.sli_label2id == {
        'TIC': 0,
        'ENG': 1,
    }

def test_load_lr():
    """
    `load_lr` should return a LogisticRegression object and a Namespace
    with the attrs `sli_map`, `sli_id2label` and `sli_label2id`
    """
    lr1, args1 = load_lr(lr_model=LOGREG_PATH)
    args2 = init_argparser().parse_args([])
    args2.lr_model=LOGREG_PATH
    lr2, args2 = load_lr(lr_model=LOGREG_PATH)

    for lr, args in [(lr1, args1), (lr2, args2)]:
        assert type(lr) is LogisticRegression
        assert hasattr(args, 'sli_map') and args.sli_map == [
            {"language": "Tira", "label": "TIC", "id": 0},
            {"language": "English", "label": "ENG", "id": 1}
        ]
        assert hasattr(args, 'sli_id2label') and args.sli_id2label == {
            0: 'TIC',
            1: 'ENG',
        }
        assert hasattr(args, 'sli_label2id') and args.sli_label2id == {
            'TIC': 0,
            'ENG': 1,
        }
        assert hasattr(args, 'embed_model') and args.embed_model == SB_VOXLINGUA
        assert hasattr(args, 'embed_api') and args.embed_api == 'sb'

def test_train_logreg(tmp_path):
    """
    `train_logreg` should save a dictionary with the LogReg object,
    the model and API used for generating embeddings,
    and the classification accuracy obtained on the test set.
    """
    args = init_argparser().parse_args([])
    args.dataset = TIRA_SLI
    args.num_records = 20
    args.embed_api = 'sb'
    args.output=tmp_path/'logreg.pkl'
    train_logreg(args)

    with open(args.output, 'rb') as fh:
        lr_dict = pickle.load(fh)
    assert 'sli_map' in lr_dict and lr_dict['sli_map'] == [
        {"language": "Tira", "label": "TIC", "id": 0},
        {"language": "English", "label": "ENG", "id": 1}
    ]
    assert 'sli_id2label' in lr_dict and lr_dict['sli_id2label'] == {
        0: 'TIC',
        1: 'ENG',
    }
    assert 'sli_label2id' in lr_dict and lr_dict['sli_label2id'] == {
        'TIC': 0,
        'ENG': 1,
    }
    assert 'lr_model' in lr_dict and type(lr_dict['lr_model']) is LogisticRegression
    assert 'embed_model' in lr_dict and lr_dict['embed_model'] == SB_VOXLINGUA
    assert 'embed_api' in lr_dict and lr_dict['embed_api'] == 'sb'
    assert 'scores' in lr_dict and isinstance(lr_dict['scores'], numbers.Number)

def test_infer_lr():
    """
    `infer_lr` should accept an audio dataset and return the dataset
    with an additional column named `sli_pred`
    """
    args = init_argparser().parse_args([])
    args.dataset = TIRA_SLI
    args.num_records = 20
    args.embed_api = 'sb'
    args.lr_model=LOGREG_PATH
    ds_out, args = infer_lr(args)
    for split in ds_out.values():
        assert 'sli_preds' in split.column_names
        for pred in split['sli_preds']:
            assert pred in ['TIC', 'ENG']