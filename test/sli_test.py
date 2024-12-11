import sys
sys.path.append('scripts')
from sli import train_logreg, infer_lr, init_argparser
from model_utils import load_lr
from tokenization_utils import TIRA_DRZ
from datasets import Dataset
from dataset_utils import load_sli_dataset


# def test_hf_embeddings():
#     args = init_argparser().parse_args([])
#     args.embed_api='hf'
#     args.dataset=TIRA_DRZ
    
def test_load_sli_dataset():
    args = init_argparser().parse_args([])
    args.dataset = TIRA_DRZ
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

# def test_sb_embeddings():
#     ...

# def test_load_lr():
#     ...

# def test_train_logreg():
#     ...

# def test_infer_lr():
#     ...