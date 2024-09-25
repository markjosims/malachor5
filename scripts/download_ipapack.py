from huggingface_hub import snapshot_download
import os

DATASET_DIR = os.environ.get('DATASETS', '/usr/markjos/datasets')

def main():
    datasets = [
        "fleurs_ipa",
        "mswc_ipa",
        "doreco_ipa",
    ]
    user="anyspeech/"

    for dataset in datasets:
        snapshot_download(
            repo_id=user+dataset,
            repo_type="dataset",
            local_dir=os.path.join(DATASET_DIR, dataset),
            local_dir_use_symlinks=False,
            resume_download=False,
            max_workers=4
        )

if __name__ == '__main__':
    main()