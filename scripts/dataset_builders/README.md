# dataset_builders
Scripts for building HuggingFace ASR datasets from raw data files.
Each script loads a .csv file from the `meta` folder with a list of records, with columns `audio`, `text`, (`start`, `end`), (`split`)
    - `start` and `end` only included when slicing records from longer audio, else record is entire audio
    - `split` not included if splits are made during dataset creation
Dataset lists are stored in `meta` to ensure proper versioning through Git, since datasets used are small enough to store transcriptions and other metadata on Github.

Directory paths to load audio from and save dataset to should be provided through environment variables, loaded in as constants at the top of the script.
Each script loads three directories:
- AUDIO_DIR                 directory audio source files are stored
- ${DATASET}_CLIPS_DIR      directory to store clips and metadata in
                            (only applicable when records are sliced from long audio)
- ${DATASET_}_PYARROW_DIR   directory to store pyarrow dataset in

Each script performs a series of preprocessing functions, with each preprocessing step adding a metadata str describing what action was taken with quantitative metrics e.g. "350 records excluded."
These metadata strs will be aggregated into a `README.md` file in the dataset directory.
Scripts should also output `metadata.yaml` in the output directory containing metrics such as the number of records and duration in time by split, language and, if applicable, speaker.

# dataset builder scripts

## audio_segment_ds_builder.py
Given an input directory pointing to an audio dataset, creates a new dataset where each row is an audio frame
sliced from a record in the original dataset.

## ground_truth_prompts.py
Given input TextGrids, divides each TextGrid into chunks and creates a prompt for each chunk using Tira words present.
Saves a `.json` file with the resulting timestamps and prompt sentences.

## tira_asr_dataset_builder.py
Loads list stored at `meta/tira_elan_raw.csv` and generates an ASR dataset of monolingual Tira using (relatively) clean IPA transcriptions.

## tira_elan_scraper.py
Reads .eaf files from `TIRA_RECORDINGS_GDRIVE` and makes a list stored in `meta/tira_elan_raw.csv` with file basenames, timestamps, Elan tier names and values.

## tira_eval_transcript.py
Reads .eaf files stored in `meta/` folder and generates `.txt` files with transcriptions.
For each reacording outputs `$RECORDING.txt` and `$RECORDING-no-overlap.txt`, where all overlapped segments are removed using following criterion:
- If two segments overlap, delete shorter segment IF overlap portion >=50% of shorter segment duration.

## tira_morph_dataset_builder.py
Loads list stored at `meta/tira_elan_raw.csv` and generates a text dataset for morphological analysis using IPA transcriptions
with same preprocessing steps used by `tira_asr_dataset_builder.py`.

## tira_verb_dataset_builder.py
Loads Excel spreadsheet stored at `meta/tira_verbs_raw.xlsx` and creates a `.csv` file with