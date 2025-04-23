# dataset_builders
Scripts for building HuggingFace ASR datasets from raw data files.
Each script should take the following args:
`--audio` `-a`      Directory of audio files to generate dataset from
`--list` `-l`       Text file with list of basenames audio files to include (if each audio file is a single record)
**OR**
`--timestamps` `-t` .csv file with columns audio_basename,start,end,(split) indicating timestamps for each record (if records are sliced from a longer audio file)
`--output` `-o`     Directory path to save PyArrow dataset to
`--version` `-v`    Dataset version

Each script performs a series of preprocessing functions, with each preprocessing step adding a metadata str describing what action was taken with quantitative metrics e.g. "350 records excluded."
These metadata strs will be aggregated into a `README.md` file in the dataset directory.
Scripts should also output `metadata.yaml` in the output directory containing metrics such as the number of records and duration in time by split, language and, if applicable, speaker.