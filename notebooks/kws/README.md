# Keyword Search notebooks

## prepare_recordings_for_alignment.ipynb
Clean up data in `.eaf` files for validation and test data, generate Praat TextGrids for alignment with MFA, and create dictionaries and keyword lists for all Tira words in validation and test data.

## evaluate_kws.ipynb
Using predicted keyword probabilities from CLAP-IPA stored in `data/tira_eval_kws`, evaluate correlation with ground truth timestamps for keywords using TextGrid alignments from MFA.