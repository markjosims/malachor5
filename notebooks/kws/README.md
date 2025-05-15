# Keyword Search notebooks

## prepare_recordings_for_alignment.ipynb
Clean up data in `.eaf` files for validation and test data, generate Praat TextGrids for alignment with MFA, and create dictionaries and keyword lists for all Tira words in validation and test data.

## evaluate_kws_initial.ipynb
Using predicted keyword probabilities from CLAP-IPA stored in `data/tira_eval_kws`, evaluate correlation with ground truth timestamps for keywords using TextGrid alignments from MFA.
Much of code in this notebook was later encapsulated in `scripts/kws.py`.

## visualize_kws.ipynb
Visualize output of various runs of KWS.

## hmm_inference.ipynb
Decode keyword similarity matrices from KWS using HMMs.

## embed_distributions.ipynb
Sandboxing custom distributions with `pomegranate` for embedding similarity.