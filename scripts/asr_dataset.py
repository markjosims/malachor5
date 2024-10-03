from glob import glob
from pympi import Elan
from typing import Optional, Sequence, List, Tuple, Dict, Union
from argparse import ArgumentParser, Namespace
import pandas as pd
import os
import shutil
import librosa
import numpy as np
import soundfile
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Audio, Dataset, DatasetDict, IterableDataset
from transformers import AutomaticSpeechRecognitionPipeline, AutoProcessor, DebertaV2Tokenizer, AutoTokenizer
import torch
from tempfile import TemporaryDirectory
import csv
from unidecode import unidecode
import json
from train_whisper import load_whisper_pipeline
# TODO: move heavy imports (torch, transformers, datasets) into methods

GDRIVE_DIR = '/Users/markjos/Library/CloudStorage/GoogleDrive-mjsimmons@ucsd.edu/Shared drives/Tira/Recordings'
SNREVAL_DIR = '/Users/markjos/projects/snreval'
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
device_type = lambda s: int(s) if s!='cpu' else s
tqdm.pandas()

def init_parser() -> ArgumentParser:
    # TODO: some command arguments are very wet. Make them DRY.
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', default=GDRIVE_DIR)
    parser.add_argument('--output', '-o')
    parser.add_argument('--logfile', '-l', default='error.txt')
    parser.add_argument('--split', '-s')
    parser.add_argument('--fleurs_lang', default='all')
    parser.add_argument('--num_records', '-n', type=int)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--keep_cols', action='store_true', help='If true, only remove `audio` col from original dataset.')
    parser.set_defaults(func=empty_command)

    commands=parser.add_subparsers(help='Command to run')

    pool_eaf_data_parser=commands.add_parser(
        'pool_eaf_data',
        help="Glob all `.eaf` files in input directory recursively and create `.csv` file "+\
        "containing rows for each interval with columns `start`, `end`, `transcription`, "+\
        "`eaf_source` and `wav_source.` --output may be path to `.csv` file or directory to "+\
        "create `metadata.csv` in."
    )
    pool_eaf_data_parser.add_argument('--transcription_tier', '-t', default='IPA Transcription')
    pool_eaf_data_parser.set_defaults(func=pool_eaf_data)

    make_clips_parser=commands.add_parser(
        'make_clips',
        help=make_clips.__doc__
    )
    make_clips_parser.add_argument('--check_clips_exist', action='store_true')
    make_clips_parser.set_defaults(func=make_clips)

    validate_clips_parser=commands.add_parser(
        'validate_clips',
        help=validate_clips.__doc__
    )
    validate_clips_parser.add_argument('--recursive', '-r', action='store_true')
    validate_clips_parser.add_argument('--is_dataset', action='store_true')
    validate_clips_parser.set_defaults(func=validate_clips)

    hf_dataset_parser=commands.add_parser(
        'make_hf_dataset',
        help=make_hf_dataset.__doc__
    )
    hf_dataset_parser.add_argument('--make_splits', action='store_true')
    hf_dataset_parser.set_defaults(func=make_hf_dataset)

    remove_clips_parser=commands.add_parser(
        'remove_extra_clips',
        help=remove_extra_clips.__doc__
    )
    remove_clips_parser.set_defaults(func=remove_extra_clips)

    infer_asr_parser = commands.add_parser('infer_asr', help=infer_asr.__doc__)
    infer_asr_parser.add_argument('--model', '-m', default='openai/whisper-large-v3')
    infer_asr_parser.add_argument('--device', '-D', type=device_type, default=DEVICE,)
    infer_asr_parser.add_argument('--batch_size', '-b', type=int, default=32)
    infer_asr_parser.add_argument('--language', '-l', nargs='+')
    infer_asr_parser.set_defaults(func=infer_asr)

    infer_vad_parser = commands.add_parser('infer_vad', help=infer_vad.__doc__)
    infer_vad_parser.add_argument('--model', '-m', default='pyannote/voice-activity-detection')
    infer_vad_parser.add_argument('--device', '-D', type=device_type, default=DEVICE,)
    infer_vad_parser.set_defaults(func=infer_vad)

    infer_allosaurus_parser = commands.add_parser('infer_allosaurus', help=infer_allosaurus.__doc__)
    infer_allosaurus_parser.add_argument('--model', '-m', default='uni2005')
    infer_allosaurus_parser.add_argument('--batch_size', '-b', default=32)
    infer_allosaurus_parser.add_argument('--lang', '-l', default='tic')
    infer_allosaurus_parser.add_argument(
        '--device',
        '-D',
        type=lambda s: -1 if s=='cpu' else int(s),
        # allosaurus uses -1 for 'cpu'
        default=DEVICE,
    )
    infer_allosaurus_parser.set_defaults(func=infer_allosaurus)

    clap_ipa_sim_parser = commands.add_parser('clap_ipa_sim', help=clap_ipa_sim.__doc__)
    clap_ipa_sim_parser.add_argument(
        '--device',
        '-D',
        type=device_type,
        default=DEVICE
    )
    clap_ipa_sim_parser.add_argument(
        '--batch_size', '-b', type=int, default=32,
    )
    clap_ipa_sim_parser.add_argument(
        '--model_size',
        '-m',
        choices=['tiny', 'base', 'small'],
        default='small'
    )
    clap_ipa_sim_parser.add_argument('--g2p', action='store_true')
    clap_ipa_sim_parser.add_argument('--script')
    clap_ipa_sim_parser = add_string_norm_args(clap_ipa_sim_parser)
    clap_ipa_sim_parser.set_defaults(func=clap_ipa_sim)

    clap_ipa_text_sim_parser = commands.add_parser('clap_ipa_text_sim', help=clap_ipa_sim.__doc__)
    clap_ipa_text_sim_parser.add_argument(
        '--device',
        '-D',
        type=device_type,
        default=DEVICE
    )
    clap_ipa_text_sim_parser.add_argument(
        '--batch_size', '-b', type=int, default=32,
    )
    clap_ipa_text_sim_parser.add_argument(
        '--model_size',
        '-m',
        choices=['tiny', 'base', 'small'],
        default='small'
    )
    clap_ipa_text_sim_parser.add_argument('--col1')
    clap_ipa_text_sim_parser.add_argument('--col2')
    clap_ipa_text_sim_parser = add_string_norm_args(clap_ipa_text_sim_parser)
    clap_ipa_text_sim_parser.set_defaults(func=clap_ipa_text_sim)


    detect_clipping_parser = commands.add_parser('detect_clipping', help=detect_clipping.__doc__)
    detect_clipping_parser.set_defaults(func=detect_clipping)

    snr_parser = commands.add_parser('snr', help=calculate_snr.__doc__)
    snr_parser.set_defaults(func=calculate_snr)

    split_ds_parser = commands.add_parser('split_dataset', help=split_dataset.__doc__)
    split_ds_parser.add_argument('--dataset', '-d')
    split_ds_parser.add_argument('--splitsize', nargs=3, default=[0.8, 0.1, 0.1])
    split_ds_parser.set_defaults(func=split_dataset)

    return parser

# -------------- #
# Parser helpers #
# -------------- #

def add_string_norm_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--no_tone', action='store_true')
    parser.add_argument('--ascii_only', action='store_true')
    parser.add_argument('--no_space', action='store_true')
    return parser

# ------------------- #
# ELAN helper methods #
# ------------------- #

def get_media_path(eaf):
    media_paths = [x['MEDIA_URL'] for x in eaf.media_descriptors]
    media = media_paths[0]
    # trim prefix added by ELAN
    # have to keep initial / on posix systems
    # and remove on Windows
    if os.name == 'nt':
        media = media.replace('file:///', '')
    else:
        media = media.replace('file://', '')
    # computers why must you be so silly
    return media

def samples_to_ms(samples, sampling_rate=16_000) -> int:
    seconds = samples/sampling_rate
    ms=int(seconds*1_000)
    return ms

def ms_to_samples(ms, sampling_rate=16_000) -> int:
    seconds = ms/1_000
    samples=int(seconds*sampling_rate)
    return samples

def ms_to_humantime(ms: int) -> str:
    timestr = ''
    hours = ms//3_600_000
    if hours:
        timestr=f'h{hours:02}'
    minutes = (ms%3_600_000)//60_000
    timestr=timestr+f'm{minutes:02}'
    seconds = (ms%60_000)//1_000
    ms_remain = ms%1000
    timestr=timestr+f's{seconds:02}ms{ms_remain:03}'
    return timestr

def clip_segment(
        start_ms: int,
        end_ms: int,
        wav_basename: str,
        target_dir: str,
        wav: Optional[np.ndarray]=None,
        sampling_rate: int=16_000,
    ) -> str:
    """
    Generate filepath from `wav_basename` corresponding to the start
    and end timestamps specified by `start_ms` and `end_ms`.
    If `wav` argument is provided, clip to respective timestamps
    and save to filepath. Either way, return filepath as str.
    """
    start_timestr=ms_to_humantime(start_ms)
    end_timestr=ms_to_humantime(end_ms)

    time_suffix = f"-{start_timestr}-{end_timestr}.wav"
    clip_basename = wav_basename.removesuffix('.wav')+time_suffix
    clip_fp = os.path.join(
        target_dir,
        clip_basename
    )

    if wav is not None:
        start_samples=ms_to_samples(start_ms, sampling_rate=sampling_rate)
        end_samples=ms_to_samples(end_ms, sampling_rate=sampling_rate)
        wav_clip = wav[start_samples:end_samples]
        soundfile.write(clip_fp, wav_clip, samplerate=sampling_rate)

    return clip_fp

def check_clips_exist(is_source: pd.Series, wav_source: str, clip_dir: str) -> bool:
    num_source = is_source.value_counts()[True]
    source_basename = os.path.basename(wav_source)
    glob_basename = source_basename.removesuffix('.wav')+'*.wav'
    glob_str = os.path.join(clip_dir, glob_basename)
    clip_paths = glob(glob_str)
    return len(clip_paths) == num_source

def move_clip_to_split(row, args) -> str:
    clip_relpath=row['file_name']
    clip_path=os.path.join(args.input, clip_relpath)
    clip_dir=os.path.dirname(clip_path)
    clip_basename=os.path.basename(clip_path)
    split=row['split']
    split_dir = os.path.join(clip_dir, split)
    if not os.path.isdir(split_dir):
        os.mkdir(split_dir)
    
    new_clip_path=os.path.join(split_dir, clip_basename)
    new_clip_relpath=os.path.relpath(new_clip_path, args.input)
    shutil.move(clip_path, new_clip_path)
    return new_clip_relpath

def validate_clipfile(f, wav, wav_obj=None):
    try:
        if wav_obj is None:
            wav_obj=soundfile.read(wav, always_2d=True)[0]
        if len(wav_obj)==0:
            f.write(f"{wav} is empty.\n")
    except Exception as e:
        f.write(f"{wav} could not be opened due to exception {e}.\n")

# ------------------------- #
# Signal processing helpers #
# ------------------------- #

def get_clipped_segments(np_array: np.ndarray) -> Dict[str, Union[List[Tuple[int, int]], int]]:
    """
    Given numpy array representing audio samples
    return a list of tuples containing beginning and end indices of clipped segments,
    and an integer indicating the percentage of samples which are clipped.
    Taken from https://github.com/papercup-open-source/tutorials/tree/master/declipping on 7 November 2023.
    """
    nmax = max(np_array)
    nmin = min(np_array)

    clipped_segments = []
    clipped_samples = 0
    inside_clip = False
    clip_start = 0
    clip_end = 0

    for i, sample in enumerate(np_array):
        if (sample <= nmin + 1) or (sample >= nmax - 1):  # sample equal to or extremely close to max or min
            if not inside_clip:
                inside_clip = True  # declare we are inside clipped segment
                clip_start = i  # this is the first clipped sample

        elif inside_clip:
            inside_clip = False  # not longer inside clipped segment
            clip_end = i-1  # previous sample is end of segment
            clipped_segment = (clip_start, clip_end)  # save segment as tuple
            clipped_samples += clip_end-clip_start+1 # save number of samples in segment
            clipped_segments.append(clipped_segment)  # store tuple in list of clipped segments

    percent_clipped = clipped_samples / len(np_array)
    return {
        'clipped_segments': clipped_segments,
        'percent_clipped': percent_clipped,
    }

# --------------- #
# Dataset helpers #
# --------------- #

def load_dataset_safe(args) -> Union[Dataset, DatasetDict]:
    """
    If dataset points to a path on disk, load using `load_from_disk`,
    otherwise use `load_dataset` (to load from HF hub or local cache).
    """
    if hasattr(args, 'dataset'):
        dataset_path=args.dataset
    else:
        dataset_path=args.input
    split=getattr(args, 'split', None)
    make_split=getattr(args, 'make_split', False)
    if os.path.exists(dataset_path):
        dataset=load_from_disk(dataset_path)
        if split and args.num_records:
            return dataset[split].select(range(args.num_records))
        if split:
            return dataset[split]
        if make_split:
            dataset=make_ds_split(dataset)
        if args.num_records:
            for split in dataset:
                dataset[split]=dataset[split].select(range(args.num_records))
        return dataset    

    if 'fleurs' in dataset_path:
        return load_dataset(dataset_path, args.fleurs_lang, split=split, streaming=args.stream)
    dataset = load_dataset(dataset_path, split=split)
    if (args.num_records) and (not args.stream) and (split):
        dataset = dataset.select(range(args.num_records))
    return dataset

def make_ds_split(dataset: DatasetDict, percent_val: float=0.2) -> DatasetDict:
    """
    Make an ad-hoc train-val split.
    Assume dataset only has `train`.
    Select the first `percent_val` records to go into the validation split.
    """
    num_records = len(dataset['train'])
    num_val=int(percent_val*num_records)
    dataset['validation']=dataset['train'].select(range(num_val))
    dataset['train']=dataset['train'].select(range(num_val, num_records))
    return dataset

def save_dataset_safe(args, dataset, output_path: Optional[str]=None):
    """
    If dataset is IterableDataset, first cast to list of rows then to pandas.Dataframe 
    before saving to .csv. Otherwise, call `Dataset.to_csv`.
    Default to saving csv to path specified in `args.output`.
    Pass `output_path` arg to override this.
    """
    output_path = output_path or args.output

    if type(dataset) is Dataset:
        dataset.to_csv(output_path)
        return
    
    if type(dataset) is DatasetDict:
        split_dfs=[]
        for split in dataset:
            split_df=dataset[split].to_pandas()
            split_df['split']=split
            split_dfs.append(split_df)
        df=pd.concat(split_dfs)
        df.to_csv(output_path, index=False)
        return
    
    # IterableDataset

    with open(output_path, 'w') as f:
        first_row = next(iter(dataset))
        writer = csv.DictWriter(f, fieldnames=first_row.keys())
        writer.writeheader()
        writer.writerow(first_row)

        for i, row in tqdm(enumerate(dataset), total=args.num_records-1):
            if args.num_records and (i == args.num_records-1):
                break
            writer.writerow(row)

def get_remove_cols(args, ds):
    if args.keep_cols:
        return 'audio'
    if type(ds) is DatasetDict:
        return ds.values()[0].column_names
    return ds.column_names

# -------------- #
# String helpers #
# -------------- #

def normalize_str(s: Union[str, List[str]], args) -> Union[str, List[str]]:
    """
    Perform normalizations on str `s` depending on options in `args`.
    If `args.no_tone`, remove tone diacritics but keep other non-ascii characters.
    If `args.ascii_only`, use `unidecode` to remove or change all non-ascii characters.
    If `args.no_space`, remove spaces.
    """ 
    if type(s) is list:
        return [normalize_str(x, args) for x in s]

    if args.no_tone:
        tone_markers = {
            'grave': "\u0300",
            'macrn': "\u0304",
            'acute': "\u0301",
            'circm': "\u0302",
            'caron': "\u030C",
        }
        for c in tone_markers.keys():
            s = s.replace(c, '')
    if args.ascii_only:
        s = unidecode(s)
    if args.no_space:
        s = ''.join(s.split())
    return s

def get_epitran(fleurs_lang_tag, script: Optional[str]=None):
    """
    Instantiate and return an Epitran transliteration object
    for the given `fleurs_lang`.
    """
    import epitran
    with open('meta/language_codes.json') as f:
        lang_codes = json.load(f)
    lang_dict = [d for d in lang_codes if d['fleurs']==fleurs_lang_tag][0]
    iso3 = lang_dict['iso3']
    if not script:
        script=lang_dict['fleurs_script']
    
    return epitran.Epitran(f"{iso3}-{script}")

# ------------------------- #
# Cosine similarity helpers #
# ------------------------- #

def sim_to_mean(embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    `embeds` is a k*d matrix (k embeddings of length d).
    Return a vector of length k where each element is the similarity of
    the k'^th embedding to the average of all embeddings,
    excluding masked embeddings.
    """
    mean_embed = torch.mean(embeds[mask], dim=0)
    mean_embed = mean_embed.expand(embeds.shape)
    cos_sim = torch.nn.functional.cosine_similarity(embeds, mean_embed)
    return cos_sim

def find_least_similar_embedding(embeds: torch.Tensor, mask: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """
    `embeds` is a k*d matrix (k embeddings of length d).
    Identify the least similar embedding to all others,
    then add it to the mask.
    """
    embed_sim = sim_to_mean(embeds, mask)
    # set masked elements to inf so they wont be minimum
    embed_sim[~mask]=float('inf')
    least_similar_idx = torch.argmin(embed_sim).item()
    # update mask
    mask[least_similar_idx]=False
    return least_similar_idx, mask

def partition_embeddings(embeds: torch.Tensor, split_ratio: float=0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split embeddings maximizing cosine distance between partitions,
    and return indices for each partition.
    """
    num_embeds = embeds.shape[0]
    mask = torch.ones(num_embeds, dtype=bool)
    splitsize = int(num_embeds*split_ratio)
    remain_splitsize = num_embeds-splitsize
    major_splitsize = max(splitsize, remain_splitsize)
    while len(embeds[mask]) > major_splitsize:
        _, mask = find_least_similar_embedding(embeds, mask)
    
    major_split_idcs = mask.nonzero().squeeze()
    minor_split_idcs = (~mask).nonzero().squeeze()

    if split_ratio >=0.5:
        return major_split_idcs, minor_split_idcs
    return minor_split_idcs, major_split_idcs
    
# --------------- #
# Command methods #
# --------------- #

def empty_command(args) -> int:
    print("Please specify a command.")
    return 1

def pool_eaf_data(args) -> int:
    eafs = glob(os.path.join(args.input, '**/*.eaf'), recursive=True)
    start, end, val, eaf_source, wav_source = [], [], [], [], []
    for eaf_fp in eafs:
        eaf=Elan.Eaf(eaf_fp)
        tiers=eaf.get_tier_names()
        if args.transcription_tier not in tiers:
            print(f"{eaf_fp} missing tier {args.transcription_tier}, skipping")
            continue
        intervals=eaf.get_annotation_data_for_tier(args.transcription_tier)
        if not intervals:
            continue
        eaf_start, eaf_end, eaf_val = zip(*intervals)
        start.extend(eaf_start)
        end.extend(eaf_end)
        val.extend(eaf_val)

        wav_fp = get_media_path(eaf)
        eaf_source.extend(eaf_fp for _ in intervals)
        wav_source.extend(wav_fp for _ in intervals)


    df = pd.DataFrame({
        'start':start,
        'end':end,
        'transcription':val,
        'eaf_source': eaf_source,
        'wav_source': wav_source,
    })
    csv_path = args.output
    if os.path.isdir(csv_path):
        csv_path=os.path.join(csv_path, 'metadata.csv')
    df.to_csv(csv_path, index=False)
    return 0

def make_clips(args) -> int:
    """
    Open input `.csv` file, create `clips` dir in output dir which is populated with
    `.wav` files for each interval in the input csv and add column with relative path of clip.
    If --output is csv file, save to output path, else if output is dir, save to `clipdata.csv`
    in output dir.
    """
    df=pd.read_csv(args.input)
    # choosing colname `file_name` bc that's what HF AudioFolder expects
    df['file_name']=''
    output_dir=args.output if os.path.isdir(args.output) else os.path.dirname(args.output)
    csv_path=args.output if args.output.endswith('.csv') else os.path.join(output_dir, 'clipdata.csv')
    clip_dir=os.path.join(output_dir, 'clips')
    os.makedirs(clip_dir, exist_ok=True)
    for wav_source in tqdm(df['wav_source'].unique()):
        is_wav_source=df['wav_source']==wav_source
        if args.check_clips_exist and check_clips_exist(is_wav_source, wav_source, clip_dir):
            # if clips already exist, skip loading in wav but still save clip filepaths
            wav, sampling_rate = None, None
        else:
            try:
                wav, sampling_rate=librosa.load(wav_source, sr=16_000, mono=True)
            except:
                with open(args.logfile, 'a') as f:
                    f.write("Could not open wav_source "+wav_source)
                continue
        df.loc[is_wav_source, 'file_name']=df.loc[is_wav_source].progress_apply(
            lambda row: clip_segment(
                wav=wav,
                start_ms=row['start'],
                end_ms=row['end'],
                wav_basename=os.path.basename(wav_source),
                target_dir=clip_dir,
                sampling_rate=sampling_rate,
            ),
            axis=1
        )
    df.to_csv(csv_path, index=False)
    return 0

def validate_clips(args) -> int:
    """
    Checks every .wav file in input dir can be opened and has non-zero length.
    Saves a list of errors to output file.
    """
    if args.is_dataset:
        ds=load_dataset_safe(args)
        with open(args.output, 'w') as f:
            ds.map(lambda r: validate_clipfile(f, r['audio']['path'], wav_obj=r['audio']['array']))
        return 0
    if args.recursive:
        wavs=glob(os.path.join(args.input, '**/*.wav'), recursive=True)
    else:
        wavs=glob(os.path.join(args.input, '*.wav'))
    with open(args.output, 'w') as f:
        for wav in tqdm(wavs, desc='Validating wav files'):
            validate_clipfile(f, wav)
    return 0

def make_hf_dataset(args) -> int:
    """
    Create an AudioFolder dataset from input dir and saves to output dir.
    Expects input dir to contain `metadata.csv` with column `clip` or `file_name.`
    """
    metadata_path=os.path.join(args.input, 'metadata.csv')
    df = pd.read_csv(metadata_path)
    if 'file_name' not in df.columns:
        # assume csv has column 'clip' which includes absolute path
        # or path relative from project directory for each clip
        # add new column 'file_name' which is relative to dataset directory
        df['file_name']=df['clip'].apply(
            lambda s: os.path.join(
                os.path.relpath(s, args.input)
            )
        )
    if args.make_splits:
        df['file_name']=df.apply(lambda r: move_clip_to_split(r, args), axis=1)
    df.to_csv(metadata_path, index=False)
    del df
    ds = load_dataset("audiofolder", data_dir=args.input)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds.save_to_disk(args.output)
    return 0

def remove_extra_clips(args) -> int:
    """
    Looks for any clips not found in input `.csv` file and deletes from `clips` dir.
    """
    df = pd.read_csv(args.input)
    data_dir = os.path.dirname(args.input)
    clip_dir = args.output or os.path.join(
        data_dir,
        'clips',
    )
    clip_wavs = glob(os.path.join(clip_dir, '*.wav'))
    clip_wavs = [os.path.relpath(path, data_dir) for path in clip_wavs]
    df_clip_wavs = df['clip'].apply(lambda path: os.path.relpath(path, data_dir))
    for clip_wav in tqdm(clip_wavs):
        if clip_wav not in df_clip_wavs:
            os.remove(clip_wav)

def infer_asr(args) -> int:
    """
    Run ASR using HF pipeline on `audio` column in input dataset.
    Save output csv with results in column named after model checkpoint specified.
    """
    ds = load_dataset_safe(args)
    model = load_whisper_pipeline(args)
    if args.language==['all']:
        with open('meta/language_codes.json') as f:
            language_codes=json.load(f)
        args.language=[lang['whisper'] for lang in language_codes if 'whisper' in lang]
    if args.language:
        language_prompts={}
        tokenizer=AutoTokenizer.from_pretrained(args.model)
        for language in tqdm(args.language, desc='Getting language prompt tokens...'):
            lang_prompt = tokenizer.get_decoder_prompt_ids(
                language=language,
                task="transcribe"
            )
            language_prompts[language]=lang_prompt
    def map_pipe(batch):
        out={}
        if args.language:
            if len(args.language)==1:
                language=args.language[0]
                result = pipe(
                        [audio['array'] for audio in batch['audio']],
                        generate_kwargs={'forced_decoder_ids': language_prompts[language]},
                )
                out[language] = [item['text'] for item in result]
            else:
                for language in tqdm(args.language, desc='Transcribing batch for specified languages...'):
                    result = pipe(
                        [audio['array'] for audio in batch['audio']],
                        generate_kwargs={'forced_decoder_ids': language_prompts[language]},
                    )
                    out[language] = [item['text'] for item in result]
        else:
            result = pipe([audio['array'] for audio in batch['audio']])
            model_col = args.model.split(sep='/')[-1]
            out[model_col] = [item['text'] for item in result]
        out['path'] = [audio['path'] for audio in batch['audio']]
        return out
    remove_columns = get_remove_cols(args, ds)
    ds=ds.map(map_pipe, batched=True, batch_size=args.batch_size, remove_columns=remove_columns)
    save_dataset_safe(args, ds)
    return 0

def infer_vad(args) -> int:
    """
    Run VAD using PyAnnote speaker diarization set to detect one speaker.
    Add column indicating number of ms of detected speech.
    """
    from pyannote.audio import Pipeline

    ds = load_dataset_safe(args)
    pipe=Pipeline.from_pretrained(args.model)
    drz='diarization' in args.model
    pipe.to(torch.device(args.device))
    def map_pipe(row):
        if drz:
            # only pass num_speakers if using a diarization model
            result = pipe(
                {
                    'waveform': torch.tensor(row['audio']['array']).unsqueeze(0).to(args.device).float(),
                    'sample_rate': row['audio']['sampling_rate'],
                },
                num_speakers=1,
            )
        else:
            result = pipe(
                {
                    'waveform': torch.tensor(row['audio']['array']).unsqueeze(0).to(args.device).float(),
                    'sample_rate': row['audio']['sampling_rate'],
                },
            )
        out={}
        # item.chart() returns list of shape [('SPEAKER_00', num_sec)]
        model_col = args.model.split(sep='/')[-1]
        out[model_col]=result.to_lab().replace('\n', ';')
        out['path'] = row['audio']['path']
        return out
    remove_columns = get_remove_cols(args, ds)
    ds=ds.map(map_pipe, remove_columns=remove_columns)
    save_dataset_safe(args, ds)
    return 0

def infer_allosaurus(args):
    """
    Load in HF audio dataset and run Allosaurus on each row.
    """
    from allosaurus.app import read_recognizer
    import torchaudio
    ds = load_dataset_safe(args)
    config=Namespace(
        model=args.model,
        device_id=args.device,
        lang=args.lang,
        approximate=False,
        prior=None
    )
    model=read_recognizer(config)
    def map_allosaurus(row):
        clip_paths=[audio['path'] for audio in row['audio']]
        clip_basenames=[os.path.basename(clip) for clip in clip_paths]
        audio_tensors=[torch.tensor(audio['array']).unsqueeze(0) for audio in row['audio']]
        sample_rate=row['audio'][0]['sampling_rate']
        result=[]
        with TemporaryDirectory() as tempdir:
            audio_paths=[os.path.join(tempdir, basename) for basename in clip_basenames]
            for path, tensor in zip(audio_paths, audio_tensors):
                torchaudio.save(path, tensor, sample_rate)
                result.append(model.recognize(path, args.lang))
        out = {
            'path': row['clip'],
            'allosaurus': result,
        }
        return out
    remove_columns = get_remove_cols(args, ds)
    ds=ds.map(map_allosaurus, batched=True, batch_size=args.batch_size, remove_columns=remove_columns)
    save_dataset_safe(args, ds)
    return 0

def clap_ipa_sim(args) -> int:
    """
    Computes speech and text embeddings for audio dataset.
    Saves cosine similarity scores to `clap_ipa_sim.csv`,
    speech embeddings to `clap_ipa_speech_embeds.pt`
    and phone embeddings to `clap_ipa_phone_embeds.pt`,
    all in output dir.
    """
    from clap.encoders import SpeechEncoder, PhoneEncoder
    # Code taken in part from https://github.com/lingjzhu/clap-ipa
    # TODO: make different functions for speech embeddings, phone emebeddings
    # and cos similarity for more efficient computation
    print("Loading clap-ipa speech encoder...")
    speech_encoder = SpeechEncoder.from_pretrained(f'anyspeech/clap-ipa-{args.model_size}-speech')
    print("Loading clap-ipa phone encoder...")
    phone_encoder = PhoneEncoder.from_pretrained(f'anyspeech/clap-ipa-{args.model_size}-phone')
    phone_encoder.eval().to(args.device)
    speech_encoder.eval().to(args.device)

    tokenizer = DebertaV2Tokenizer.from_pretrained('charsiu/IPATokenizer')
    processor = AutoProcessor.from_pretrained('openai/whisper-tiny')

    ds = load_dataset_safe(args)
    phone_embeds = []
    speech_embeds = []

    if args.g2p:
        print("Loading Epitran G2P...")
        epitran_obj = get_epitran(args.fleurs_lang, args.script)

    def process_str(s_list):
        if args.g2p:
            s_list = [epitran_obj.transliterate(s) for s in s_list]
        s_list = normalize_str(s_list, args)
        return tokenizer(
            s_list,
            return_tensors='pt',
            return_token_type_ids=False,
            padding=True,
        )

    def map_clapipa(row):
        audio_arrays = [audio['array'] for audio in row['audio']]
        audio_paths = [audio['path'] for audio in row['audio']]
        sampling_rate = row['audio'][0]['sampling_rate']
        audio_input = processor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors='pt',
            return_attention_mask=True,
        )
        ipa_input = process_str(row['transcription'])
        audio_input=audio_input.to(args.device)
        ipa_input=ipa_input.to(args.device)

        with torch.no_grad():
            speech_embed = speech_encoder(**audio_input)['pooler_output'].to('cpu')
            phone_embed = phone_encoder(**ipa_input)['pooler_output'].to('cpu')
        similarity = torch.nn.functional.cosine_similarity(speech_embed,phone_embed,dim=-1).tolist()

        del audio_input
        del ipa_input

        phone_embeds.append(phone_embed)
        speech_embeds.append(speech_embed)

        return {
            'similarity': similarity,
            'path':  audio_paths,
        }
    remove_columns = get_remove_cols(args, ds)
    ds = ds.map(map_clapipa, batched=True, batch_size=args.batch_size, remove_columns=remove_columns)
    csv_path = os.path.join(args.output, 'clap_ipa_sim.csv')
    save_dataset_safe(args, ds, output_path=csv_path)

    # convert tensor lists into tensor matrices and save
    speech_embeds=torch.concat(speech_embeds, dim=0)
    speech_embed_path=os.path.join(args.output, 'clap_ipa_speech_embeds.pt')
    torch.save(speech_embeds, speech_embed_path)

    phone_embeds=torch.concat(phone_embeds, dim=0)
    phone_embed_path=os.path.join(args.output, 'clap_ipa_phone_embeds.pt')
    torch.save(phone_embeds, phone_embed_path)
    
    return 0

def clap_ipa_text_sim(args) -> int:
    """
    Expects a csv file as input. Calculates cosine similarity of phone embeddings
    for --col1 and --col2 and outputs to a csv file.
    """
    from clap.encoders import PhoneEncoder
    print("Loading clap-ipa phone encoder...")
    phone_encoder = PhoneEncoder.from_pretrained(f'anyspeech/clap-ipa-{args.model_size}-phone')
    phone_encoder.eval().to(args.device)

    tokenizer = DebertaV2Tokenizer.from_pretrained('charsiu/IPATokenizer')
    process_str = lambda s: tokenizer(
            normalize_str(s, args),
            return_tensors='pt',
            return_token_type_ids=False,
            padding=True,
    )

    df = pd.read_csv(args.input, keep_default_na=False)
    def map_clapipa(batch):
        col1_input = process_str(batch[args.col1])
        col1_input=col1_input.to(args.device)

        col2_input = process_str(batch[args.col2])
        col2_input=col2_input.to(args.device)

        with torch.no_grad():
            col1_embed = phone_encoder(**col1_input)['pooler_output'].to('cpu')
            col2_embed = phone_encoder(**col2_input)['pooler_output'].to('cpu')
        similarity = torch.nn.functional.cosine_similarity(col1_embed,col2_embed,dim=-1)

        return similarity

    dl = torch.utils.data.DataLoader(df.to_dict('records'), batch_size=args.batch_size)
    sim = []
    for batch in tqdm(dl):
        sim.extend(map_clapipa(batch))
    df[f'clapipa-{args.col1}-{args.col2}'] = sim
    df.to_csv(args.output, index=False)
    return 0

def detect_clipping(args) -> int:
    """
    Detect segments of audio clipping for each wavfile in the input dataset.
    Saves output csv containing indices for clipped segments and the percentage
    of audio clipping for each record in the dataset.
    """
    ds = load_dataset_safe(args)
    def map_get_clipped_segments(row):
        clipped_dict = get_clipped_segments(row['audio']['array'])
        clipped_dict['path']=row['audio']['path']
        return clipped_dict
    remove_columns = get_remove_cols(args, ds)
    ds = ds.map(map_get_clipped_segments, remove_columns=remove_columns)
    save_dataset_safe(args, ds)
    return 0

def calculate_snr(args):
    """
    Calculate WADA SNR and NIST STNR for each audio record in the input dataset
    and saves to output .csv file.
    Uses matlab code from Labrosa at https://labrosa.ee.columbia.edu/projects/snreval/
    Downloaded 9 Sep 2024
    """
    from matlab import engine
    print("Loading matlab engine...")
    eng = engine.start_matlab()
    # look for SNREVAL matlab code at directory specified by SNREVAL_DIR env variable
    # if not specified, default to value in SNREVAL_DIR constant
    eng.addpath(os.environ.get('SNREVAL_DIR', SNREVAL_DIR))

    ds = load_dataset_safe(args)
    def map_snr(row):
        path=row['audio']['path']
        array=row['audio']['array']
        sampling_rate=row['audio']['sampling_rate']
        wada = eng.wada_snr(array, sampling_rate)
        nist_stnr = eng.nist_stnr_m(array, sampling_rate)
        return {'path': path, 'wada_snr': wada, 'nist_stnr': nist_stnr}
    remove_columns = get_remove_cols(args, ds)
    ds = ds.map(map_snr, remove_columns=remove_columns)
    save_dataset_safe(args, ds)

def split_dataset(args):
    """
    Load embeddings from `args.input` that correspond to rows
    the dataset specified in `args.dataset`. Create a train-test-val split
    that maximizes cosine distance between each partition.
    Save dataset to path specified in `args.output`.
    """
    ds = load_dataset_safe(args)
    train_size, val_size, test_size = args.splitsize
    embeds = torch.load(args.input)

    train, val_test = partition_embeddings(embeds, split_ratio=train_size)
    val_subidcs, test_subidcs = partition_embeddings(embeds[val_test], split_ratio=val_size/(val_size+test_size))
    val = val_test[val_subidcs]
    test = val_test[test_subidcs]

    ds=DatasetDict({
        'train': ds.select(train),
        'validation': ds.select(val),
        'test': ds.select(test)
    })

    ds.save_to_disk(args.output)

    return 0    

# ---- #
# main #
# ---- #

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == '__main__':
    main()