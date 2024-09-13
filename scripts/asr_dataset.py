from glob import glob
from pympi import Elan
from typing import Optional, Sequence, List, Tuple, Dict, Union
from argparse import ArgumentParser, Namespace
import pandas as pd
import os
import librosa
import numpy as np
import soundfile
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Audio, Dataset, DatasetDict, IterableDataset
from transformers import pipeline, AutoProcessor, DebertaV2Tokenizer
import torch
from tempfile import TemporaryDirectory
import csv
from unidecode import unidecode
import json
# TODO: move heavy imports (torch, transformers, datasets) into methods

GDRIVE_DIR = '/Users/markjos/Library/CloudStorage/GoogleDrive-mjsimmons@ucsd.edu/Shared drives/Tira/Recordings'
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
SNREVAL_DIR = '/Users/markjos/projects/snreval'
tqdm.pandas()

def init_parser() -> ArgumentParser:
    device_type = lambda s: int(s) if s!='cpu' else s
    # TODO: some command arguments are very wet. Make them DRY.
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', default=GDRIVE_DIR)
    parser.add_argument('--output', '-o')
    parser.add_argument('--logfile', '-l', default='error.txt')
    parser.add_argument('--split', '-s', default='train')
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

    hf_dataset_parser=commands.add_parser(
        'make_hf_dataset',
        help=make_hf_dataset.__doc__
    )
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
    infer_asr_parser.set_defaults(func=infer_asr)

    infer_vad_parser = commands.add_parser('infer_vad', help=infer_vad.__doc__)
    infer_vad_parser.add_argument('--model', '-m', default='pyannoate/speaker-diarization-3.1')
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
        wav: Optional[np.ndarray],
        start_ms: int,
        end_ms: int,
        wav_basename: str,
        target_dir: str,
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
    if os.path.exists(args.input):
        dataset=load_from_disk(args.input)
        if args.split and args.num_records:
            return dataset[args.split][:args.num_records]
        if args.split:
            return dataset[args.split]
        return dataset    

    if 'fleurs' in args.input:
        return load_dataset(args.input, args.fleurs_lang, split=args.split, streaming=args.stream)
    dataset = load_dataset(args.input, split=args.split)
    if (args.num_records) and (not args.stream) and (args.split):
        dataset = dataset[:args.num_records]
    return dataset

def save_dataset_safe(args, dataset):
    """
    If dataset is IterableDataset, first cast to list of rows then to pandas.Dataframe 
    before saving to .csv. Otherwise, call `Dataset.to_csv`.
    """
    if type(dataset) is Dataset:
        dataset.to_csv(args.output)
        return
    
    # IterableDataset

    with open(args.output, 'w') as f:
        first_row = next(iter(dataset))
        writer = csv.DictWriter(f, fieldnames=first_row.keys())
        writer.writeheader()
        writer.writerow(first_row)

        for i, row in tqdm(enumerate(dataset), total=args.num_records-1):
            if args.num_records and (i == args.num_records-1):
                break
            writer.writerow(row)

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
    with open('../meta/language_codes.json') as f:
        lang_codes = json.load(f)
    lang_dict = [d for d in lang_codes if d['fleurs']==fleurs_lang_tag][0]
    iso3 = lang_dict['iso3']
    if not script:
        script=lang_dict['fleurs_script']
    
    return epitran.Epitran(f"{iso3}-{script}")

    
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
    pipe=pipeline('automatic-speech-recognition', args.model, device=args.device)
    def map_pipe(row):
        result = pipe([audio['array'] for audio in row['audio']])
        out={}
        model_col = args.model.split(sep='/')[-1]
        out[model_col] = [item['text'] for item in result]
        out['path'] = row['audio']['path']
        return out
    remove_columns = 'audio' if args.keep_cols else ds.column_names
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
    remove_columns = 'audio' if args.keep_cols else ds.column_names
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
    remove_columns = 'audio' if args.keep_cols else ds.column_names
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
        epitran_obj = get_epitran(args.fleurs_lang, args.script)

    def process_str(s):
        s = normalize_str(s, args)
        if args.g2p:
            s = epitran_obj.transliterate(s)
        return tokenizer(
            s,
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
        similarity = torch.nn.functional.cosine_similarity(speech_embed,phone_embed,dim=-1).item()

        del audio_input
        del ipa_input

        phone_embeds.append(phone_embed)
        speech_embeds.append(speech_embed)

        return {
            'similarity': similarity,
            'path':  audio_paths,
        }
    remove_columns = 'audio' if args.keep_cols else ds.column_names
    ds = ds.map(map_clapipa, batched=True, batch_size=args.batch_size, remove_columns=remove_columns)

    sim_df = pd.DataFrame({'path': ds['path'], 'clap_ipa_cos_sim': ds['similarity']})
    sim_csv_path=os.path.join(args.output, 'clip_ipa_sim.csv')
    sim_df.to_csv(sim_csv_path, index=False)

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
    remove_columns = 'audio' if args.keep_cols else ds.column_names
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
    remove_columns = 'audio' if args.keep_cols else ds.column_names
    ds = ds.map(map_snr, remove_columns=remove_columns)
    save_dataset_safe(args, ds)

# ---- #
# main #
# ---- #

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == '__main__':
    main()