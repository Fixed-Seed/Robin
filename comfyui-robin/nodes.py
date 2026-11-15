import asyncio
import hashlib
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from aiohttp import web

import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
import folder_paths
from server import PromptServer

from .robin import Robin, pieces
from .robin import train as trainer
from .robin.vae import load_vae
from .robin.pipeline import SPEECH, VAE, WEIGHTS

folder_paths.add_model_folder_path('robin', os.path.join(folder_paths.models_dir, 'robin'))

REPO, MODEL, LORA = 'FixedSeed/Robin', 'robin', 'robin_character_lora.safetensors'
MODEL_FILES = (WEIGHTS, 'config.json', 'tokenizer.json', SPEECH)
DTYPES = dict(bf16=torch.bfloat16, fp32=torch.float32)
LANGUAGES = {'English': 'en', 'Japanese': 'ja', 'English + Japanese': 'mix'}
UPLOADS = 'robin_datasets'
SOUND_FILES = ('.wav', '.flac', '.mp3', '.ogg', '.opus', '.aiff', '.aif')
MAX_SOURCE_SECONDS = 45
WORKING_MEMORY = 2 * 1024 ** 3
TRAINING_MEMORY = 4 * 1024 ** 3
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}')
SAVED = re.compile(r'step_\d+')


def model_folders():
    found = set()
    for base in folder_paths.get_folder_paths('robin'):
        base = Path(base)
        if base.is_dir():
            for config in base.rglob('config.json'):
                if config.parent != base and (config.parent/WEIGHTS).is_file():
                    found.add(config.parent.relative_to(base).as_posix())
    return [MODEL] + sorted(found - {MODEL})


def model_path(name):
    for base in folder_paths.get_folder_paths('robin'):
        path = Path(base)/name
        if (path/'config.json').is_file():
            return path
    return Path(folder_paths.get_folder_paths('robin')[0])/name


def shown(path):
    try:
        return Path(path).resolve().relative_to(Path(folder_paths.base_path).resolve()).as_posix()
    except ValueError:
        return str(path)


def own_vae(vae):
    return vae == VAE and not folder_paths.get_full_path('vae', VAE)


def model_status(model, vae=VAE):
    path = model_path(model)
    files = MODEL_FILES + ((VAE,) if own_vae(vae) else ())
    present = [path/f for f in files if (path/f).is_file()]
    return dict(model=model, ready=len(present) == len(files), size=sum(f.stat().st_size for f in present),
                voice=bool(folder_paths.get_full_path('loras', LORA)))


def download(file, folder):
    from huggingface_hub import hf_hub_download
    logging.info(f'Robin: downloading {file} from Hugging Face ({REPO}) into {folder}')
    try:
        hf_hub_download(REPO, file, local_dir=str(folder))
    except Exception as e:
        raise RuntimeError(f'Robin could not download {file} from Hugging Face ({REPO}): {e}') from e


def lora_path():
    found = folder_paths.get_full_path('loras', LORA)
    if found:
        return found
    folder = Path(folder_paths.get_folder_paths('loras')[0])
    download(LORA, folder)
    return str(folder/LORA)


def loras():
    found = [f for f in folder_paths.get_filename_list('loras') if f.endswith('.safetensors') and '/cache/' not in f.replace(os.sep, '/')]
    return [LORA] + [f for f in found if f != LORA]


def input_audio():
    folder = Path(folder_paths.get_input_directory())
    return sorted(f.name for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SOUND_FILES) if folder.is_dir() else []


def dataset_path(folder):
    path = Path(folder.strip()).expanduser()
    return path if path.is_absolute() else Path(folder_paths.get_input_directory())/path


def folder_signature(folder):
    path = dataset_path(folder)
    if not folder.strip() or not path.is_dir():
        return ''
    files = sorted(f for f in path.rglob('*') if f.is_file())
    return hashlib.sha256('\n'.join(f'{f}:{f.stat().st_size}:{f.stat().st_mtime_ns}' for f in files).encode()).hexdigest()


def check_folder(folder, language):
    if not folder.strip():
        return None, 'Choose a folder of clips, or type its path.'
    path = dataset_path(folder)
    if not path.is_dir():
        return None, f'There is no folder at {path}.'
    lang = LANGUAGES[language]
    found = trainer.scan(path, lang)
    found['folder'] = shown(path)
    lines = []
    try:
        trainer.describe(found, log=lines.append)
    except SystemExit as e:
        lines.append(str(e))
    return dict(folder=str(path), lang=lang), '\n'.join(lines)


def tell(event, node, **data):
    server = PromptServer.instance
    data = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in data.items()}
    server.send_sync(event, dict(node=node, **data), server.client_id)


def preview(path, subfolder):
    folder = Path(folder_paths.get_temp_directory())/subfolder
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, folder/Path(path).name)
    return dict(filename=Path(path).name, subfolder=subfolder, type='temp')


def take(audio, seed):
    folder = Path(folder_paths.get_temp_directory())/'robin'
    folder.mkdir(parents=True, exist_ok=True)
    name = f'take_{uuid.uuid4().hex[:12]}.flac'
    sf.write(str(folder/name), np.clip(audio, -1, 1), pieces.RATE, subtype='PCM_24')
    return dict(filename=name, subfolder='robin', type='temp', seconds=round(len(audio) / pieces.RATE, 2), seed=seed)


def trainable(robin):
    for net in (robin.planner, robin.decoder):
        for module in net.modules():
            for key, value in list(module._parameters.items()):
                if value is not None and value.is_inference():
                    module._parameters[key] = torch.nn.Parameter(value.clone(), requires_grad=False)
            for key, value in list(module._buffers.items()):
                if value is not None and value.is_inference():
                    module._buffers[key] = value.clone()
            for key, value in list(vars(module).items()):
                if isinstance(value, torch.Tensor) and value.is_inference():
                    setattr(module, key, value.clone())


def devices():
    return ['default', 'cpu'] + ([f'cuda:{i}' for i in range(torch.cuda.device_count())] if torch.cuda.device_count() > 1 else [])


class Core:
    def __init__(self, robin, load_device, offload_device):
        self.robin, self.load_device, self.offload_device = robin, load_device, offload_device
        bundle = torch.nn.ModuleDict(dict(planner=robin.planner, decoder=robin.decoder, vae=robin.vae))
        self.patcher = self.patch(bundle)
        self.speech_patcher = self.lora_patcher = None
        self.lora_file = None

    def patch(self, module):
        return comfy.model_patcher.ModelPatcher(module, load_device=self.load_device, offload_device=self.offload_device)


class RobinModel:
    def __init__(self, core, lora=None):
        self.core, self.lora = core, lora

    def ready(self, speech=False, memory=WORKING_MEMORY):
        core, robin = self.core, self.core.robin
        if self.lora is None:
            if robin.lora is not None:
                robin.remove_lora(); core.lora_patcher = core.lora_file = None
        else:
            path, scale = self.lora
            if core.lora_file != path:
                robin.load_lora(path, scale)
                core.lora_file, core.lora_patcher = path, core.patch(robin.lora)
            robin.set_lora_scale(scale)
        patchers = [core.patcher] + ([core.lora_patcher] if core.lora_patcher else [])
        if speech:
            tokenizer = robin.speech
            if core.speech_patcher is None:
                core.speech_patcher = core.patch(torch.nn.ModuleDict(dict(speech=tokenizer.model)))
            patchers.append(core.speech_patcher)
        mm.load_models_gpu(patchers, memory_required=memory, force_full_load=True)
        return robin


class Progress:
    def __init__(self, renders):
        self.bar, self.total, self.done = comfy.utils.ProgressBar(100 * renders), renders, 0
        self.expected, self.plan, self.decode = 1, 0., 0.

    def start(self, seconds):
        self.expected, self.plan, self.decode = max(1., seconds * 25), 0., 0.
        self.show()

    def __call__(self, stage, step, total):
        mm.throw_exception_if_processing_interrupted()
        if stage == 'plan':
            self.plan = min(1., step / self.expected)
            if step % 25: return
        else:
            self.plan, self.decode = 1., step / total
        self.show()

    def finish(self, extra=0):
        self.done += 1; self.total += extra; self.show()

    def show(self):
        within = .8 * self.plan + .2 * self.decode
        self.bar.update_absolute(int(100 * (self.done + (within if self.done < self.total else 0))), 100 * self.total)


def mono(audio):
    wave = audio['waveform']
    return wave[0].float().mean(0).cpu().numpy(), int(audio['sample_rate'])


def to_audio(x, rate=pieces.RATE):
    return {'waveform': torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None], 'sample_rate': rate}


SAMPLING = {
    'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True}),
    'temperature': ('FLOAT', {'default': .9, 'min': 0., 'max': 2., 'step': .01, 'tooltip': 'Higher is livelier and less predictable; 0 always picks the likeliest sound.'}),
    'top_k': ('INT', {'default': 25, 'min': 0, 'max': 500, 'tooltip': 'Choose among only this many likeliest sounds at each step (0 = no limit).'}),
    'top_p': ('FLOAT', {'default': .8, 'min': .01, 'max': 1., 'step': .01, 'tooltip': 'Choose among only the likeliest sounds that add up to this share.'}),
}
RETRIES = ('INT', {'default': 2, 'min': 0, 'max': 6, 'advanced': True,
                   'tooltip': 'Tries a piece again with a new seed when it comes out silent or the wrong length.'})


class RobinLoader:
    DESCRIPTION = ('Loads Robin from a folder in models/robin. The first run downloads whatever is missing from Hugging Face, '
                   f'along with Robin\'s own voice ({LORA}) for the Robin LoRA node.')
    RETURN_TYPES = ('ROBIN_MODEL',)
    RETURN_NAMES = ('robin',)
    FUNCTION = 'load'
    CATEGORY = 'Robin'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': (model_folders(), {'default': MODEL, 'tooltip': f'A Robin folder in models/robin. "{MODEL}" is downloaded on the first run if it is not there yet.'}),
                'vae': ([VAE] + [f for f in folder_paths.get_filename_list('vae') if f != VAE],
                        {'default': VAE, 'tooltip': f'The VAE weights Robin decodes with. {VAE} is downloaded into the Robin folder on the '
                                                    'first run, unless models/vae already has it.'}),
                'device': (devices(), {'default': 'default'}),
                'dtype': (list(DTYPES), {'default': 'bf16', 'tooltip': 'bf16 needs about 8 GB of GPU memory; fp32 about twice that.'}),
            },
            'hidden': {'unique_id': 'UNIQUE_ID'},
        }

    def load(self, model, vae, device, dtype, unique_id=None):
        path = model_path(model)
        missing = [f for f in MODEL_FILES + ((VAE,) if own_vae(vae) else ()) if not (path/f).is_file()]
        for file in missing:
            tell('robin.loader', unique_id, stage='download', file=file)
            download(file, path)
        if not folder_paths.get_full_path('loras', LORA):
            tell('robin.loader', unique_id, stage='download', file=LORA)
        lora_path()
        tell('robin.loader', unique_id, stage='load')
        load_device = mm.get_torch_device() if device == 'default' else torch.device(device)
        offload_device = load_device if load_device.type == 'cpu' else mm.unet_offload_device()
        vae_file = path/VAE if own_vae(vae) else folder_paths.get_full_path_or_raise('vae', vae)
        try:
            decoder = load_vae(vae_file, offload_device)
        except (RuntimeError, KeyError) as e:
            raise ValueError(f'{vae} is not a VAE Robin can decode with.') from e
        robin = Robin.from_pretrained(str(path), vae=decoder, device=offload_device, dtype=DTYPES[dtype])
        robin.device = load_device
        status = dict(model_status(model, vae), loaded=dtype, downloaded=bool(missing))
        return {'ui': {'robin_status': [status]}, 'result': (RobinModel(Core(robin, load_device, offload_device)),)}


class RobinLoRA:
    DESCRIPTION = (f'Applies a voice LoRA trained with Robin. {LORA} is Robin\'s own voice. One LoRA at a time: a second LoRA node '
                   'replaces the first.')
    RETURN_TYPES = ('ROBIN_MODEL',)
    RETURN_NAMES = ('robin',)
    FUNCTION = 'apply'
    CATEGORY = 'Robin'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'robin': ('ROBIN_MODEL',),
                'lora': (loras(), {'default': LORA}),
                'strength': ('FLOAT', {'default': 1., 'min': 0., 'max': 2., 'step': .05, 'tooltip': '1 is the voice as trained, 0 is Robin without the LoRA.'}),
            },
        }

    def apply(self, robin, lora, strength):
        path = lora_path() if lora == LORA else folder_paths.get_full_path_or_raise('loras', lora)
        from safetensors import safe_open
        with safe_open(path, framework='pt', device='cpu') as f:
            if 'robin_lora' not in (f.metadata() or {}):
                raise ValueError(f'{lora} is not a Robin LoRA.')
        return (RobinModel(robin.core, (path, strength)),)


class RobinSpeak:
    DESCRIPTION = ('Reads text aloud in a voice you describe. Long text is read in pieces that carry the voice from one to the next. '
                   '[laughs] and [coughs] can go anywhere in the text. The last take plays on the node; connect Save Audio to keep '
                   'every take in the output folder.')
    RETURN_TYPES = ('AUDIO',)
    RETURN_NAMES = ('audio',)
    FUNCTION = 'speak'
    OUTPUT_NODE = True
    CATEGORY = 'Robin'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'robin': ('ROBIN_MODEL',),
                'text': ('STRING', {'multiline': True, 'default': 'Hello there! [laughs] It is so good to finally hear my own voice.',
                                    'tooltip': 'What to say. [laughs] and [coughs] can go anywhere in it.'}),
                'voice': ('STRING', {'multiline': True, 'default': 'A warm, low voice, unhurried and close to the mic.',
                                     'tooltip': 'The voice in plain words: age, pitch, pace, mood, how close to the mic. Leave it empty to let Robin choose.'}),
                **SAMPLING,
                'cfg': ('FLOAT', {'default': 1., 'min': 1., 'max': 10., 'step': .1,
                                  'tooltip': 'Above 1 follows the voice description more closely (try 3).'}),
            },
            'optional': {
                'retries': RETRIES,
                'max_words': ('INT', {'default': 35, 'min': 8, 'max': 60, 'advanced': True, 'tooltip': 'Long text is read in pieces of about this many words.'}),
                'language': (list(LANGUAGES), {'default': 'English', 'tooltip': 'The language of the text. English + Japanese is for lines that mix the two.'}),
            },
        }

    def speak(self, robin, text, voice, seed, temperature, top_k, top_p, cfg, retries=2, max_words=35, language='English'):
        lang = LANGUAGES[language]
        chunks = pieces.split(text, max_words, lang)
        if not chunks:
            raise ValueError('There is no text to read.')
        model = robin.ready(speech=len(chunks) > 1)
        audio = pieces.read(model, text, voice, seed=seed, cfg=cfg, retries=retries, max_words=max_words, lang=lang,
                            progress=Progress(len(chunks)), temperature=temperature, top_k=top_k, top_p=top_p)
        return {'ui': {'robin_take': [take(audio, seed)]}, 'result': (to_audio(audio),)}


class RobinVoiceChanger:
    DESCRIPTION = ('Performs a recording again in another voice: the same words, timing and delivery. The new voice comes from the '
                   'description, and from a Robin LoRA when one is connected. The last take plays on the node; connect Save Audio to '
                   'keep every take in the output folder.')
    RETURN_TYPES = ('AUDIO',)
    RETURN_NAMES = ('audio',)
    FUNCTION = 'change'
    OUTPUT_NODE = True
    CATEGORY = 'Robin'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'robin': ('ROBIN_MODEL',),
                'source_audio': ('AUDIO', {'tooltip': f'The performance to keep, up to {MAX_SOURCE_SECONDS} seconds.'}),
                'source_text': ('STRING', {'multiline': True, 'default': '', 'tooltip': 'Exactly what the source clip says.'}),
                'voice': ('STRING', {'multiline': True, 'default': '',
                                     'tooltip': 'The new voice in plain words. It can stay empty when a Robin LoRA is connected.'}),
                **SAMPLING,
            },
            'optional': {
                'retries': RETRIES,
                'language': (list(LANGUAGES), {'default': 'English', 'tooltip': 'The language of the recording. English + Japanese is for lines that mix the two.'}),
            },
        }

    def change(self, robin, source_audio, source_text, voice, seed, temperature, top_k, top_p, retries=2, language='English'):
        if not source_text.strip():
            raise ValueError('Fill in source_text with exactly what the source clip says.')
        if not voice.strip() and robin.lora is None:
            raise ValueError('Describe the new voice, or connect a Robin LoRA.')
        source, source_rate = mono(source_audio)
        seconds = len(source) / source_rate
        if seconds > MAX_SOURCE_SECONDS:
            raise ValueError(f'The source clip is {seconds:.0f} s long; the Voice Changer takes up to {MAX_SOURCE_SECONDS} s at a time.')
        model = robin.ready(speech=True)
        audio = pieces.change(model, source, source_rate, source_text, voice=voice, seed=seed, retries=retries, lang=LANGUAGES[language],
                              progress=Progress(2), temperature=temperature, top_k=top_k, top_p=top_p)
        return {'ui': {'robin_take': [take(audio, seed)]}, 'result': (to_audio(audio),)}


class RobinLoadAudio:
    DESCRIPTION = ('A recording for the Voice Changer: record one with your microphone, or choose an audio file. Recordings are '
                   'trimmed to the line and kept in ComfyUI\'s input folder.')
    RETURN_TYPES = ('AUDIO',)
    RETURN_NAMES = ('audio',)
    FUNCTION = 'load'
    CATEGORY = 'Robin'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'audio': (input_audio(), {'tooltip': 'A recording in ComfyUI\'s input folder. Record and Choose file add new ones.'}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, audio):
        path = Path(folder_paths.get_annotated_filepath(audio))
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ''

    @classmethod
    def VALIDATE_INPUTS(cls, audio):
        if not audio or not folder_paths.exists_annotated_filepath(audio):
            return 'Record a line or choose an audio file first.' if not audio else f'There is no {audio} in ComfyUI\'s input folder.'
        return True

    def load(self, audio):
        x, rate = sf.read(folder_paths.get_annotated_filepath(audio), dtype='float32', always_2d=True)
        return ({'waveform': torch.from_numpy(np.ascontiguousarray(x.T))[None], 'sample_rate': int(rate)},)


class RobinCheckDataset:
    DESCRIPTION = ('Checks a folder of clips for training a voice: each audio file needs a .txt transcript with the same name, and a '
                   'voice.txt can describe the voice. Lists the clips it will leave out and anything worth fixing, in plain words. '
                   f'Choose folder uploads a folder from your computer into input/{UPLOADS}.')
    RETURN_TYPES = ('ROBIN_DATASET', 'STRING')
    RETURN_NAMES = ('dataset', 'report')
    FUNCTION = 'check'
    OUTPUT_NODE = True
    CATEGORY = 'Robin'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'folder': ('STRING', {'default': '', 'tooltip': 'The folder of clips: audio files, each with a .txt transcript of the same name. '
                                                                'A relative path is inside ComfyUI\'s input folder.'}),
                'language': (list(LANGUAGES), {'default': 'English', 'tooltip': 'The language of the recordings.'}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, folder, language):
        return folder_signature(folder) + language

    def check(self, folder, language):
        dataset, report = check_folder(folder, language)
        if dataset is None:
            raise ValueError(report)
        return {'ui': {'text': [report]}, 'result': (dataset, report)}


class RobinTrainLoRA:
    DESCRIPTION = ('Trains a voice LoRA on a checked dataset, with the same recipe as the Robin app. Every save writes a checkpoint and '
                   'reads the sample line in the new voice, into models/loras/robin/<name>. Cancel the queue to stop; running it again '
                   'continues from the last save.')
    RETURN_TYPES = ('ROBIN_MODEL', 'STRING')
    RETURN_NAMES = ('robin', 'lora_name')
    FUNCTION = 'train'
    OUTPUT_NODE = True
    CATEGORY = 'Robin'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'robin': ('ROBIN_MODEL',),
                'dataset': ('ROBIN_DATASET',),
                'name': ('STRING', {'default': 'my_voice', 'tooltip': 'The LoRA is saved as models/loras/robin/<name>/lora.safetensors.'}),
                'steps': ('INT', {'default': 1000, 'min': 10, 'max': 20000, 'step': 10}),
                'save_every': ('INT', {'default': 200, 'min': 10, 'max': 5000, 'step': 10, 'tooltip': 'How often to save a checkpoint and read the sample line.'}),
                'sample_text': ('STRING', {'multiline': True, 'default': trainer.SAMPLE_TEXT['en'],
                                           'tooltip': 'Read in the new voice at every save, so you can pick a step by ear.'}),
                'restart': ('BOOLEAN', {'default': False, 'tooltip': 'Start over instead of continuing an unfinished run with this name.'}),
            },
            'optional': {
                'rank_planner': ('INT', {'default': 32, 'min': 0, 'max': 256, 'step': 8, 'advanced': True}),
                'rank_decoder': ('INT', {'default': 64, 'min': 0, 'max': 256, 'step': 8, 'advanced': True}),
                'lr_planner': ('FLOAT', {'default': 5e-5, 'min': 1e-6, 'max': 1e-2, 'step': 1e-6, 'round': False, 'advanced': True}),
                'lr_decoder': ('FLOAT', {'default': 2e-4, 'min': 1e-6, 'max': 1e-2, 'step': 1e-6, 'round': False, 'advanced': True}),
                'clips_per_step': ('INT', {'default': 8, 'min': 1, 'max': 64, 'advanced': True}),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 2 ** 31 - 1, 'control_after_generate': False, 'advanced': True}),
            },
            'hidden': {'unique_id': 'UNIQUE_ID'},
        }

    def train(self, robin, dataset, name, steps, save_every, sample_text, restart, rank_planner=32, rank_decoder=64, lr_planner=5e-5,
              lr_decoder=2e-4, clips_per_step=8, seed=0, unique_id=None):
        name = name.strip()
        if not NAME.fullmatch(name):
            raise ValueError('Name the LoRA with letters, digits, dashes, dots or underscores.')
        base = Path(folder_paths.get_folder_paths('loras')[0])
        out = base/'robin'/name
        bar = comfy.utils.ProgressBar(steps)
        listen, samples = f'robin/{name}_{uuid.uuid4().hex[:8]}', []

        def progress(stage, step, total, path=None, **info):
            if stage == 'sample':
                samples.append(dict(preview(path, listen), step=step))
                info = dict(sample=samples[-1])
            else:
                bar.update_absolute(step, total)
            tell('robin.train', unique_id, stage=stage, step=step, total=total, **info)

        tell('robin.train', unique_id, stage='start', step=0, total=steps, save_every=save_every)
        if not restart:
            for f in sorted((out/'samples').glob('*.wav')):
                if f.stem == 'base' or SAVED.fullmatch(f.stem):
                    progress('sample', 0 if f.stem == 'base' else int(f.stem[5:]), steps, path=f)
        model = RobinModel(robin.core).ready(speech=True, memory=TRAINING_MEMORY)
        started = time.time()
        with torch.inference_mode(False), torch.enable_grad():
            trainable(model)
            try:
                path = trainer.train(model, dataset['folder'], out, lang=dataset['lang'], restart=restart, log=lambda m: logging.info(f'Robin: {m}'),
                                     stop=mm.processing_interrupted, progress=progress, steps=steps, save_every=save_every,
                                     rank_planner=rank_planner, rank_decoder=rank_decoder, lr_planner=lr_planner, lr_decoder=lr_decoder,
                                     accum=clips_per_step, sample_text=sample_text.strip() or None, seed=seed)
            except SystemExit as e:
                tell('robin.train', unique_id, stage='stopped' if mm.processing_interrupted() else 'failed')
                mm.throw_exception_if_processing_interrupted()
                raise ValueError(str(e)) from None
        if path is None or mm.processing_interrupted():
            tell('robin.train', unique_id, stage='stopped')
            mm.throw_exception_if_processing_interrupted()
            raise ValueError('Training stopped before it finished.')
        done = dict(stage='done', step=steps, total=steps, minutes=(time.time() - started) / 60, samples=samples)
        tell('robin.train', unique_id, **done)
        return {'ui': {'robin_train': [done]}, 'result': (RobinModel(robin.core, (str(path), 1.)), Path(path).relative_to(base).as_posix())}


routes = PromptServer.instance.routes


@routes.get('/robin/status')
async def status_route(request):
    model = request.query.get('model', MODEL)
    return web.json_response(model_status(model if model in model_folders() else MODEL, request.query.get('vae', VAE)))


@routes.get('/robin/uploaded')
async def uploaded_route(request):
    base = (Path(folder_paths.get_input_directory())/UPLOADS).resolve()
    folder = (base/request.query.get('name', '')).resolve()
    if folder.parent != base:
        return web.Response(status=400)
    files = {f.relative_to(folder).as_posix(): f.stat().st_size for f in folder.rglob('*') if f.is_file()} if folder.is_dir() else {}
    return web.json_response(dict(files=files))


@routes.post('/robin/open_recordings')
async def open_recordings_route(request):
    folder = folder_paths.get_input_directory()
    here = request.remote in ('127.0.0.1', '::1', '::ffff:127.0.0.1') and request.host.rsplit(':', 1)[-1] == str(PromptServer.instance.port)
    desktop = sys.platform in ('darwin', 'win32') or os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    if not (here and desktop):
        return web.json_response(dict(opened=False, folder=folder))
    if sys.platform == 'win32':
        os.startfile(folder)
    else:
        subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    return web.json_response(dict(opened=True, folder=folder))


@routes.post('/robin/check')
async def check_route(request):
    body = await request.json()
    language = body.get('language') if body.get('language') in LANGUAGES else 'English'
    _, report = await asyncio.to_thread(check_folder, str(body.get('folder', '')), language)
    return web.json_response(dict(report=report))


NODE_CLASS_MAPPINGS = {
    'RobinLoader': RobinLoader,
    'RobinLoRA': RobinLoRA,
    'RobinSpeak': RobinSpeak,
    'RobinVoiceChanger': RobinVoiceChanger,
    'RobinLoadAudio': RobinLoadAudio,
    'RobinCheckDataset': RobinCheckDataset,
    'RobinTrainLoRA': RobinTrainLoRA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'RobinLoader': 'Robin Loader',
    'RobinLoRA': 'Robin LoRA',
    'RobinSpeak': 'Robin Speak',
    'RobinVoiceChanger': 'Robin Voice Changer',
    'RobinLoadAudio': 'Robin Load Audio',
    'RobinCheckDataset': 'Robin Check Dataset',
    'RobinTrainLoRA': 'Robin Train LoRA',
}
