import random
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from . import pieces
from . import train as tr
from .pipeline import Cancelled

LANGUAGES = {'English': 'en', 'Japanese': 'ja', 'English + Japanese': 'mix'}
MAX_SOURCE_SECONDS = 45
GROUND, PANEL, LINE, CREAM, RED, RED_FILL, RED_TEXT = '#121212', '#1b1b1c', '#2c2c2d', '#F2EEE6', '#D9302A', '#BF2A24', '#F0564F'
STEP_LINE = re.compile(r'^step (\d+)/(\d+) .*?\| ([\d.]+) s/step \| (\d+) min left(?: \| GPU ([\d.]+) GB)?')
NONE = ''

CSS = f"""
.gradio-container {{ max-width: 1200px !important; margin: 0 auto !important; }}
#robin-head {{ align-items: baseline; border-bottom: 2px solid {RED}; padding: 2px 2px 10px; margin-bottom: 2px; }}
#robin-head h1 {{ margin: 0; font-size: 34px; line-height: 1; letter-spacing: .32em; font-weight: 800; color: {RED_TEXT}; }}
#robin-head p {{ margin: 0; color: {CREAM}; opacity: .7; }}
#adapter {{ border: 1px solid {LINE}; border-radius: 12px; padding: 4px 8px; background: {PANEL}; }}
.take {{ border: 1px solid {RED}88 !important; border-radius: 12px !important; background: {PANEL} !important; }}
.kept {{ outline: 2px solid {RED}; }}
.bar {{ height: 10px; border-radius: 6px; background: {RED}2a; overflow: hidden; }}
.bar i {{ display: block; height: 100%; background: {RED}; }}
.summary table {{ width: 100%; }}
.summary td, .summary th {{ text-align: center !important; }}
"""


def theme():
    red = gr.themes.Color(c50='#fdf0ef', c100='#fadcd9', c200='#f4b6b1', c300='#ec8a82', c400=RED_TEXT, c500=RED, c600=RED_FILL,
                          c700='#9a1f1a', c800='#771814', c900='#55110e', c950='#3a0b09', name='robin')
    look = dict(body_background_fill=GROUND, background_fill_primary=GROUND, background_fill_secondary=PANEL, block_background_fill=PANEL,
                panel_background_fill=PANEL, block_border_color=LINE, border_color_primary=LINE, body_text_color=CREAM,
                body_text_color_subdued='#b8b2a9', block_label_text_color=CREAM, block_title_text_color=CREAM, block_info_text_color='#b8b2a9',
                block_label_background_fill=PANEL, input_background_fill=GROUND, input_border_color='#3a3a3b', input_placeholder_color='#7d7872',
                button_primary_background_fill=RED_FILL, button_primary_background_fill_hover=RED, button_primary_text_color=CREAM,
                button_secondary_background_fill=CREAM, button_secondary_background_fill_hover='#ffffff', button_secondary_text_color=GROUND,
                slider_color=RED, color_accent=RED, color_accent_soft=f'{RED}33', link_text_color=RED_TEXT, checkbox_background_color=GROUND,
                checkbox_background_color_selected=RED_FILL, checkbox_border_color='#3a3a3b')
    base = gr.themes.Base(primary_hue=red, neutral_hue=gr.themes.colors.stone, font=['ui-sans-serif', 'system-ui', 'sans-serif'],
                          font_mono=['ui-monospace', 'monospace'])
    return base.set(**look, **{f'{k}_dark': v for k, v in look.items() if hasattr(base, f'{k}_dark')})


GUIDE = """
### The voice learns what's in the folder.

You bring the recordings and the transcripts. Clean, consistent audio with exact transcripts makes the biggest difference to how a
trained voice sounds.

```
my_voice/
  voice.txt     (optional: the voice in words)
  line_001.wav
  line_001.txt  (what line_001.wav says)
  ...
```

The surest way to exact transcripts: write the script first, then record it line by line. The script is the transcript.

- **How much:** 20 minutes of speech or more. Our test voice used 23 minutes across 227 clips.
- **Each clip:** one or two sentences, 3 to 15 seconds. Anything from 1 to 30 seconds is accepted, so split longer takes.
- **Recording:** one speaker in a quiet, dead-sounding room, with the same mic and distance throughout. No music, effects or other
  voices. Keep peaks around −6 dBFS.
- **Format:** WAV or FLAC at 44.1 or 48 kHz. Lower rates train a duller top end, and clips below 16 kHz are left out.
- **Transcripts:** exactly the words spoken. Write numbers the way they're said ("fifty percent"), and put [laughs] or [coughs] where
  they happen. Re-record a stumble instead of transcribing it.
- **Range:** mix statements, questions and exclamations, short lines and long ones. A calm read trains a calm voice, so record the
  delivery you want back.

**What Check data does**

- **Leaves out:** no transcript or an empty one, an unreadable file or a format it can't read, below 16 kHz, almost silent, under 1 s
  or over 30 s, a transcript far too long or short for its audio.
- **Keeps, with a note:** below 44.1 kHz, clipped peaks, digits in a transcript, a transcript that may not match, the same line twice.
  It also points out transcripts that have no audio file.

Train on voices you have the rights to use.
"""

def read_mono(path):
    x, rate = sf.read(str(path), dtype='float32', always_2d=True)
    return np.ascontiguousarray(x.mean(1)), rate


def slug(text, n=40):
    return re.sub(r'[^a-z0-9]+', '-', text.lower())[:n].strip('-')


def new_seed():
    return random.randrange(2 ** 31)


class Meter:
    def __init__(self, progress, renders, halt, label):
        self.progress, self.total, self.done, self.halt, self.label = progress, max(1, renders), 0, halt, label
        self.expected, self.frac = 1., 0.

    def start(self, seconds):
        self.expected, self.frac = max(1., seconds * 25), 0.
        self.show()

    def finish(self, extra=0):
        self.done += 1; self.total += extra; self.frac = 0.
        self.show()

    def __call__(self, stage, step, total):
        if self.halt.is_set(): return False
        self.frac = .8 * min(1., step / self.expected) if stage == 'plan' else .8 + .2 * step / total
        if stage == 'decode' or step % 10 == 0: self.show()

    def show(self):
        self.progress(min(1., (self.done + self.frac) / self.total), desc=self.label)


class Studio:
    def __init__(self, robin, out):
        self.robin, self.out = robin, Path(out).expanduser().resolve()
        for d in ('saved', 'voices', '.listen'): (self.out/d).mkdir(parents=True, exist_ok=True)
        self.lock, self.halt = threading.Lock(), threading.Event()
        self.job, self.adapter = None, None

    def training(self):
        return self.job is not None and self.job['thread'].is_alive()

    def claim(self):
        if self.training(): raise gr.Error('A voice is training. Robin speaks again when the training finishes or you stop it.')
        if not self.lock.acquire(blocking=False): raise gr.Error('Robin is busy with another request. Try again in a moment.')
        self.halt.clear()

    def stop(self):
        self.halt.set()

    def saved(self):
        return [str(p) for p in sorted((self.out/'saved').glob('*.flac'), key=lambda p: p.stat().st_mtime, reverse=True)[:60]]

    def listen(self, x):
        d = self.out/'.listen'
        for old in sorted(d.glob('*.flac'), key=lambda p: p.stat().st_mtime)[:-40]: old.unlink(missing_ok=True)
        f = d/f'{datetime.now():%Y%m%d-%H%M%S-%f}.flac'
        sf.write(str(f), np.clip(x, -1, 1), pieces.RATE, subtype='PCM_24')
        return str(f)

    def save(self, take):
        if not take: raise gr.Error('There is no take to save yet.')
        name = f"{datetime.now():%Y%m%d-%H%M%S}_{slug(take['text']) or 'take'}_seed{take['seed']}.flac"
        sf.write(str(self.out/'saved'/name), take['audio'], pieces.RATE, subtype='PCM_24')
        gr.Info(f'Saved {name}')
        return self.saved()

    def label(self, path):
        path = Path(path)
        try:
            run = path.parent.parent if path.parent.name == 'checkpoints' else path.parent
            name = run.relative_to(self.out/'voices').as_posix()
            return f"{name} · {'final' if path.name == 'lora.safetensors' else 'step ' + str(int(path.stem.split('_')[1]))}"
        except (ValueError, IndexError):
            return path.name

    def adapters(self):
        found = []
        for run in sorted(p for p in (self.out/'voices').iterdir() if p.is_dir()):
            if (run/'lora.safetensors').exists(): found.append(run/'lora.safetensors')
            found += sorted((run/'checkpoints').glob('step_*.safetensors'), reverse=True)
        return [('None: Robin as it comes', NONE)] + [(self.label(p), str(p)) for p in found]

    def adapter_status(self):
        if self.adapter is None: return 'No voice adapter: Robin as it comes.'
        path, scale = self.adapter
        return f'Using **{self.label(path)}** at strength {scale:g}, in every tab.'

    def apply(self, choice, scale):
        choice = str(choice or '').strip()
        self.claim()
        try:
            if choice == NONE:
                self.robin.remove_lora(); self.adapter = None
            else:
                path = Path(choice).expanduser()
                if not path.is_file(): raise gr.Error(f'No voice adapter file at {path}')
                try:
                    self.robin.load_lora(path, scale)
                except ValueError as e:
                    raise gr.Error(str(e))
                self.adapter = (str(path), scale)
        finally:
            self.lock.release()
        return self.adapter_status()

    def strength(self, scale):
        if self.adapter is not None:
            self.robin.set_lora_scale(scale); self.adapter = (self.adapter[0], scale)
        return self.adapter_status()

    def takes(self, which, text, voice, seed, cfg, temperature, top_k, top_p, language, state, progress):
        lang = LANGUAGES.get(language, 'en')
        text = (text or '').strip()
        if not text: raise gr.Error('Type something for Robin to say.')
        state = dict(state or {}); kept = state.get('kept', set())
        if which == 'both':
            roll = [k for k in 'AB' if k not in kept]
            if not roll: raise gr.Error('Both takes are kept. Un-keep one to roll it again.')
            seeds = {k: int(seed) + i for i, k in enumerate(roll)}
        else:
            seeds = {which: new_seed()}
        meter = Meter(progress, len(pieces.split(text, 35, lang)) * len(seeds), self.halt, 'Robin is speaking')
        self.claim()
        try:
            for k, s in seeds.items():
                audio = pieces.read(self.robin, text, voice, seed=s, cfg=cfg, lang=lang, progress=meter, temperature=temperature, top_k=top_k,
                                    top_p=top_p)
                state[k] = dict(seed=s, audio=audio, text=text)
        except Cancelled:
            raise gr.Error('Stopped.')
        finally:
            self.lock.release()
        out = []
        for k in 'AB':
            if k in seeds: out += [self.listen(state[k]['audio']), f"seed {state[k]['seed']} · {len(state[k]['audio']) / pieces.RATE:.1f} s"]
            else: out += [gr.skip(), gr.skip()]
        return out + [state]

    def keep(self, which, state):
        state = dict(state or {}); kept = set(state.get('kept', set()))
        if which not in state: raise gr.Error(f'There is no take {which} yet.')
        kept ^= {which}; state['kept'] = kept
        on = which in kept
        return gr.update(value=f'Kept {which}' if on else f'Keep {which}', variant='primary' if on else 'secondary'), state

    def change(self, source, source_text, voice, seed, temperature, top_k, top_p, language, progress):
        if source is None: raise gr.Error('Add the recording to change.')
        if not (source_text or '').strip(): raise gr.Error('Type exactly what the recording says.')
        if not (voice or '').strip() and self.adapter is None: raise gr.Error('Describe the new voice, or use a voice adapter.')
        x, rate = read_mono(source)
        if len(x) / rate > MAX_SOURCE_SECONDS:
            raise gr.Error(f'The recording is {len(x) / rate:.0f} s long; the voice changer takes up to {MAX_SOURCE_SECONDS} s at a time.')
        meter = Meter(progress, 2, self.halt, 'Robin is changing the voice')
        self.claim()
        try:
            audio = pieces.change(self.robin, x, rate, source_text, voice=voice, seed=int(seed), lang=LANGUAGES.get(language, 'en'), progress=meter,
                                  temperature=temperature, top_k=top_k, top_p=top_p)
        except Cancelled:
            raise gr.Error('Stopped.')
        finally:
            self.lock.release()
        return self.listen(audio), f'seed {int(seed)} · {len(audio) / pieces.RATE:.1f} s', dict(seed=int(seed), audio=audio, text=source_text)

    def check(self, data, language):
        data = Path(str(data or '').strip()).expanduser()
        if not str(data).strip() or not data.is_dir(): raise gr.Error(f'{data} is not a folder.')
        found = tr.scan(data, LANGUAGES.get(language, 'en'))
        lines = []
        try:
            tr.describe(found, lines.append)
        except SystemExit as e:
            lines.append(str(e))
        left = sum(len(v) for v in found['problems'].values())
        notes = sum(len(v) for v in found['warnings'].values())
        minutes = sum(c['seconds'] for c in found['clips']) / 60
        table = ('| Ready clips | Minutes of speech | Left out | Notes |\n|---|---|---|---|\n'
                 f"| {len(found['clips'])} | {minutes:.1f} | {left} | {notes} |")
        return table, '\n'.join(lines[1:]), slug(data.name) or 'my-voice'

    def where(self, name):
        name = slug(str(name or ''))
        return f'Saves to `{self.out/"voices"/name}`' if name else 'Give the voice a name.'

    def start(self, data, name, steps, save_every, sample_text, restart, language):
        data, name = Path(str(data or '').strip()).expanduser(), slug(str(name or ''))
        if not data.is_dir(): raise gr.Error(f'{data} is not a folder.')
        if not name: raise gr.Error('Give the voice a name.')
        if self.training(): raise gr.Error('A voice is already training.')
        if self.lock.locked(): raise gr.Error('Robin is busy with another request. Try again in a moment.')
        job = dict(data=data, out=self.out/'voices'/name, steps=int(steps), save_every=int(save_every),
                   sample_text=(sample_text or '').strip() or None, restart=bool(restart), lang=LANGUAGES.get(language, 'en'),
                   stop=threading.Event(), log=[], result=None,
                   error=None, ended=None)
        job['thread'] = threading.Thread(target=self._train, args=(job,), daemon=True)
        self.job = job; job['thread'].start()
        gr.Info('Training started.')

    def _train(self, job):
        log = job['log'].append
        try:
            with self.lock:
                if self.robin.lora is not None:
                    self.robin.remove_lora(); self.adapter = None
                    log('The voice adapter in use was taken off for training.')
                job['result'] = tr.train(self.robin, job['data'], job['out'], lang=job['lang'], restart=job['restart'], log=log, stop=job['stop'].is_set,
                                         steps=job['steps'], save_every=job['save_every'], sample_text=job['sample_text'])
        except SystemExit as e:
            log(str(e))
        except Exception as e:
            job['error'] = f'{type(e).__name__}: {e}'
            log(f"Training stopped with an error: {job['error']}"); log(traceback.format_exc(limit=4))
        finally:
            job['ended'] = time.time()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def stop_training(self):
        if not self.training(): raise gr.Error('Nothing is training.')
        self.job['stop'].set()
        gr.Info('Stopping after the current step...')

    def saves(self, out):
        rows = []
        if (out/'samples'/'base.wav').exists(): rows.append(('before training', str(out/'samples'/'base.wav')))
        rows += [(f"step {int(p.stem.split('_')[1])}", str(p)) for p in sorted((out/'samples').glob('step_*.wav'))]
        return rows

    def poll(self, name, shown):
        job = self.job
        out = job['out'] if job else self.out/'voices'/(slug(str(name or '')) or '_')
        if job is None:
            status, bar, log = 'Nothing is training.', '', ''
        else:
            m = next((STEP_LINE.match(l) for l in reversed(job['log']) if STEP_LINE.match(l)), None)
            alive = job['thread'].is_alive()
            if alive and m:
                step, total, pace, left, gb = m.groups()
                status = (f"Training **{out.name}**: step {step} of {total} · {float(pace):.1f} s/step · about {left} min left"
                          + (f' · {gb} GB of GPU memory' if gb else ''))
            elif alive: status = f'Training **{out.name}**: preparing the clips...'
            elif job['error']: status = f"Training stopped with an error: {job['error']}"
            elif job['result']: status = f"Finished **{out.name}**. Pick a step below and use it."
            else: status = 'Stopped. Start again with the same name to continue where it left off.'
            if alive and job['stop'].is_set(): status += ' (stopping after this step)'
            pct = 100 * int(m.group(1)) / int(m.group(2)) if m else 0
            bar = f'<div class="bar"><i style="width:{pct:.1f}%"></i></div>' if m else ''
            log = '\n'.join(job['log'][-300:])
        rows = self.saves(out)
        paths = [r[1] for r in rows]
        choices = gr.update(choices=rows, value=paths[-1] if paths else None) if paths != (shown or []) else gr.skip()
        return status, bar, log, choices, paths

    def use(self, sample, scale):
        if not sample: raise gr.Error('Pick a saved step first.')
        p = Path(sample)
        if p.stem == 'base': raise gr.Error('That is Robin before training. Pick a saved step.')
        lora = p.parent.parent/'checkpoints'/f'{p.stem}.safetensors'
        if not lora.exists(): raise gr.Error(f'No voice adapter was saved for {p.stem}.')
        status = self.apply(str(lora), scale)
        gr.Info(f'Using {self.label(lora)}')
        return status, gr.update(choices=self.adapters(), value=str(lora))


def knobs(cfg=True):
    with gr.Row():
        g = gr.Slider(1, 10, 1, step=.1, label='Guidance', info='Above 1 follows the voice description more closely; try 3.', visible=cfg)
        temperature = gr.Slider(0, 2, .9, step=.01, label='Temperature', info='Higher is livelier and less predictable.')
        top_k = gr.Slider(0, 500, 25, step=1, label='Top-k', info='Choose among only this many likeliest sounds (0 = no limit).')
        top_p = gr.Slider(.01, 1, .8, step=.01, label='Top-p', info='Choose among the likeliest sounds that add up to this share.')
    with gr.Row():
        seed = gr.Number(0, label='Seed', precision=0, minimum=0, scale=3)
        reseed = gr.Button('New seed', scale=1)
    reseed.click(new_seed, None, seed)
    return g, temperature, top_k, top_p, seed


def speak_tab(studio, language):
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label='What to say', lines=6, info='Any length. [laughs] and [coughs] can go anywhere in the text.',
                              value="Okay, okay, [laughs] I really didn't think you'd do it! [coughs] Sorry. Anyway, that was brilliant.")
            with gr.Row():
                laugh = gr.Button('+ [laughs]', size='sm')
                cough = gr.Button('+ [coughs]', size='sm')
        voice = gr.Textbox(label='The voice', lines=8, scale=2,
                           info='Describe it in plain words: age, pitch, pace, mood, how close to the mic. Leave it empty to let Robin choose.',
                           value='A cheerful woman in her thirties with a clear, medium-pitched voice, chatting casually with a friend.')
    cfg, temperature, top_k, top_p, seed = knobs()
    with gr.Row():
        go = gr.Button('Speak: two takes', variant='primary', scale=4)
        stop = gr.Button('Stop', scale=1)
    takes, state = {}, gr.State({})
    with gr.Row(equal_height=True):
        for k in 'AB':
            with gr.Column(elem_classes='take'):
                audio = gr.Audio(label=f'Take {k}', type='numpy', interactive=False, buttons=['download'])
                info = gr.Markdown('')
                with gr.Row():
                    keep = gr.Button(f'Keep {k}', size='sm')
                    redo = gr.Button(f'Redo {k}', size='sm')
                    save = gr.Button(f'Save {k}', size='sm')
                takes[k] = (audio, info, keep, redo, save)
    saved = gr.Files(label='Saved takes', value=studio.saved, interactive=False)
    inputs = [text, voice, seed, cfg, temperature, top_k, top_p, language, state]
    outputs = [takes['A'][0], takes['A'][1], takes['B'][0], takes['B'][1], state]
    go.click(lambda *a, progress=gr.Progress(): studio.takes('both', *a, progress), inputs, outputs)
    for k in 'AB':
        takes[k][2].click(lambda st, k=k: studio.keep(k, st), state, [takes[k][2], state])
        takes[k][3].click(lambda *a, k=k, progress=gr.Progress(): studio.takes(k, *a, progress), inputs, outputs)
        takes[k][4].click(lambda st, k=k: studio.save(st.get(k)), state, saved)
    laugh.click(lambda t: (t or '').rstrip() + ' [laughs] ', text, text)
    cough.click(lambda t: (t or '').rstrip() + ' [coughs] ', text, text)
    stop.click(studio.stop)


def changer_tab(studio, language):
    gr.Markdown(f'Performs a recording again in another voice: the same words, timing and delivery, up to {MAX_SOURCE_SECONDS} seconds at '
                'a time. The new voice comes from the description, and from the voice adapter when one is in use.')
    with gr.Row():
        with gr.Column():
            source = gr.Audio(sources=['upload', 'microphone'], type='filepath', label='The recording to change')
            source_text = gr.Textbox(label='Exactly what it says', lines=3)
        voice = gr.Textbox(label='The new voice', lines=6, info='Describe it in plain words. It can stay empty when a voice adapter is in use.',
                           value='A soft-spoken young man with a calm, low voice.')
    _, temperature, top_k, top_p, seed = knobs(cfg=False)
    with gr.Row():
        go = gr.Button('Change the voice', variant='primary', scale=4)
        stop = gr.Button('Stop', scale=1)
    with gr.Column(elem_classes='take'):
        result = gr.Audio(label='Changed', type='numpy', interactive=False, buttons=['download'])
        info = gr.Markdown('')
        save = gr.Button('Save', size='sm')
    take = gr.State(None)
    saved = gr.Files(label='Saved takes', value=studio.saved, interactive=False)
    go.click(lambda *a, progress=gr.Progress(): studio.change(*a, progress), [source, source_text, voice, seed, temperature, top_k, top_p, language],
             [result, info, take])
    save.click(studio.save, take, saved)
    stop.click(studio.stop)


def train_tab(studio, adapter, strength, status_line, language):
    gr.Markdown('Teach Robin a voice from recordings: a folder of clips, each with a `.txt` file of the same name holding exactly what it '
                'says. Every few hundred steps it saves the voice with a sample, so you can pick a step by ear.')
    with gr.Accordion('Dataset guide', open=False):
        gr.Markdown(GUIDE)
    with gr.Row():
        data = gr.Textbox(label='Folder of clips', scale=4)
        check = gr.Button('Check data', scale=1)
    summary = gr.Markdown('', elem_classes='summary')
    notes = gr.Textbox(label='Notes', lines=6, interactive=False)
    with gr.Row():
        name = gr.Textbox(label='Name', scale=2)
        steps = gr.Number(1000, label='Steps', precision=0, minimum=10)
        save_every = gr.Number(200, label='Save every', precision=0, minimum=10)
    where = gr.Markdown('')
    with gr.Row():
        sample_text = gr.Textbox(tr.SAMPLE_TEXT['en'], label='Sample line read at every save', scale=4)
        restart = gr.Checkbox(False, label='Start over (replaces an earlier run with this name)')
    with gr.Row():
        start = gr.Button('Start', variant='primary', scale=4)
        halt = gr.Button('Stop', scale=1)
    status = gr.Markdown('Nothing is training.')
    bar = gr.HTML('')
    with gr.Accordion('Training log', open=False):
        log = gr.Textbox(show_label=False, lines=12, max_lines=12, interactive=False, autoscroll=True)
    with gr.Row(equal_height=True):
        sample = gr.Dropdown([], label='Saves', scale=2)
        player = gr.Audio(label='Sample', type='filepath', interactive=False, scale=3)
        use = gr.Button('Use this step', variant='primary', scale=1)
    shown = gr.State([])
    check.click(studio.check, [data, language], [summary, notes, name]).then(studio.where, name, where)
    name.change(studio.where, name, where)
    start.click(studio.start, [data, name, steps, save_every, sample_text, restart, language])
    language.change(lambda l: tr.SAMPLE_TEXT[LANGUAGES.get(l, 'en')], language, sample_text)
    halt.click(studio.stop_training)
    sample.change(lambda p: p or None, sample, player)
    use.click(studio.use, [sample, strength], [status_line, adapter])
    gr.Timer(2).tick(studio.poll, [name, shown], [status, bar, log, sample, shown])


def build(studio):
    with gr.Blocks(title='Robin', analytics_enabled=False) as demo:
        with gr.Row(elem_id='robin-head'):
            gr.HTML('<h1>ROBIN</h1>')
            gr.HTML('<p>Speak, change voices, train voices.</p>')
        with gr.Row(elem_id='adapter', equal_height=True):
            adapter = gr.Dropdown(studio.adapters(), value=NONE, label='Voice adapter', allow_custom_value=True, scale=3,
                                  info='A voice trained in "Train a voice", or the path to one.')
            strength = gr.Slider(0, 2, 1, step=.05, label='Strength', info='1 is the voice as trained.', scale=2)
            language = gr.Dropdown(list(LANGUAGES), value='English', label='Language', scale=1,
                                   info='The language Robin speaks, on every tab.')
            with gr.Column(scale=1, min_width=120):
                apply = gr.Button('Use', variant='primary', size='sm')
                refresh = gr.Button('Refresh list', size='sm')
        status_line = gr.Markdown(studio.adapter_status)
        demo.load(lambda: (gr.update(choices=studio.adapters(), value=studio.adapter[0] if studio.adapter else NONE),
                           studio.adapter[1] if studio.adapter else 1, studio.adapter_status()), None, [adapter, strength, status_line])
        apply.click(studio.apply, [adapter, strength], status_line)
        refresh.click(lambda: gr.update(choices=studio.adapters()), None, adapter)
        strength.change(studio.strength, strength, status_line)
        with gr.Tabs():
            with gr.Tab('Speak'):
                speak_tab(studio, language)
            with gr.Tab('Voice changer'):
                changer_tab(studio, language)
            with gr.Tab('Train a voice'):
                train_tab(studio, adapter, strength, status_line, language)
    return demo


def launch(robin, out='robin_studio', host='127.0.0.1', port=7860):
    studio = Studio(robin, out)
    demo = build(studio)
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=host, server_port=port, theme=theme(), css=CSS, allowed_paths=[str(studio.out)], show_error=True)
