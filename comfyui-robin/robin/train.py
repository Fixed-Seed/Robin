import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .lora import LoRA, file_sha256
from .pipeline import plain, prefix_mask

AUDIO = ('.wav', '.flac', '.mp3', '.ogg', '.opus', '.aiff', '.aif')
OTHER_AUDIO = ('.m4a', '.aac', '.wma', '.mp4', '.webm', '.mov')
SPEECH_TOKENS = 'speech_tokenizer_v3_25hz'
REF_WHOLE, REF_EXCERPT = 400, 375
SAMPLE_TEXT = dict(en='Hello there! This is a quick test of my new voice. How does it sound to you?',
                   ja='こんにちは。これは新しい声のテストです。どう聞こえますか？',
                   mix='Hello! これは新しい声のテストです。How does it sound?')
CHARS_PER_SECOND = dict(en=((2.5, 35), (6, 24)), ja=((1, 25), (3, 15)), mix=((1, 35), (3, 24)))

PROBLEMS = {
    'no_text': 'no transcript (add a .txt file with the same name)',
    'empty_text': 'the transcript is empty',
    'unreadable': "the audio can't be read",
    'other_format': "this audio format can't be read (convert it to .wav or .flac)",
    'low_rate': 'recorded below 16 kHz (too low to learn a voice from)',
    'silent': 'almost silent',
    'short': 'too short',
    'long': 'too long (split it into shorter clips)',
    'mismatch': "the transcript doesn't fit the length of the audio (check the .txt)",
    'tokens': "the audio couldn't be processed",
}
WARNINGS = {
    'rate': 'recorded below 44.1 kHz: the voice will learn a duller top end',
    'clipped': 'clipped (distorted peaks)',
    'digits': 'digits in the transcript: write numbers out as words, the way they are spoken',
    'pace': 'the transcript may not fit the audio (unusually fast or slow speech for its length)',
    'duplicate': 'the same transcript appears more than once',
    'orphan': 'a transcript with no audio file of the same name',
}


def frames_db(x, rate):
    n = max(1, rate // 100); k = len(x) // n
    if k == 0: return np.full(1, -120.)
    return 20 * np.log10(np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1)) + 1e-9)


def trim(x, rate):
    db = frames_db(x, rate); loud = float(db.max())
    on = np.where(db > max(min(-45., loud - 30), -60.))[0]
    if len(on) == 0: return x[:0], loud
    n = rate // 100
    return x[max(0, on[0] * n - int(.15 * rate)):min(len(x), (on[-1] + 1) * n + int(.25 * rate))], loud


def read_mono(path):
    x, rate = sf.read(str(path), dtype='float32', always_2d=True)
    return x.mean(1), rate


def scan(folder, lang='en', min_seconds=1., max_seconds=30.):
    folder = Path(folder)
    if not folder.is_dir(): raise SystemExit(f'{folder} is not a folder')
    desc_file = folder/'voice.txt'
    description = plain(' '.join(desc_file.read_text(encoding='utf-8').split())) if desc_file.exists() else None
    problems, warnings, clips, seen = defaultdict(list), defaultdict(list), [], defaultdict(list)
    files = sorted(p for p in folder.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(folder).parts))
    hard, soft = CHARS_PER_SECOND.get(lang, CHARS_PER_SECOND['en'])
    for p in files:
        rel = p.relative_to(folder).as_posix(); ext = p.suffix.lower()
        if ext in OTHER_AUDIO: problems['other_format'].append(rel); continue
        if ext not in AUDIO: continue
        txt = p.with_suffix('.txt')
        if not txt.exists(): problems['no_text'].append(rel); continue
        text = plain(' '.join(txt.read_text(encoding='utf-8', errors='replace').split()))
        if not text: problems['empty_text'].append(rel); continue
        try:
            x, rate = read_mono(p)
        except Exception:
            problems['unreadable'].append(rel); continue
        if rate < 16000: problems['low_rate'].append(rel); continue
        y, loud = trim(x, rate); seconds = len(y) / rate
        if loud < -40 or seconds == 0: problems['silent'].append(rel); continue
        if seconds < min_seconds: problems['short'].append(f'{rel} ({seconds:.1f} s)'); continue
        if seconds > max_seconds: problems['long'].append(f'{rel} ({seconds:.0f} s)'); continue
        cps = len(re.sub(r'\[[^\]]*\]|[\W_]', '', text)) / seconds
        if not hard[0] <= cps <= hard[1]: problems['mismatch'].append(rel); continue
        if not soft[0] <= cps <= soft[1]: warnings['pace'].append(rel)
        if rate < 44100: warnings['rate'].append(rel)
        if (np.abs(x) >= .999).mean() > 1e-3: warnings['clipped'].append(rel)
        if re.search(r'\d', re.sub(r'\[[^\]]*\]', '', text)): warnings['digits'].append(rel)
        seen[text.lower()].append(rel)
        st = p.stat()
        key = hashlib.sha1(f'{rel}|{st.st_size}|{st.st_mtime_ns}|{text}|{min_seconds}|{max_seconds}'.encode()).hexdigest()[:20]
        clips.append(dict(file=rel, text=text, seconds=round(seconds, 2), rate=rate, key=key))
    for rels in seen.values():
        if len(rels) > 1: warnings['duplicate'] += rels
    audio = {p.with_suffix('') for p in files if p.suffix.lower() in AUDIO + OTHER_AUDIO}
    warnings['orphan'] += [p.relative_to(folder).as_posix() for p in files
                           if p.suffix.lower() == '.txt' and p != desc_file and p.with_suffix('') not in audio]
    if not warnings['orphan']: del warnings['orphan']
    return dict(clips=clips, problems=problems, warnings=warnings, description=description, folder=folder)


def describe(found, log=print):
    clips, problems, warnings = found['clips'], found['problems'], found['warnings']
    left = sum(len(v) for v in problems.values())
    minutes = sum(c['seconds'] for c in clips) / 60
    log(f"Checked {len(clips) + left} clips in {found['folder']}")
    log(f'  ready: {len(clips)} clips, {minutes:.1f} minutes of speech')
    if left:
        log(f'  left out: {left} clips')
        for kind, rels in problems.items():
            log(f'    {PROBLEMS[kind]}: {len(rels)}')
            for r in rels[:8]: log(f'      {r}')
            if len(rels) > 8: log(f'      ... and {len(rels) - 8} more')
    for kind, rels in warnings.items():
        log(f'  note, {WARNINGS[kind]}: {len(rels)}' + (f' ({", ".join(rels[:3])}{"..." if len(rels) > 3 else ""})' if len(rels) < len(clips) else ''))
    log(f'  voice description (voice.txt): {found["description"]!r}' if found['description'] else
        '  voice description: none (optional: describe the voice in a voice.txt file in the folder)')
    if minutes < 2:
        raise SystemExit(f'Only {minutes:.1f} minutes of usable speech. A LoRA needs at least 2 minutes; 20 to 60 minutes works best.')
    if minutes < 10:
        log('  note: under 10 minutes of speech. It can work, but 20 to 60 minutes gives a much closer voice.')


@torch.no_grad()
def encode(robin, found, cache, log=print, stop=None, progress=None):
    from torchaudio.functional import resample
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    vae_id = file_sha256(robin.vae_file) if getattr(robin, 'vae_file', None) else 'module'
    meta_file = cache/'meta.json'; meta = dict(vae=vae_id, speech_tokens=SPEECH_TOKENS)
    if meta_file.exists() and json.loads(meta_file.read_text()) != meta:
        for f in cache.glob('*.npz'): f.unlink()
    meta_file.write_text(json.dumps(meta))
    out, bad, t0, done = [], [], time.time(), 0
    todo = sum(not (cache/f"{c['key']}.npz").exists() for c in found['clips'])
    if todo: log(f'Preparing {todo} clips...')
    for c in found['clips']:
        f = cache/f"{c['key']}.npz"
        if not f.exists():
            if stop is not None and stop():
                raise SystemExit(f'Stopped while preparing clips ({done} of {todo} done; they are kept for next time).')
            try:
                x, rate = read_mono(found['folder']/c['file'])
                y = torch.from_numpy(trim(x, rate)[0])
                tokens = robin.speech(y.numpy(), rate)
                z = robin.vae.encode(resample(y, rate, 48000)[None].repeat(2, 1)[None].to(robin.device))[0].T.float().cpu().numpy()
                if abs(len(tokens) - len(z)) > 2: raise ValueError('length mismatch')
                k = min(len(tokens), len(z))
                np.savez(f, tokens=tokens[:k].astype(np.int16), latents=z[:k].astype(np.float16))
            except Exception:
                bad.append(c['file']); continue
            done += 1
            if progress is not None: progress('prepare', done, todo)
            if done % 25 == 0 or done == todo: log(f'  {done}/{todo} prepared ({time.time() - t0:.0f} s)')
        d = np.load(f)
        out.append(dict(c, tokens=d['tokens'].astype(np.int64), latents=d['latents']))
    if bad:
        found['problems']['tokens'] += bad
        log(f"  left out {len(bad)} clips: {PROBLEMS['tokens']}: {', '.join(bad[:5])}{'...' if len(bad) > 5 else ''}")
    return out


class Trainer:
    def __init__(self, robin, clips, description, out, *, lang='en', steps=1000, rank_planner=32, rank_decoder=64, lr_planner=5e-5,
                 lr_decoder=2e-4, accum=8, warmup=100, lr_floor=.2, p_reference=.4, save_every=200, sample_text=None, seed=0, log=print, stop=None,
                 progress=None):
        self.robin, self.description, self.out, self.lang, self.log, self.stop = robin, description, Path(out), lang, log, stop
        self.progress = progress
        self.hp = dict(steps=steps, rank_planner=rank_planner, rank_decoder=rank_decoder, lr_planner=lr_planner, lr_decoder=lr_decoder,
                       accum=accum, warmup=min(warmup, max(1, steps // 10)), lr_floor=lr_floor, p_reference=p_reference, save_every=save_every,
                       lang=lang, seed=seed)
        self.sample_text = sample_text or SAMPLE_TEXT.get(lang, SAMPLE_TEXT['en'])
        rng = random.Random(seed); order = list(range(len(clips))); rng.shuffle(order); torch.manual_seed(seed)
        hold = min(10, len(clips) // 20) if len(clips) >= 40 else 0
        self.holdout = [clips[i] for i in order[:hold]]; self.clips = [clips[i] for i in order[hold:]]
        self.short = [c for c in self.clips if len(c['tokens']) <= REF_WHOLE]
        ranks = {k: r for k, r in (('planner', rank_planner), ('decoder', rank_decoder)) if r > 0}
        nets = dict(planner=robin.planner.layers, decoder=robin.decoder.model.layers)
        self.lora = LoRA({k: nets[k] for k in ranks}, ranks, seed=seed).attach()
        groups = [dict(params=[p for n, p in zip(self.lora.names, getattr(self.lora, which)) if n.startswith(net + '.')],
                       lr=dict(planner=lr_planner, decoder=lr_decoder)[net]) for net in ranks for which in ('A', 'B')]
        for g in groups: g['base_lr'] = g['lr']
        self.optimizer = torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=0., eps=1e-8)
        self.rng, self.step, self.norms = rng, 0, []
        self.sel = torch.cat((torch.arange(robin.offset, robin.offset + 32768), torch.tensor([robin.audio_end]))).to(robin.device)
        self.base_sha = robin.base_sha256()

    def lr_scale(self, step):
        hp = self.hp
        if step <= hp['warmup']: return step / hp['warmup']
        f = min(1., (step - hp['warmup']) / max(1, hp['steps'] - hp['warmup']))
        return hp['lr_floor'] + (1 - hp['lr_floor']) * .5 * (1 + math.cos(math.pi * f))

    def example(self, clip, rng):
        mode = 'reference' if len(self.clips) > 1 and rng.random() < self.hp['p_reference'] else 'tts'
        desc = self.description if mode == 'tts' or rng.random() < .5 else None
        if mode == 'tts' and rng.random() < .1: desc = None
        ref = ref_text = None
        if mode == 'reference':
            pool = [c for c in (self.short or self.clips) if c is not clip] or [c for c in self.clips if c is not clip]
            other = rng.choice(pool); tok = other['tokens']
            if len(tok) <= REF_WHOLE:
                ref, ref_text = tok, other['text']
            else:
                s = rng.randrange(len(tok) - REF_EXCERPT + 1); ref = tok[s:s + REF_EXCERPT]
        return self.robin.prompt(clip['text'], desc, self.lang, ref, ref_text)

    def losses(self, ids, clip, train=True, gen=None, sigmas=None):
        r = self.robin; dev = r.device
        seq = torch.tensor(list(ids) + [int(t) + r.offset for t in clip['tokens']] + [r.audio_end], device=dev)
        ss, n = len(ids), len(seq)
        mask = prefix_mask(n, ss, dev)
        h = r.planner(seq[None], mask)[0][ss - 1:-1]
        targets = seq[ss:]
        targets = torch.where(targets == r.audio_end, torch.full_like(targets, 32768), targets - r.offset)
        W = r.planner.lm_head.weight[self.sel]
        ce = h.new_zeros((), dtype=torch.float32)
        for s in range(0, len(targets), 256):
            ce = ce + checkpoint(lambda hs, ys: F.cross_entropy(F.linear(hs, W).float(), ys, reduction='sum'),
                                 h[s:s + 256], targets[s:s + 256], use_reentrant=False)
        ce = ce / len(targets)
        if train: (ce / self.hp['accum']).backward()
        del h
        with torch.no_grad():
            ks, vs, _ = r.planner.prefill(seq, mask)
        clean = torch.from_numpy(clip['latents'].astype(np.float32)).to(dev)
        flows, sigmas = [], sigmas or [min(.98, max(.02, self.rng.betavariate(2, 2))) for _ in range(2)]
        for sigma in sigmas:
            noise = torch.randn(clean.shape, device=dev, generator=gen)
            xt = sigma * noise + (1 - sigma) * clean
            v = r.decoder(xt[None], torch.tensor([sigma], device=dev), ks, vs, checkpointing=train)[0]
            fl = F.mse_loss(v, noise - clean)
            if train: (fl / (len(sigmas) * self.hp['accum'])).backward()
            flows.append(float(fl.detach()))
        return float(ce.detach()), sum(flows) / len(flows)

    @torch.no_grad()
    def evaluate(self):
        if not self.holdout: return None
        gen = torch.Generator(device=self.robin.device).manual_seed(1)
        rows = [self.losses(self.robin.prompt(c['text'], self.description, self.lang), c, train=False, gen=gen, sigmas=(.25, .5, .75))
                for c in self.holdout]
        return dict(planner=round(sum(r[0] for r in rows) / len(rows), 4), decoder=round(sum(r[1] for r in rows) / len(rows), 4))

    @torch.no_grad()
    def sample(self, name):
        if self.robin.vae is None: return
        try:
            wav, sr, _ = self.robin.speak(self.sample_text, self.description, lang=self.lang, seed=0)
            (self.out/'samples').mkdir(exist_ok=True)
            path = self.out/'samples'/f'{name}.wav'
            sf.write(str(path), wav, sr, subtype='PCM_16')
        except Exception as e:
            self.log(f'  (could not render a sample: {e})')
            return
        if self.progress is not None: self.progress('sample', self.step, self.hp['steps'], path=path)

    def save(self, final=False, light=False):
        (self.out/'checkpoints').mkdir(exist_ok=True)
        path = self.out/'checkpoints'/f'step_{self.step:05d}.safetensors'
        self.lora.save(path, self.base_sha)
        if final: self.lora.save(self.out/'lora.safetensors', self.base_sha)
        state = dict(step=self.step, hp=self.hp, lora=self.lora.state_dict(), optimizer=self.optimizer.state_dict(), rng=self.rng.getstate(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None, norms=self.norms)
        tmp = self.out/'resume.pt.tmp'; torch.save(state, tmp); os.replace(tmp, self.out/'resume.pt')
        ev = None if light else self.evaluate()
        if not light: self.sample(f'step_{self.step:05d}')
        self.log(f'  saved {path.relative_to(self.out)}' + (f' | held-out loss: planner {ev["planner"]:.3f}, decoder {ev["decoder"]:.4f}' if ev else ''))
        with (self.out/'log.jsonl').open('a') as f: f.write(json.dumps(dict(step=self.step, saved=path.name, heldout=ev)) + '\n')

    def resume(self):
        state = torch.load(self.out/'resume.pt', map_location='cpu', weights_only=False)
        keep = ('rank_planner', 'rank_decoder', 'lang')
        if any(state['hp'][k] != self.hp[k] for k in keep):
            raise SystemExit(f'{self.out} holds a run with different settings ({", ".join(f"{k}={state["hp"][k]}" for k in keep)}). '
                             'Use the same settings, a new output folder, or start over (--restart).')
        self.lora.load_state_dict(state['lora']); self.optimizer.load_state_dict(state['optimizer'])
        self.rng.setstate(state['rng']); torch.set_rng_state(state['torch_rng'])
        if state.get('cuda_rng') is not None and torch.cuda.is_available(): torch.cuda.set_rng_state(state['cuda_rng'])
        self.step, self.norms = state['step'], state.get('norms', [])

    def run(self):
        hp, r = self.hp, self.robin
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out/'train_config.json').write_text(json.dumps(dict(hp, clips=len(self.clips), held_out=len(self.holdout),
                                                                  description=bool(self.description)), indent=1))
        if self.step == 0:
            self.lora.scale = 0; self.sample('base'); self.lora.scale = 1
        params = [p for p in self.lora.parameters()]
        n_params = sum(p.numel() for p in params)
        self.log(f'Training a LoRA ({n_params / 1e6:.0f}M parameters) on {len(self.clips)} clips'
                 + (f', {len(self.holdout)} held out for checking' if self.holdout else '') + f', {hp["steps"]} steps.')
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        t_start, step_times, run_loss = time.time(), [], []
        while self.step < hp['steps']:
            t0 = time.time(); self.step += 1
            ces, flows = [], []
            for _ in range(hp['accum']):
                for _attempt in range(4):
                    clip = self.rng.choice(self.clips)
                    try:
                        ce, fl = self.losses(self.example(clip, self.rng), clip)
                        ces.append(ce); flows.append(fl); break
                    except torch.cuda.OutOfMemoryError:
                        self.optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); ces, flows = [], []
                        self.log(f"  ran out of GPU memory on {clip['file']} ({clip['seconds']:.0f} s), leaving it out. "
                                 'If this keeps happening, use a smaller --max-seconds.')
                    except Exception as e:
                        self.optimizer.zero_grad(set_to_none=True); ces, flows = [], []
                        self.log(f"  couldn't train on {clip['file']} ({type(e).__name__}: {e}), leaving it out.")
                    self.clips = [c for c in self.clips if c is not clip]
                    self.short = [c for c in self.short if c is not clip]
                    if not self.clips: raise SystemExit('No clips left to train on.')
            for g in self.optimizer.param_groups: g['lr'] = g['base_lr'] * self.lr_scale(self.step)
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            med = sorted(self.norms)[len(self.norms) // 2] if len(self.norms) >= 50 else None
            if not ces or not math.isfinite(norm) or (med is not None and norm > 20 * med):
                self.optimizer.zero_grad(set_to_none=True)
                if ces: self.log(f'  step {self.step}: an unusual update (gradient {norm:.2f}) was skipped')
            else:
                self.norms = (self.norms + [norm])[-200:]
                self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
            step_times.append(time.time() - t0); run_loss.append((np.mean(ces) if ces else float('nan'), np.mean(flows) if flows else float('nan')))
            with (self.out/'log.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=self.step, planner=round(float(run_loss[-1][0]), 4), decoder=round(float(run_loss[-1][1]), 5),
                                        grad=round(norm, 4), seconds=round(step_times[-1], 2))) + '\n')
            pace = float(np.mean(step_times[-20:])); left = (hp['steps'] - self.step) * pace
            pl, dl = np.nanmean([x[0] for x in run_loss[-10:]]), np.nanmean([x[1] for x in run_loss[-10:]])
            if self.progress is not None:
                self.progress('train', self.step, hp['steps'], planner=float(pl), decoder=float(dl), pace=pace, left=left)
            if self.step % 10 == 0 or self.step <= 3 or self.step == hp['steps']:
                mem = f' | GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB' if torch.cuda.is_available() else ''
                self.log(f"step {self.step}/{hp['steps']} | planner loss {pl:.3f} | decoder loss {dl:.4f} | {pace:.1f} s/step | "
                         f"{left / 60:.0f} min left{mem}")
            if self.step % hp['save_every'] == 0 or self.step == hp['steps']:
                self.save(final=self.step == hp['steps'])
            elif self.stop is not None and self.stop():
                self.save(light=True)
            if self.stop is not None and self.stop() and self.step < hp['steps']:
                self.log(f'Stopped at step {self.step}. Train again with the same folder to continue from here.')
                return None
        self.log(f"Done in {(time.time() - t_start) / 60:.0f} min. The LoRA is {self.out/'lora.safetensors'}; every saved step is in "
                 f"{self.out/'checkpoints'} with a sample of each in {self.out/'samples'}.")
        return self.out/'lora.safetensors'


def train(robin, data, out, *, lang='en', min_seconds=1., max_seconds=30., restart=False, log=print, stop=None, progress=None, **settings):
    out = Path(out)
    if restart:
        old = [out/f for f in ('resume.pt', 'lora.safetensors', 'log.jsonl')] + list(out.glob('checkpoints/step_*.safetensors')) + list(out.glob('samples/*.wav'))
        for f in old:
            if f.exists(): f.unlink()
    elif (out/'lora.safetensors').exists() and not (out/'resume.pt').exists():
        raise SystemExit(f'{out} already holds a finished LoRA. Use a new output folder, or start over (--restart) to train again.')
    found = scan(data, lang, min_seconds, max_seconds); describe(found, log)
    clips = encode(robin, found, out/'cache', log, stop, progress)
    robin._speech = None
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    trainer = Trainer(robin, clips, found['description'], out, lang=lang, log=log, stop=stop, progress=progress, **settings)
    try:
        if (out/'resume.pt').exists():
            trainer.resume()
            if trainer.step >= trainer.hp['steps']:
                if not (out/'lora.safetensors').exists(): trainer.lora.save(out/'lora.safetensors', trainer.base_sha)
                log(f'This run already finished ({trainer.step} steps). Start over (--restart) to train again, or give more steps (--steps) to keep going.')
                return out/'lora.safetensors'
            log(f'Continuing the unfinished run from step {trainer.step}.')
        return trainer.run()
    finally:
        trainer.lora.detach()
