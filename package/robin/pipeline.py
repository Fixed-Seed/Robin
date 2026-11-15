import json
import warnings
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from .model import Config, Planner, Decoder

LANGS = dict(en='English', ja='Japanese', mix='English and Japanese')
WEIGHTS = 'robin_v1.safetensors'
VAE = 'robin_vae.safetensors'
SPEECH = 'speech_tokenizer_v3_25hz.onnx'
APOSTROPHES = str.maketrans(dict.fromkeys('\u2019\u2018\u201b\u02bc\uff07\u2032\u00b4`', "'"))


def plain(s):
    return s.translate(APOSTROPHES) if s else s
FRAME_RATE, SAMPLE_RATE = 25, 48000


class Cancelled(Exception):
    pass


def prefix_mask(n, ss, device):
    m = torch.full((n, n), float('-inf'), device=device).triu_(1)
    m[:ss, :ss] = 0
    return m


class Robin:
    def __init__(self, planner, decoder, vae, tokenizer, config, device):
        self.planner, self.decoder, self.vae, self.tok, self.device = planner, decoder, vae, tokenizer, device
        t = config['tokens']
        self.eod, self.score_start, self.score_end = t['eod'], t['score_start'], t['score_end']
        self.audio_start, self.audio_end, self.offset, self.codes = t['audio_start'], t['audio_end'], t['audio_offset'], t['audio_codes']
        self.instruction, self.defaults = config['instruction'], config['sampling']
        self._speech, self.lora, self.weights_file, self.vae_file = None, None, None, None

    @classmethod
    def from_pretrained(cls, path, vae=None, device='cuda', dtype=torch.bfloat16, lora=None, lora_scale=1.0):
        path = Path(path)
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(str(path)))
        if vae is None:
            vae = path/VAE
            if not vae.is_file():
                raise ValueError(f'VAE weights are required: put {VAE} in {path}, or pass vae=<path to the .safetensors file>')
        config = json.loads((path/'config.json').read_text())
        c = Config(**config['model'])
        with torch.device('meta'):
            planner, decoder = Planner(c), Decoder(c)
        sd = load_file(str(path/WEIGHTS), device=str(torch.device(device)))
        for net, prefix in ((planner, 'robin.planner.'), (decoder, 'robin.decoder.')):
            net.load_state_dict({k[len(prefix):]: sd.pop(k) for k in [k for k in sd if k.startswith(prefix)]}, strict=True, assign=True)
        if sd: raise ValueError(f'unexpected weights in {path/WEIGHTS}: {", ".join(list(sd)[:5])}')
        del sd
        vae_file = None
        if not isinstance(vae, torch.nn.Module):
            from .vae import load_vae
            vae_file, vae = Path(vae), load_vae(vae, device)
        planner = planner.to(device, dtype).eval().requires_grad_(False); decoder = decoder.to(device, dtype).eval().requires_grad_(False)
        robin = cls(planner, decoder, vae, Tokenizer.from_file(str(path/'tokenizer.json')), config, device)
        robin.weights_file, robin.vae_file = path/WEIGHTS, vae_file
        if lora is not None:
            robin.load_lora(lora, lora_scale)
        return robin

    def base_sha256(self):
        from .lora import file_sha256
        return file_sha256(self.weights_file) if self.weights_file else None

    def load_lora(self, path, scale=1.0):
        from .lora import LoRA
        self.remove_lora()
        lora, config = LoRA.load(path, dict(planner=self.planner.layers, decoder=self.decoder.model.layers), self.device)
        base = config.get('base_sha256')
        if base and self.weights_file and base != self.base_sha256():
            warnings.warn(f'{Path(path).name} was trained on different Robin weights; it may not sound right')
        lora.scale = scale
        self.lora = lora.attach()
        return self

    def set_lora_scale(self, scale):
        if self.lora is None: raise ValueError('no LoRA is loaded')
        self.lora.scale = scale

    def remove_lora(self):
        if self.lora is not None:
            self.lora.detach(); self.lora = None

    @property
    def speech(self):
        if self._speech is None:
            from .speech import SpeechTokenizer
            self._speech = SpeechTokenizer(self.weights_file.parent/SPEECH, self.device)
        return self._speech

    def tags(self, description=None, lang='en'):
        base = f"speech, spoken word, {LANGS.get(lang, lang)}"
        return f"{base}. {description.strip()}" if description and description.strip() else base

    def _block(self, label, tokens, text=None):
        head = f'[{label}]\n' + (f'{text}\n' if text else '')
        return self.tok.encode(head).ids + [self.audio_start] + [int(t) + self.offset for t in tokens] + [self.audio_end] + self.tok.encode('\n').ids

    def prompt(self, text, description=None, lang='en', reference=None, reference_text=None, source=None):
        text, description, reference_text = plain(text), plain(description), plain(reference_text)
        ids = [self.eod] + self.tok.encode(f"{self.instruction}\n[Tags]\n{self.tags(description, lang)}\n[Lyrics]\n{text}\n").ids
        if reference is not None:
            ids += self._block('Reference voice', reference, reference_text)
        if source is not None:
            ids += self._block('Source performance', source)
        return ids + [self.score_start, self.score_end, self.audio_start]

    def reference_tokens(self, audio, text=None, sample_rate=None, max_frames=400, excerpt_frames=375):
        tokens = self.speech(audio, sample_rate)
        if len(tokens) <= max_frames:
            return tokens, text
        s = (len(tokens) - excerpt_frames) // 2
        return tokens[s:s + excerpt_frames], None

    def _branch(self, ids, budget):
        ks, vs, h = self.planner.prefill(torch.tensor(ids, device=self.device), prefix_mask(len(ids), len(ids), self.device))
        cap = len(ids) + budget + 1
        kb, vb = [], []
        for k, v in zip(ks, vs):
            kbuf = torch.zeros(1, k.shape[1], cap, k.shape[3], dtype=k.dtype, device=self.device); kbuf[:, :, :len(ids)] = k
            vbuf = torch.zeros_like(kbuf); vbuf[:, :, :len(ids)] = v
            kb.append(kbuf); vb.append(vbuf)
        return dict(k=kb, v=vb, pos=len(ids), logits=self.planner.lm_head(h[:, -1]).float()[0])

    @torch.no_grad()
    def plan(self, ids, budget, *, seed=0, temperature=None, top_k=None, top_p=None, min_tokens=25, negative=None, cfg=1.,
             pause_guard=None, guard_distinct=None, callback=None):
        d = self.defaults
        temperature = d['temperature'] if temperature is None else temperature
        top_k = d['top_k'] if top_k is None else top_k
        top_p = d['top_p'] if top_p is None else top_p
        pause_guard = d['pause_guard'] if pause_guard is None else pause_guard
        guard_distinct = d['guard_distinct'] if guard_distinct is None else guard_distinct
        g = torch.Generator(device=self.device).manual_seed(int(seed) % 2 ** 64)
        branches = [self._branch(ids, budget)] + ([self._branch(negative, budget)] if negative is not None and cfg != 1. else [])
        allowed = torch.full_like(branches[0]['logits'], float('-inf'))
        allowed[self.offset:self.offset + self.codes] = 0; allowed[self.audio_end] = 0
        out = []
        for step in range(budget):
            lg = branches[0]['logits'] if len(branches) == 1 else branches[1]['logits'] + cfg * (branches[0]['logits'] - branches[1]['logits'])
            s = lg + allowed
            if step < min_tokens: s[self.audio_end] = float('-inf')
            if pause_guard and len(out) >= pause_guard:
                recent = set(out[-pause_guard:])
                if len(recent) <= guard_distinct: s[torch.tensor(sorted(recent), device=self.device)] = float('-inf')
            if temperature == 0:
                tok = int(s.argmax())
            else:
                s = s / temperature
                if top_k: s = s.masked_fill(s < s.topk(min(top_k, s.numel())).values[-1], float('-inf'))
                if top_p < 1:
                    v, ix = s.sort(descending=True); cp = v.softmax(-1).cumsum(-1)
                    drop = cp - v.softmax(-1) > top_p; s = s.masked_fill(torch.zeros_like(s, dtype=torch.bool).scatter(0, ix, drop), float('-inf'))
                tok = int(torch.multinomial(s.softmax(-1), 1, generator=g))
            if tok == self.audio_end: return out, False
            out.append(tok)
            if callback is not None and callback('plan', step + 1, budget) is False: raise Cancelled()
            for b in branches:
                b['logits'] = self.planner.step(tok, b['pos'], b['k'], b['v']); b['pos'] += 1
        return out, True

    @torch.no_grad()
    def latents(self, ids, tokens, *, seed=0, steps=None, sampler=None, callback=None):
        steps = self.defaults['decoder_steps'] if steps is None else steps
        sampler = self.defaults['decoder_sampler'] if sampler is None else sampler
        full = list(ids) + list(tokens) + [self.audio_end]
        ks, vs, _ = self.planner.prefill(torch.tensor(full, device=self.device), prefix_mask(len(full), len(ids), self.device))
        T = len(tokens)
        x = torch.randn((1, 64, T), generator=torch.Generator().manual_seed(int(seed) % 2 ** 64)).to(self.device).transpose(1, 2)
        sig = torch.cat([torch.linspace(1.0, 0.001, steps + 1)[:-1], torch.zeros(1)]).to(self.device)

        def denoised(x_, s_):
            return x_ - self.decoder.velocity(x_, s_.reshape(1), ks, vs) * s_

        if sampler == 'dpm_2':
            for i in range(steps):
                s0, s1 = sig[i], sig[i + 1]
                d = (x - denoised(x, s0)) / s0
                if s1 == 0:
                    x = x + d * (s1 - s0)
                else:
                    sm = s0.log().lerp(s1.log(), 0.5).exp()
                    x2 = x + d * (sm - s0)
                    x = x + (x2 - denoised(x2, sm)) / sm * (s1 - s0)
                if callback is not None and callback('decode', i + 1, steps) is False: raise Cancelled()
        elif sampler == 'dpmpp_2m':
            old = None
            for i in range(steps):
                s0, s1 = sig[i], sig[i + 1]
                den = denoised(x, s0)
                t, tn = -s0.log(), -s1.log()
                h = tn - t
                if old is None or s1 == 0:
                    x = (s1 / s0) * x - (-h).expm1() * den
                else:
                    r = (t + sig[i - 1].log()) / h
                    x = (s1 / s0) * x - (-h).expm1() * ((1 + 1 / (2 * r)) * den - (1 / (2 * r)) * old)
                old = den
                if callback is not None and callback('decode', i + 1, steps) is False: raise Cancelled()
        else:
            raise ValueError(f'unknown sampler {sampler}')
        return x.transpose(1, 2).float()

    @torch.no_grad()
    def decode(self, latents):
        return self.vae.decode(latents.to(self.device, torch.float32))[0].float().mean(0).cpu().numpy()

    def _budget(self, text, lang, max_seconds):
        if lang == 'mix': frames = sum(5.0 if ord(c) > 127 else 2.2 for c in text)
        else: frames = len(text) * (5.0 if lang == 'ja' else 2.2)
        return int(min(max_seconds * FRAME_RATE, frames + 100))

    def _render(self, ids, budget, seed, negative=None, cfg=1., steps=None, sampler=None, callback=None, **sampling):
        tokens, capped = self.plan(ids, budget, seed=seed, negative=negative, cfg=cfg, callback=callback, **sampling)
        if not tokens: raise RuntimeError('nothing was generated')
        lat = self.latents(ids, tokens, seed=seed, steps=steps, sampler=sampler, callback=callback)
        return self.decode(lat), SAMPLE_RATE, dict(frames=len(tokens), seconds=len(tokens) / FRAME_RATE, capped=capped,
                                                   tokens=[t - self.offset for t in tokens])

    @torch.no_grad()
    def speak(self, text, description=None, *, reference=None, reference_text=None, reference_rate=None, lang='en', seed=0, cfg=1.,
              max_seconds=45, steps=None, sampler=None, callback=None, **sampling):
        ref = None
        if reference is not None:
            ref, reference_text = self.reference_tokens(reference, reference_text, reference_rate)
        ids = self.prompt(text, description, lang, ref, reference_text)
        negative = self.prompt(text, None, lang) if description and cfg != 1. and ref is None else None
        return self._render(ids, self._budget(text, lang, max_seconds), seed, negative, cfg, steps, sampler, callback, **sampling)

    @torch.no_grad()
    def convert(self, source, text, reference, *, reference_text=None, source_rate=None, reference_rate=None, description=None,
                lang='en', seed=0, steps=None, sampler=None, callback=None, **sampling):
        src = self.speech(source, source_rate)
        ref, reference_text = self.reference_tokens(reference, reference_text, reference_rate)
        ids = self.prompt(text, description, lang, ref, reference_text, src)
        return self._render(ids, int(len(src) * 1.6) + 50, seed, None, 1., steps, sampler, callback, **sampling)
