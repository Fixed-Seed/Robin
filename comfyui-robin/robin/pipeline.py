import json
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
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


# ---------------------------------------------------------------- speed
# Planning makes one token at a time, and a step on its own is too small to keep the GPU busy: launching
# its few hundred kernels from Python takes longer than running them. So on CUDA a step is recorded once
# as a CUDA graph over a fixed-size KV cache and replayed (same numbers, ~3x faster), and a LoRA is merged
# into the weights it targets instead of running as hooks around every layer (~2x faster again).
# ROBIN_CUDA_GRAPHS=0 or ROBIN_MERGE_LORA=0 turns either off.
GRAPHS = os.environ.get('ROBIN_CUDA_GRAPHS', '1') != '0'
MERGE_LORA = os.environ.get('ROBIN_MERGE_LORA', '1') != '0'
LANE_BLOCK, LANES_KEPT = 512, 4


def _fingerprint(module):
    return tuple(p.data_ptr() for p in module.parameters())


def _hooked(module):
    if getattr(torch.nn.modules.module, '_global_forward_hooks', None) or getattr(torch.nn.modules.module, '_global_forward_pre_hooks', None):
        return True
    return any(m._forward_hooks or m._forward_pre_hooks for m in module.modules())


class _Lane:
    """One sequence being planned: its KV cache at a fixed size, and one recorded step that reads the token
    and position from GPU buffers, writes that position's keys and values, and attends over the cache
    with every later position masked out."""

    def __init__(self, fast, cap):
        p, c = fast.planner, fast.planner.c
        dev, dt = fast.device, fast.dtype
        self.fast, self.cap, self.pos = fast, cap, 0
        self.k = [torch.zeros(1, c.num_key_value_heads, cap, c.head_dim, dtype=dt, device=dev) for _ in p.layers]
        self.v = [torch.zeros_like(k) for k in self.k]
        self.token = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.at = torch.full((1,), cap - 1, dtype=torch.long, device=dev)
        self.span = torch.arange(cap, device=dev)
        self.logits = torch.zeros(fast.head.shape[0], dtype=torch.float32, device=dev)
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(2): self.forward()
        torch.cuda.current_stream(dev).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode='thread_local'):
            self.forward()

    def forward(self):
        f, p = self.fast, self.fast.planner
        x = p.embed_tokens(self.token)
        cos, sin = f.cos.index_select(0, self.at), f.sin.index_select(0, self.at)
        mask = (self.span <= self.at)[None, None, None]
        for i, layer in enumerate(p.layers):
            q, k, v = layer.self_attn.qkv(layer.input_layernorm(x), cos, sin)
            self.k[i].index_copy_(2, self.at, k); self.v[i].index_copy_(2, self.at, v)
            x = x + layer.self_attn.attend(q, self.k[i], self.v[i], mask)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        self.logits.copy_(F.linear(p.norm(x)[:, -1], f.head)[0])

    def start(self, ids):
        f = self.fast
        ks, vs, h = f.planner.prefill(torch.tensor(ids, device=f.device), prefix_mask(len(ids), len(ids), f.device))
        for i in range(len(ks)):
            self.k[i][:, :, :len(ids)] = ks[i]; self.v[i][:, :, :len(ids)] = vs[i]
        self.pos = len(ids)
        self.logits.copy_(F.linear(h[:, -1], f.head)[0])
        return self

    def step(self, token):
        self.token.fill_(token); self.at.fill_(self.pos)
        self.graph.replay()
        self.pos += 1


class _FastPlanner:
    """Recorded planner steps for one placement of the weights in memory (rebuilt if they move).
    Only the rows of the output layer that planning can pick (the audio codes and audio_end) are kept."""

    def __init__(self, robin):
        p = self.planner = robin.planner
        w = p.lm_head.weight
        self.device, self.dtype, self.key = w.device, w.dtype, _fingerprint(p)
        rows = torch.cat((torch.arange(robin.offset, robin.offset + robin.codes), torch.tensor([robin.audio_end]))).to(self.device)
        self.head = w.index_select(0, rows)
        c = p.c
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.float32, device=self.device) / c.head_dim))
        ang = torch.arange(c.max_position_embeddings, dtype=torch.float32, device=self.device)[:, None] * inv[None]
        self.cos, self.sin = ang.cos(), ang.sin()
        self.lanes = OrderedDict()

    def lane(self, need, slot):
        cap = min(-(-need // LANE_BLOCK) * LANE_BLOCK, self.cos.shape[0])
        key = (cap, slot)
        if key in self.lanes:
            self.lanes.move_to_end(key)
        else:
            while len(self.lanes) >= LANES_KEPT: self.lanes.popitem(last=False)
            self.lanes[key] = _Lane(self, cap)
        return self.lanes[key]


class Robin:
    def __init__(self, planner, decoder, vae, tokenizer, config, device):
        self.planner, self.decoder, self.vae, self.tok, self.device = planner, decoder, vae, tokenizer, device
        t = config['tokens']
        self.eod, self.score_start, self.score_end = t['eod'], t['score_start'], t['score_end']
        self.audio_start, self.audio_end, self.offset, self.codes = t['audio_start'], t['audio_end'], t['audio_offset'], t['audio_codes']
        self.instruction, self.defaults = config['instruction'], config['sampling']
        self._speech, self.lora, self.weights_file, self.vae_file = None, None, None, None
        self._lora_count, self._merged, self._merged_mods, self._backup, self._probe = 0, None, [], None, None
        self._fast, self._graphs_failed, self.planned_with = None, False, None

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
        self._lora_count += 1; lora.serial = self._lora_count
        self.lora = lora if MERGE_LORA else lora.attach()  # merged into the weights on first use (see _sync_lora)
        return self

    def set_lora_scale(self, scale):
        if self.lora is None: raise ValueError('no LoRA is loaded')
        self.lora.scale = scale

    def remove_lora(self):
        if self.lora is not None:
            self.lora.detach(); self.lora = None
        self._sync_lora()  # put the original weights back now: training builds on them

    # A merged LoRA is undone by reading the original tensors back from the weights file, not by
    # subtracting (bf16 rounding would build up over strength changes). Merging happens on first use,
    # when the weights are on the GPU, and a probe of the merged values catches the weights having been
    # swapped out from under it.
    def _weight_key(self, name):
        net, rest = name.split('.', 1)
        return f'robin.planner.{rest}.weight' if net == 'planner' else f'robin.decoder.model.{rest}.weight'

    def _from_file(self, names):
        if self.weights_file is None or not Path(self.weights_file).is_file(): return False
        with safe_open(str(self.weights_file), framework='pt', device='cpu') as f:
            keys = set(f.keys())
        return all(self._weight_key(n) in keys for n in names)

    def _probe_values(self):
        mods = self._merged_mods
        picks = mods[::max(1, len(mods) // 12)] + mods[-1:]
        return [m.weight.detach().reshape(-1)[:16].float().cpu() for _, m in picks]

    @torch.no_grad()
    def _sync_lora(self):
        lo = self.lora
        want = (lo.serial, float(lo.scale)) if lo is not None and not lo._handles and lo.scale else None
        if want is None and self._merged is None: return
        if want is not None and want == self._merged:
            if all(torch.equal(a, b) for a, b in zip(self._probe_values(), self._probe)): return
            warnings.warn('Robin: the LoRA merge was lost (the weights were replaced); merging it again')
        if self._merged is not None: self._restore_weights()
        if want is not None: self._merge_lora(lo); self._merged = want

    def _merge_lora(self, lo):
        mods = list(zip(lo.names, lo._mods))
        self._backup = None if self._from_file(lo.names) else {}
        for i, (name, m) in enumerate(mods):
            w = m.weight
            if self._backup is not None: self._backup[name] = w.detach().to('cpu', copy=True)
            w.copy_((w.float() + lo.delta(i, w.device)).to(w.dtype))
        self._merged_mods = mods
        self._probe = self._probe_values()

    def _restore_weights(self):
        if self._backup is not None:
            for name, m in self._merged_mods: m.weight.copy_(self._backup[name])
        else:
            with safe_open(str(self.weights_file), framework='pt', device='cpu') as f:
                for name, m in self._merged_mods: m.weight.copy_(f.get_tensor(self._weight_key(name)))
        self._merged, self._merged_mods, self._backup, self._probe = None, [], None, None

    def _fast_planner(self, need):
        """The recorded-step planner, or None where it does not apply (CPU, hooks on the planner as in
        training, or a sequence longer than Robin's positions)."""
        p = self.planner
        if not GRAPHS or self._graphs_failed or p.lm_head.weight.device.type != 'cuda': return None
        if need > p.c.max_position_embeddings or _hooked(p): return None
        if self._fast is None or self._fast.key != _fingerprint(p):
            self._fast = None
            self._fast = _FastPlanner(self)
        return self._fast

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
        self._sync_lora()
        g = torch.Generator(device=self.device).manual_seed(int(seed) % 2 ** 64)
        seqs = [ids] + ([negative] if negative is not None and cfg != 1. else [])
        lanes, fast = None, self._fast_planner(max(map(len, seqs)) + budget + 1)
        if fast is not None:
            try:
                lanes = [fast.lane(len(s) + budget + 1, slot).start(s) for slot, s in enumerate(seqs)]
            except RuntimeError as e:  # a GPU or driver that cannot record graphs: plan step by step from now on
                warnings.warn(f'Robin: CUDA graphs are not available here ({e}); planning without them')
                self._graphs_failed, self._fast, lanes = True, None, None
        self.planned_with = 'cuda graph' if lanes is not None else 'steps'
        if lanes is not None:  # token indices are counted from the first audio code; audio_end comes last
            end, shift, allowed = self.codes, self.offset, None
            logits = lambda: lanes[0].logits if len(lanes) == 1 else lanes[1].logits + cfg * (lanes[0].logits - lanes[1].logits)

            def advance(tok):
                for lane in lanes: lane.step(tok + shift)
        else:
            branches = [self._branch(s, budget) for s in seqs]
            end, shift = self.audio_end, 0
            allowed = torch.full_like(branches[0]['logits'], float('-inf'))
            allowed[self.offset:self.offset + self.codes] = 0; allowed[self.audio_end] = 0
            logits = lambda: branches[0]['logits'] if len(branches) == 1 else branches[1]['logits'] + cfg * (branches[0]['logits'] - branches[1]['logits'])

            def advance(tok):
                for b in branches:
                    b['logits'] = self.planner.step(tok, b['pos'], b['k'], b['v']); b['pos'] += 1
        out = []
        for step in range(budget):
            s = logits() + allowed if allowed is not None else logits().clone()
            if step < min_tokens: s[end] = float('-inf')
            if pause_guard and len(out) >= pause_guard:
                recent = set(out[-pause_guard:])
                if len(recent) <= guard_distinct: s[torch.tensor(sorted(recent), device=s.device)] = float('-inf')
            if temperature == 0:
                tok = int(s.argmax())
            else:
                s = s / temperature
                if top_k: s = s.masked_fill(s < s.topk(min(top_k, s.numel())).values[-1], float('-inf'))
                if top_p < 1:
                    v, ix = s.sort(descending=True); cp = v.softmax(-1).cumsum(-1)
                    drop = cp - v.softmax(-1) > top_p; s = s.masked_fill(torch.zeros_like(s, dtype=torch.bool).scatter(0, ix, drop), float('-inf'))
                tok = int(torch.multinomial(s.softmax(-1), 1, generator=g))
            if tok == end: return [t + shift for t in out], False
            out.append(tok)
            if callback is not None and callback('plan', step + 1, budget) is False: raise Cancelled()
            advance(tok)
        return [t + shift for t in out], True

    @torch.no_grad()
    def latents(self, ids, tokens, *, seed=0, steps=None, sampler=None, callback=None):
        steps = self.defaults['decoder_steps'] if steps is None else steps
        sampler = self.defaults['decoder_sampler'] if sampler is None else sampler
        self._sync_lora()
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
