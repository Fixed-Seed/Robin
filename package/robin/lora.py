import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

TARGETS = ('self_attn.qkv_proj', 'self_attn.o_proj', 'mlp.gate_up_proj')
FORMAT = 'robin-lora-1'


def file_sha256(path):
    path = Path(path).resolve(); st = path.stat()
    key = f'{path}:{st.st_size}:{st.st_mtime_ns}'
    cache = Path(os.environ.get('ROBIN_CACHE', Path.home()/'.cache'/'robin'))/'sha256.json'
    known = {}
    try: known = json.loads(cache.read_text())
    except (OSError, ValueError): pass
    if key not in known:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''): h.update(chunk)
        known[key] = h.hexdigest()
        try:
            cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(known))
        except OSError: pass
    return known[key]


class LoRA(nn.Module):
    def __init__(self, nets, ranks, alphas=None, targets=TARGETS, seed=0):
        super().__init__()
        self.ranks, self.alphas, self.targets = dict(ranks), dict(alphas or ranks), tuple(targets)
        self.scale, self.names, self._mods, self._factor, self._handles = 1.0, [], [], [], []
        self.A, self.B = nn.ParameterList(), nn.ParameterList()
        g = torch.Generator().manual_seed(seed)
        for net, layers in nets.items():
            r = self.ranks[net]
            for i, layer in enumerate(layers):
                for t in self.targets:
                    m = layer.get_submodule(t)
                    a = torch.randn(r, m.in_features, generator=g) / math.sqrt(m.in_features)
                    self.A.append(nn.Parameter(a.to(m.weight.device)))
                    self.B.append(nn.Parameter(torch.zeros(m.out_features, r, device=m.weight.device)))
                    self.names.append(f'{net}.layers.{i}.{t}'); self._mods.append(m); self._factor.append(self.alphas[net] / r)

    def attach(self):
        if not self._handles:
            self._handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(self._mods)]
        return self

    def detach(self):
        for h in self._handles: h.remove()
        self._handles = []

    def _hook(self, i):
        def hook(_module, args, out):
            if self.scale == 0: return out
            delta = (args[0].float() @ self.A[i].T) @ self.B[i].T
            return out + (delta * (self._factor[i] * self.scale)).to(out.dtype)
        return hook

    def config(self, base_sha256=None):
        return dict(format=FORMAT, ranks=self.ranks, alphas=self.alphas, targets=list(self.targets), base_sha256=base_sha256)

    def save(self, path, base_sha256=None, dtype=torch.bfloat16):
        path = Path(path); tmp = path.with_name(path.name + '.tmp')
        state = {}
        for name, a, b in zip(self.names, self.A, self.B):
            state[f'{name}.lora_A'] = a.detach().to(dtype).cpu().contiguous(); state[f'{name}.lora_B'] = b.detach().to(dtype).cpu().contiguous()
        save_file(state, str(tmp), metadata=dict(robin_lora=json.dumps(self.config(base_sha256))))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, nets, device='cuda'):
        with safe_open(str(path), framework='pt', device='cpu') as f:
            meta = f.metadata() or {}
            if 'robin_lora' not in meta:
                raise ValueError(f'{path} is not a Robin LoRA')
            config = json.loads(meta['robin_lora'])
            lora = cls({k: v for k, v in nets.items() if k in config['ranks']}, config['ranks'], config['alphas'], config['targets'])
            keys = set(f.keys())
            for i, name in enumerate(lora.names):
                for which, params in (('lora_A', lora.A), ('lora_B', lora.B)):
                    k = f'{name}.{which}'
                    if k not in keys: raise ValueError(f'{path} is missing {k}')
                    t = f.get_tensor(k)
                    if t.shape != params[i].shape: raise ValueError(f'{path}: {k} has shape {tuple(t.shape)}, expected {tuple(params[i].shape)}')
                    params[i].data.copy_(t)
        return lora.to(device).requires_grad_(False), config
