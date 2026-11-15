import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class Config:
    vocab_size: int = 184704
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 24576
    latent_dim: int = 64


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight.to(x.dtype), self.eps)


def rope(positions, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    ang = positions.float().to(device)[:, None] * inv[None]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    f = x.float(); h = f.shape[-1] // 2
    lo, hi = f[..., :h], f[..., h:]
    return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin], dim=-1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.kv, self.d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.qkv_proj = nn.Linear(c.hidden_size, (self.h + 2 * self.kv) * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, c.hidden_size, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.d, c.rms_norm_eps), RMSNorm(self.d, c.rms_norm_eps)

    def qkv(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv_proj(x).split((self.h * self.d, self.kv * self.d, self.kv * self.d), dim=-1)
        q = self.q_norm(q.view(B, T, self.h, self.d)).transpose(1, 2)
        k = self.k_norm(k.view(B, T, self.kv, self.d)).transpose(1, 2)
        v = v.view(B, T, self.kv, self.d).transpose(1, 2)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def attend(self, q, k, v, mask=None):
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.h != self.kv)
        return self.o_proj(o.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_up_proj = nn.Linear(c.hidden_size, 2 * c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        g, u = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = MLP(c)


def planner_block(layer, x, cos, sin, mask):
    q, k, v = layer.self_attn.qkv(layer.input_layernorm(x), cos, sin)
    x = x + layer.self_attn.attend(q, k, v, mask)
    return x + layer.mlp(layer.post_attention_layernorm(x))


def decoder_block(layer, s, cos, sin, prefix_k, prefix_v):
    q, k, v = layer.self_attn.qkv(layer.input_layernorm(s), cos, sin)
    k = torch.cat((prefix_k.to(k.dtype), k), dim=2); v = torch.cat((prefix_v.to(v.dtype), v), dim=2)
    s = s + layer.self_attn.attend(q, k, v)
    return s + layer.mlp(layer.post_attention_layernorm(s))


class Planner(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.num_hidden_layers))
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward(self, ids, mask):
        x = self.embed_tokens(ids)
        cos, sin = rope(torch.arange(ids.shape[1]), self.c.head_dim, self.c.rope_theta, x.device)
        mask = mask.to(x.dtype)
        for layer in self.layers:
            x = checkpoint(planner_block, layer, x, cos, sin, mask, use_reentrant=False)
        return self.norm(x)

    @torch.no_grad()
    def prefill(self, ids, mask):
        x = self.embed_tokens(ids[None]); T = ids.shape[0]
        cos, sin = rope(torch.arange(T), self.c.head_dim, self.c.rope_theta, x.device)
        ks, vs = [], []
        for layer in self.layers:
            q, k, v = layer.self_attn.qkv(layer.input_layernorm(x), cos, sin)
            x = x + layer.self_attn.attend(q, k, v, mask.to(x.dtype))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            ks.append(k); vs.append(v)
        return ks, vs, self.norm(x)

    @torch.no_grad()
    def step(self, token, pos, kbuf, vbuf):
        x = self.embed_tokens(torch.tensor([[token]], device=self.lm_head.weight.device))
        cos, sin = rope(torch.tensor([pos]), self.c.head_dim, self.c.rope_theta, x.device)
        for i, layer in enumerate(self.layers):
            q, k, v = layer.self_attn.qkv(layer.input_layernorm(x), cos, sin)
            kbuf[i][:, :, pos] = k[:, :, 0]; vbuf[i][:, :, pos] = v[:, :, 0]
            x = x + layer.self_attn.attend(q, kbuf[i][:, :, :pos + 1], vbuf[i][:, :, :pos + 1])
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.lm_head(self.norm(x)[:, -1]).float()[0]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([a.cos(), a.sin()], dim=-1).to(self.mlp[0].weight.dtype))


def positions(n, dim, device):
    pos = torch.arange(n, dtype=torch.float32, device=device)[:, None]
    ang = pos * torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
    return pe


class _Stack(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = nn.ModuleList(Block(c) for _ in range(c.num_hidden_layers))
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)


class Decoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.model = _Stack(c)
        self.vae2llm = nn.Linear(c.latent_dim, c.hidden_size)
        self.llm2vae = nn.Linear(c.hidden_size, c.latent_dim)
        self.time_embedder = TimestepEmbedder(c.hidden_size)
        self._pe = None

    @torch.no_grad()
    def velocity(self, x, sigma, prefix_k, prefix_v):
        return self(x, sigma, prefix_k, prefix_v)

    def forward(self, x, sigma, prefix_k, prefix_v, checkpointing=False):
        dt = self.vae2llm.weight.dtype
        A = prefix_k[0].shape[2]; n = x.shape[1] + 2
        s = F.pad(x.to(dt), (0, 0, 1, 1))
        if self._pe is None or self._pe.shape[0] < n or self._pe.device != x.device:
            self._pe = positions(max(n, 1024), self.c.hidden_size, x.device).to(dt)
        s = self.vae2llm(s) + self.time_embedder(sigma.to(dt))[:, None] + self._pe[:n][None]
        cos, sin = rope(torch.arange(A, A + n), self.c.head_dim, self.c.rope_theta, x.device)
        for i, layer in enumerate(self.model.layers):
            if checkpointing:
                s = checkpoint(decoder_block, layer, s, cos, sin, prefix_k[i], prefix_v[i], use_reentrant=False)
            else:
                s = decoder_block(layer, s, cos, sin, prefix_k[i], prefix_v[i])
        return self.llm2vae(self.model.norm(s))[:, 1:-1].float()
