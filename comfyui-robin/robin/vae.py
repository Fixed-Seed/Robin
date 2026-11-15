import torch
from safetensors.torch import load_file
from torch import nn

WIDTH, LATENT, AUDIO_CHANNELS = 64, 64, 2
MULTS = (1, 1, 2, 4, 8, 16, 32)
STRIDES = (2, 2, 4, 4, 5, 6)
NORMED = (('weight_g', 'weight_v'), ('parametrizations.weight.original0', 'parametrizations.weight.original1'))


class Snake(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(ch)); self.beta = nn.Parameter(torch.zeros(ch))

    def forward(self, x):
        freq, amp = self.alpha.exp().view(1, -1, 1), self.beta.exp().view(1, -1, 1)
        return x + torch.sin(freq * x).square() / (amp + 1e-9)


class Stack(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Residual(Stack):
    def __init__(self, ch, dilation):
        super().__init__(Snake(ch), nn.Conv1d(ch, ch, 7, padding=3 * dilation, dilation=dilation), Snake(ch), nn.Conv1d(ch, ch, 1))

    def forward(self, x):
        return x + self.layers(x)


def down(cin, cout, s):
    return Stack(Residual(cin, 1), Residual(cin, 3), Residual(cin, 9), Snake(cin), nn.Conv1d(cin, cout, 2 * s, s, (s + 1) // 2))


def up(cin, cout, s):
    return Stack(Snake(cin), nn.ConvTranspose1d(cin, cout, 2 * s, s, (s + 1) // 2, output_padding=s % 2),
                 Residual(cout, 1), Residual(cout, 3), Residual(cout, 9))


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [WIDTH * m for m in MULTS]
        n = len(STRIDES)
        self.encoder = Stack(nn.Conv1d(AUDIO_CHANNELS, ch[0], 7, padding=3), *[down(ch[i], ch[i + 1], STRIDES[i]) for i in range(n)],
                             Snake(ch[-1]), nn.Conv1d(ch[-1], 2 * LATENT, 3, padding=1))
        self.decoder = Stack(nn.Conv1d(LATENT, ch[-1], 7, padding=3), *[up(ch[i + 1], ch[i], STRIDES[i]) for i in reversed(range(n))],
                             Snake(ch[0]), nn.Conv1d(ch[0], AUDIO_CHANNELS, 7, padding=3, bias=False))

    @torch.no_grad()
    def encode(self, audio):
        return self.encoder(audio)[:, :LATENT]

    @torch.no_grad()
    def decode(self, latents):
        return self.decoder(latents)


def plain_weights(state):
    state = {k.removeprefix('vae.'): v.float() for k, v in state.items()}
    out = {}
    for k, v in state.items():
        mag = next((m for m, d in NORMED if k.endswith(m)), None)
        if mag:
            stem = k[:-len(mag)]; direction = state[stem + dict(NORMED)[mag]]
            out[stem + 'weight'] = direction * (v / direction.flatten(1).norm(dim=1).view(-1, *[1] * (direction.dim() - 1)))
        elif not any(k.endswith(d) for _, d in NORMED):
            out[k] = v
    return out


def load_vae(path, device='cuda'):
    vae = VAE(); vae.load_state_dict(plain_weights(load_file(str(path))), strict=True)
    return vae.to(device, torch.float32).eval()
