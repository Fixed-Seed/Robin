from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample

HOP = 640
CORE, CTX = 700, 25


def load_audio(audio, sample_rate=None):
    if isinstance(audio, (str, bytes)) or hasattr(audio, '__fspath__'):
        x, sample_rate = sf.read(str(audio), dtype='float32', always_2d=True)
        return torch.from_numpy(x.mean(1)), sample_rate
    if sample_rate is None:
        raise ValueError('pass sample_rate with an array')
    x = torch.as_tensor(np.asarray(audio, dtype=np.float32))
    return (x.mean(1) if x.ndim == 2 else x), sample_rate


class SpeechTokenizer:
    def __init__(self, path, device='cuda'):
        import s3tokenizer
        path = Path(path)
        if not path.is_file():
            raise ValueError(f'The speech tokenizer is missing: put {path.name} in {path.parent}. It comes with Robin on Hugging Face.')
        self.s3, self.device = s3tokenizer, device
        self.model = s3tokenizer.load_model(str(path)).to(device).eval()

    @torch.no_grad()
    def _codes(self, wav16):
        m, lens = self.s3.padding([self.s3.log_mel_spectrogram(wav16, n_mels=128)])
        codes, clens = self.model.quantize(m.to(self.device), lens.to(self.device))
        return codes[0, :clens[0]].cpu().numpy()

    def __call__(self, audio, sample_rate=None):
        x, rate = load_audio(audio, sample_rate)
        wav16 = resample(x, rate, 16000)
        total = int(round(len(wav16) / HOP))
        if total <= CORE:
            return self._codes(wav16).astype(np.int64)
        out = []
        for s in range(0, total, CORE):
            a = max(0, s - CTX); b = min(total, s + CORE + CTX)
            c = self._codes(wav16[a * HOP:b * HOP])
            keep = min(CORE, total - s); piece = c[s - a:s - a + keep]
            if len(piece) < keep: piece = np.concatenate([piece, np.repeat(c[-1:], keep - len(piece))])
            out.append(piece)
        return np.concatenate(out).astype(np.int64)
