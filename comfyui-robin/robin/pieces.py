import re

import numpy as np

from .text import speakable

RATE = 48000
SAMPLE_LINE = 'Here is a short sample of my voice, read clearly and at an easy pace.'
SAMPLE_LINES = dict(en=SAMPLE_LINE, ja='これは私の声の短いサンプルです。はっきりと、ゆっくり読んでいます。', mix=SAMPLE_LINE)
TAG = re.compile(r'\[[^\]]*\]')
ENDS = re.compile(r'(?<=[.!?])\s+|(?<=[.!?]")\s+|(?<=[。！？])\s*')


def size(text, lang='en'):
    text = TAG.sub(' ', text)
    if lang == 'en':
        return len(re.findall(r"[A-Za-z0-9']+", text))
    return len(re.sub(r'[\s\W]', '', text)) / 2.8


def sentences(text):
    out, cur = [], ''
    for part in ENDS.split(text.strip()):
        if not part: continue
        cur = f'{cur} {part}'.strip() if cur else part
        if cur.count('"') % 2 == 0: out.append(cur); cur = ''
    return out + ([cur] if cur else [])


def clauses(s, limit, lang):
    if size(s, lang) <= limit: return [s]
    best = None
    for m in re.finditer(r'[,;:、，]\s*', s):
        i = m.end()
        if s[:i].count('"') % 2 or i >= len(s): continue
        d = abs(size(s[:i], lang) - size(s, lang) / 2)
        if best is None or d < best[0]: best = (d, i)
    if best is None: return [s]
    return clauses(s[:best[1]].strip(), limit, lang) + clauses(s[best[1]:].strip(), limit, lang)


def glue(parts, lang):
    if lang == 'en': return ' '.join(parts)
    out = ''
    for p in parts:
        out += (' ' if lang == 'mix' and out and (out[-1].isascii() or p[0].isascii()) else '') + p
    return out


def split(text, max_words=35, lang='en', as_written=False):
    if lang == 'en' and not as_written: text = speakable(text)
    parts = [c for s in sentences(text) for c in clauses(s, max_words, lang)]
    chunks, cur = [], []
    for p in parts:
        if cur and size(glue(cur + [p], lang), lang) > max_words:
            chunks.append(glue(cur, lang)); cur = []
        cur.append(p)
    return chunks + ([glue(cur, lang)] if cur else [])


def frames_on(x, rate=RATE):
    f = rate // 20; n = len(x) // f
    return 20 * np.log10(np.sqrt((x[:n * f].reshape(n, f) ** 2).mean(1)) + 1e-9) > -45


def stats(x, rate=RATE):
    on = frames_on(x, rate); idx = np.flatnonzero(on); dur = len(on) / 20
    if not len(idx): return dict(sound=0., lead=dur, gap=0., tail=0.)
    gap = run = 0
    for v in on[idx[0]:idx[-1] + 1]:
        run = 0 if v else run + 1; gap = max(gap, run)
    return dict(sound=on.sum() / 20, lead=idx[0] / 20, gap=gap / 20, tail=dur - (idx[-1] + 1) / 20)


def expected_seconds(text, lang='en'):
    per_second = 2.7 if lang == 'en' else 2.5
    return max(1., size(text, lang) / per_second + .8 * len(TAG.findall(text)))


def check(x, text, lang='en', capped=False):
    st = stats(x); ratio = st['sound'] / expected_seconds(text, lang); problems = []
    if st['sound'] < .3: problems.append('no sound')
    if st['lead'] > 1.: problems.append('starts late')
    if st['gap'] > 2.5: problems.append('goes quiet')
    if ratio < .45: problems.append('too short')
    if ratio > 1.8: problems.append('too long')
    if capped and st['tail'] < .5: problems.append('still talking at the length limit')
    return (10 * len(problems) + abs(np.log(max(ratio, 1e-3)))), problems


def trim(x):
    f = RATE // 20; idx = np.flatnonzero(frames_on(x))
    if not len(idx): return x[:0]
    y = x[max(0, idx[0] * f - RATE // 20):min(len(x), (idx[-1] + 1) * f + int(.12 * RATE))].copy()
    f = min(len(y) // 2, RATE // 100)
    y[:f] *= np.linspace(0, 1, f); y[len(y) - f:] *= np.linspace(1, 0, f)
    return y


def loudness(x):
    f = RATE // 20; n = len(x) // f
    e = (x[:n * f].reshape(n, f) ** 2).mean(1); on = 10 * np.log10(e + 1e-12) > -45
    return float(np.sqrt(e[on].mean())) if on.any() else 0.


def join(pieces, texts):
    ref = loudness(pieces[0]) or 1.; out = []
    for i, (y, t) in enumerate(zip(pieces, texts)):
        level = loudness(y)
        out.append(y * (min(2., max(.5, ref / level)) if level else 1.))
        if i < len(pieces) - 1:
            pause = .35 if re.search(r'[.!?。！？]["\]]?\s*$', t) else .2
            out.append(np.zeros(int(pause * RATE), dtype=np.float32))
    return np.concatenate(out).astype(np.float32)


class Quiet:
    def start(self, seconds): pass

    def finish(self, extra=0): pass

    def __call__(self, stage, step, total): pass


def _best_take(robin, text, voice, lang, seed, retries, progress, **kwargs):
    best = None
    for k in range(retries + 1):
        progress.start(expected_seconds(text, lang))
        audio, _, info = robin.speak(text, voice, lang=lang, seed=seed + 7919 * k, callback=progress, **kwargs)
        score, problems = check(audio, text, lang, info['capped'])
        if best is None or score < best[0]: best = (score, problems, audio)
        if not problems or k == retries:
            progress.finish(); break
        progress.finish(extra=1)
    return best[2], best[1]


def read(robin, text, voice=None, *, reference=None, reference_rate=None, reference_text=None, seed=0, cfg=1., retries=2, max_words=35,
         lang='en', as_written=False, progress=None, **sampling):
    progress = progress or Quiet()
    chunks = split(text, max_words, lang, as_written)
    if not chunks: raise ValueError('There is no text to read.')
    voice = (voice or '').strip() or None
    ref = None if reference is None else (reference, reference_rate, (reference_text or '').strip() or None)
    kept = []
    for i, chunk in enumerate(chunks):
        audio, problems = _best_take(robin, chunk, voice, lang, seed + 1000 * i, retries, progress, cfg=cfg,
                                     reference=ref[0] if ref else None, reference_rate=ref[1] if ref else None,
                                     reference_text=ref[2] if ref else None, **sampling)
        take = trim(audio)
        kept.append(take)
        if not problems and len(chunks) > 1 and len(take):
            ref = (take, RATE, chunk)
    return join(kept, chunks)


def change(robin, source, source_rate, source_text, *, reference=None, reference_rate=None, reference_text=None, voice=None, seed=0,
           retries=2, lang='en', progress=None, **sampling):
    progress = progress or Quiet()
    voice = (voice or '').strip() or None
    if not (source_text or '').strip(): raise ValueError('The source text must say exactly what the source clip says.')
    if reference is None:
        line = SAMPLE_LINES.get(lang, SAMPLE_LINE)
        audio, _ = _best_take(robin, line, voice, lang, seed, retries, progress, **sampling)
        reference, reference_rate, reference_text = trim(audio), RATE, line
    seconds = len(source) / source_rate
    target = stats(source, source_rate)['sound'] or seconds
    best = None
    for k in range(retries + 1):
        progress.start(seconds)
        audio, _, info = robin.convert(source, source_text.strip(), reference, reference_text=(reference_text or '').strip() or None,
                                       source_rate=source_rate, reference_rate=reference_rate, description=voice, lang=lang,
                                       seed=seed + 7919 * k, callback=progress, **sampling)
        st = stats(audio); ratio = st['sound'] / max(target, .1)
        problems = st['sound'] < .3 or not .5 <= ratio <= 2. or (info['capped'] and st['tail'] < .5)
        score = 10 * problems + abs(np.log(max(ratio, 1e-3)))
        if best is None or score < best[0]: best = (score, audio)
        if not problems or k == retries:
            progress.finish(); break
        progress.finish(extra=1)
    return best[1]
