import argparse
import sys


def model_args(p, required=True):
    p.add_argument('--model', required=required, help='a weights folder or Hugging Face repo')
    p.add_argument('--vae', help='the VAE weights (.safetensors); by default the robin_vae.safetensors in the --model folder')
    p.add_argument('--device', default='cuda')


def main():
    p = argparse.ArgumentParser(prog='python -m robin')
    sub = p.add_subparsers(dest='command', required=True)

    s = sub.add_parser('speak', help='text to speech, and voice changing')
    s.add_argument('text', help='what to say, any length; [laughs] and [coughs] can go inline')
    s.add_argument('--voice', help='the voice in words, e.g. "a warm, low voice, unhurried and close to the mic"')
    s.add_argument('--convert', help='a recording (up to 45 s) to perform again in the --voice (or --lora) voice; the text must be exactly '
                                     'what it says')
    s.add_argument('--lora', help='a LoRA file trained with "python -m robin train"')
    s.add_argument('--lora-scale', type=float, default=1., help='how strongly the LoRA applies (1 = as trained)')
    s.add_argument('--lang', default='en', choices=('en', 'ja', 'mix'))
    s.add_argument('--as-written', action='store_true', help='read English exactly as typed, without spelling out numbers, symbols and '
                                                             'abbreviations first')
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--temperature', type=float); s.add_argument('--top-k', type=int); s.add_argument('--top-p', type=float)
    s.add_argument('--cfg', type=float, default=1., help='above 1 follows --voice more closely (try 3)')
    s.add_argument('--steps', type=int); s.add_argument('--sampler', choices=('dpm_2', 'dpmpp_2m'))
    s.add_argument('--out', default='robin.flac')
    model_args(s)

    g = sub.add_parser('gui', help='Robin in your browser: speak, change voices and train voices')
    g.add_argument('--out', default='robin_studio', help='where saved takes and trained voices go')
    g.add_argument('--host', default='127.0.0.1'); g.add_argument('--port', type=int, default=7860)
    model_args(g)

    def data_args(q):
        q.add_argument('data', help='a folder of audio clips, each with a .txt transcript of the same name, plus an optional voice.txt '
                                    'describing the voice')
        q.add_argument('--lang', default='en', choices=('en', 'ja', 'mix'))
        q.add_argument('--min-seconds', type=float, default=1., help='shorter clips are left out')
        q.add_argument('--max-seconds', type=float, default=30., help='longer clips are left out (split them first)')

    q = sub.add_parser('prep', help='check a LoRA dataset; with --out, also prepare it for training')
    data_args(q)
    q.add_argument('--out', help='the LoRA folder to prepare (the same --out you will give train)')
    model_args(q, required=False)

    t = sub.add_parser('train', help='train a LoRA on a voice')
    data_args(t)
    t.add_argument('--out', required=True, help='a folder for the LoRA, its checkpoints and samples')
    t.add_argument('--steps', type=int, default=1000)
    t.add_argument('--save-every', type=int, default=200)
    t.add_argument('--rank-planner', type=int, default=32); t.add_argument('--rank-decoder', type=int, default=64)
    t.add_argument('--lr-planner', type=float, default=5e-5); t.add_argument('--lr-decoder', type=float, default=2e-4)
    t.add_argument('--accum', type=int, default=8, help='clips per step')
    t.add_argument('--sample-text', help='the line rendered at every save, so you can pick a checkpoint by ear')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--restart', action='store_true', help='start over instead of continuing an unfinished run')
    model_args(t)

    a = p.parse_args()
    from .pipeline import Robin

    if a.command == 'gui':
        try:
            from . import gui
        except ModuleNotFoundError as e:
            if e.name != 'gradio': raise
            raise SystemExit("The GUI needs Gradio: pip install 'robin[gui]'")
        gui.launch(Robin.from_pretrained(a.model, a.vae, device=a.device), a.out, a.host, a.port)
        return

    if a.command == 'speak':
        import soundfile as sf
        from . import pieces
        tts = Robin.from_pretrained(a.model, a.vae, device=a.device, lora=a.lora, lora_scale=a.lora_scale)
        sampling = dict(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, steps=a.steps, sampler=a.sampler)
        if a.convert:
            if not (a.voice or a.lora): s.error('--convert needs --voice (a description of the new voice) or --lora')
            x, rate = sf.read(a.convert, dtype='float32', always_2d=True)
            if len(x) / rate > 45: s.error(f'{a.convert} is {len(x) / rate:.0f} s long; convert up to 45 s at a time')
            wav = pieces.change(tts, x.mean(1), rate, a.text, voice=a.voice, lang=a.lang, seed=a.seed, **sampling)
        else:
            wav = pieces.read(tts, a.text, a.voice, lang=a.lang, seed=a.seed, cfg=a.cfg, as_written=a.as_written, **sampling)
        sf.write(a.out, wav, pieces.RATE, subtype='PCM_24')
        print(f'wrote {a.out}: {len(wav) / pieces.RATE:.1f} s')
        return

    from . import train as tr
    if a.command == 'prep':
        found = tr.scan(a.data, a.lang, a.min_seconds, a.max_seconds); tr.describe(found)
        if a.out:
            if not a.model: q.error('--out needs --model')
            from pathlib import Path
            clips = tr.encode(Robin.from_pretrained(a.model, a.vae, device=a.device), found, Path(a.out)/'cache')
            print(f'{len(clips)} clips prepared in {Path(a.out)/"cache"}')
        return

    robin = Robin.from_pretrained(a.model, a.vae, device=a.device)
    tr.train(robin, a.data, a.out, lang=a.lang, min_seconds=a.min_seconds, max_seconds=a.max_seconds, restart=a.restart,
             steps=a.steps, save_every=a.save_every, rank_planner=a.rank_planner, rank_decoder=a.rank_decoder, lr_planner=a.lr_planner,
             lr_decoder=a.lr_decoder, accum=a.accum, sample_text=a.sample_text, seed=a.seed)


if __name__ == '__main__':
    sys.exit(main())
