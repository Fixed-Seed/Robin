import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

const RED = '#D9302A';
const INK = '#1B1B1C';
const CREAM = '#F2EEE6';
const AUDIO_LINK = 'rgba(242,238,230,0.75)';
const HERE = new URL('.', import.meta.url).href;
const TITLE_FONT = '600 14px "Familjen Grotesk", Inter, system-ui, sans-serif';
const TAG_FONT = '9px "IBM Plex Mono", ui-monospace, Menlo, monospace';
const UPLOADS = 'robin_datasets';
const MAX_TAKE = 45;  // the Voice Changer's limit, in seconds
const SOUND_FILES = /\.(wav|flac|mp3|ogg|opus|aiff?)$/i;  // what Robin reads as is; other audio becomes WAV in the browser first
const CLIP_FILES = /\.(wav|flac|mp3|ogg|opus|aiff?|m4a|aac|wma|mp4|webm|mov|txt)$/i;
const EMPTY_REPORT = 'Choose a folder of clips, or type its path in folder.';
const DASH = '—';
// A new node's width, and the extra height its text boxes get.
const SIZES = { RobinLoader: [330, 0], RobinLoRA: [330, 0], RobinSpeak: [430, 110], RobinVoiceChanger: [430, 90], RobinCheckDataset: [440, 90],
  RobinTrainLoRA: [460, 30], RobinLoadAudio: [380, 0] };
const ICONS = {
  play: '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="#D9302A" stroke="#D9302A" stroke-width="1.5" stroke-linejoin="round"/></svg>',
  pause: '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 3h3v10H4zM9 3h3v10H9z" fill="#D9302A"/></svg>',
  save: '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 2.5v8M4.5 7.5 8 11l3.5-3.5M3 13.5h10"/></svg>',
  folder: '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M1.5 4a1 1 0 0 1 1-1h3.7l1.5 1.6h5.8a1 1 0 0 1 1 1V12a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1z" fill="#D9302A"/></svg>',
  record: '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5" fill="#D9302A"/></svg>',
  stop: '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><rect x="3.5" y="3.5" width="9" height="9" rx="1.5" fill="#D9302A"/></svg>',
  open: '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M9 2.5h4.5V7M13.5 2.5 7.5 8.5M12 9.5v3a1 1 0 0 1-1 1H3.5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h3" fill="none" stroke="#D9302A" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  more: '<svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M3 1.5 7 5 3 8.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
};

const sheet = Object.assign(document.createElement('link'), { rel: 'stylesheet', href: HERE + 'robin.css' });
sheet.addEventListener('load', () => Promise.all([TITLE_FONT, TAG_FONT].map((font) => document.fonts.load(font))).then(redraw));
document.head.append(sheet);

const badge = document.createElement('canvas');
const art = new Image();
art.addEventListener('load', () => {
  badge.width = badge.height = 96;
  badge.getContext('2d').drawImage(art, 0, 0, 96, 96);
  badge.ready = true;
  redraw();
});
art.src = HERE + 'robin_badge.svg';

function redraw() {
  app.canvas?.setDirty(true, true);
}

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'class') el.className = value;
    else if (key === 'html') el.innerHTML = value;
    else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, '');
    else if (value != null && value !== false) el.setAttribute(key, value);
  }
  el.append(...children);
  return el;
}

function icon(name, size = 14) {
  return h('span', { html: ICONS[name].replace(/width="\d+" height="\d+"/, `width="${size}" height="${size}"`) }).firstChild;
}

function chain(proto, name, fn) {
  const original = proto[name];
  proto[name] = function () {
    const result = original?.apply(this, arguments);
    fn.apply(this, arguments);
    return result;
  };
}

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function value(node, name) {
  return widget(node, name)?.value;
}

function watch(node, names, fn) {
  for (const name of names) {
    const w = widget(node, name);
    if (w) {
      const callback = w.callback;
      w.callback = function () {
        const result = callback?.apply(this, arguments);
        fn();
        return result;
      };
    }
  }
}

function addPanel(node, name, element, height, grow = false) {
  const options = { serialize: false, getMinHeight: () => height + 20 };
  if (!grow) options.getMaxHeight = () => height + 20;
  const w = node.addDOMWidget(name, 'robin', element, options);
  w.serialize = false;
  return w;
}

function findNode(id) {
  let graph = app.rootGraph ?? app.graph;
  let node;
  for (const part of String(id ?? '').split(':')) {
    node = graph?.getNodeById(part);
    graph = node?.subgraph;
  }
  return node;
}

function viewUrl(file) {
  return api.apiURL(`/view?${new URLSearchParams({ filename: file.filename, subfolder: file.subfolder ?? '', type: file.type ?? 'temp' })}`);
}

function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function span(seconds) {
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return 'under a minute';
  return minutes < 90 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

function gigabytes(bytes) {
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

// One sound at a time across every Robin node: starting a take or a sample stops the one playing.
let playing = null;

function stopPlaying() {
  if (!playing) return;
  const { audio, changed } = playing;
  playing = null;
  audio.pause();
  changed(false);
}

function toggle(audio, changed) {
  if (playing?.audio === audio) return stopPlaying();
  stopPlaying();
  playing = { audio, changed };
  audio.onended = () => playing?.audio === audio && stopPlaying();
  changed(true);
  audio.play().catch(() => playing?.audio === audio && stopPlaying());
}

// -> the sound as mono samples at 48 kHz, from a URL or a file.
async function sound(source) {
  const data = source instanceof Blob ? await source.arrayBuffer() : await (await fetch(source)).arrayBuffer();
  const decoded = await new OfflineAudioContext(1, 1, 48000).decodeAudioData(data);
  const mono = new Float32Array(decoded.length);
  for (let c = 0; c < decoded.numberOfChannels; c++) decoded.getChannelData(c).forEach((v, i) => (mono[i] += v / decoded.numberOfChannels));
  return mono;
}

function peaks(samples, count) {
  const size = Math.max(1, Math.floor(samples.length / count));
  const found = Array.from({ length: count }, (_, i) => {
    let peak = 0;
    for (let j = i * size, end = Math.min(samples.length, j + size); j < end; j++) peak = Math.max(peak, Math.abs(samples[j]));
    return peak;
  });
  const top = Math.max(...found, 1e-6);
  return found.map((peak) => Math.sqrt(peak / top));
}

// Title bar: the badge, the name in Familjen Grotesk and a ROBIN tag, on scarlet.
function drawTitleBar(ctx, height, size) {
  const width = size[0];
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(0, -height, width, height, this.flags?.collapsed ? [8] : [8, 8, 0, 0]);
  ctx.fillStyle = RED;
  ctx.fill();
  ctx.clip();
  ctx.fillStyle = 'rgba(255,255,255,0.22)';
  ctx.fillRect(0, -height, width, 1);
  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  ctx.fillRect(0, -1, width, 1);
  ctx.restore();
}

function drawTitleBox(ctx, height) {
  if (badge.ready) ctx.drawImage(badge, 7, -height + (height - 20) / 2, 20, 20);
}

function fit(ctx, text, room) {
  if (ctx.measureText(text).width <= room) return text;
  while (text.length > 1 && ctx.measureText(text + '…').width > room) text = text.slice(0, -1);
  return text + '…';
}

function drawTitleText(ctx, height, size, scale, font, selected) {
  if (app.canvas?.low_quality) return;
  const width = size[0];
  const y = -height / 2 + 1;
  let room = width - 44;
  ctx.save();
  ctx.textBaseline = 'middle';
  if (!this.flags?.collapsed) {
    ctx.font = TAG_FONT;
    ctx.letterSpacing = '1.3px';
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(242,238,230,0.78)';
    ctx.fillText('ROBIN', width - 11, y);
    room -= ctx.measureText('ROBIN').width + 14;
    ctx.letterSpacing = '0px';
  }
  ctx.font = TITLE_FONT;
  ctx.textAlign = 'left';
  ctx.fillStyle = selected ? '#FFFFFF' : CREAM;
  ctx.fillText(fit(ctx, String(this.getTitle?.() ?? this.title), room), 34, y);
  ctx.restore();
}

// The two red bands across the bottom of the badge close every Robin node; Robin's own slots get a cream ring.
function drawBands(ctx) {
  if (this.flags?.collapsed) return;
  const [width, height] = this.size;
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(0, 0, width, height, [0, 0, 8, 8]);
  ctx.clip();
  ctx.fillStyle = RED;
  ctx.fillRect(0, height - 10, width, 3);
  ctx.fillRect(0, height - 5, width, 3);
  ctx.restore();
  if (app.canvas?.low_quality) return;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = CREAM;
  const ring = (slots, at) => slots?.forEach((slot, i) => {
    if (slot.type !== 'ROBIN_MODEL' && slot.type !== 'ROBIN_DATASET') return;
    const [x, y] = at.call(this, i);
    ctx.beginPath();
    ctx.arc(x - this.pos[0], y - this.pos[1], 5.5, 0, Math.PI * 2);
    ctx.stroke();
  });
  ring(this.inputs, this.getInputPos);
  ring(this.outputs, this.getOutputPos);
  ctx.restore();
}

function loaderPanel(node) {
  const rows = {};
  const row = (key) => h('div', { class: 'row' }, h('span', {}, key), (rows[key] = h('span', {}, DASH)));
  addPanel(node, 'robin_status', h('div', { class: 'robin-panel' }, h('div', { class: 'robin-lcd' }, row('model'), row('voice'), row('from'))), 74);
  const name = () => (value(node, 'model') === 'robin' ? 'robin_v1' : value(node, 'model'));
  const show = (model, voice, from) => Object.entries({ model, voice, from }).forEach(([key, text]) => (rows[key].textContent = text));

  async function refresh() {
    const res = await api.fetchApi(`/robin/status?${new URLSearchParams({ model: value(node, 'model') ?? 'robin', vae: value(node, 'vae') ?? '' })}`);
    if (!res.ok) return;
    const status = await res.json();
    show(`${name()} · ${status.ready ? 'on disk' : 'to download'}`, `robin · ${status.voice ? 'on disk' : 'to download'}`,
      status.ready ? `models/robin · ${gigabytes(status.size)}` : 'hugging face');
  }
  watch(node, ['model', 'vae'], refresh);
  setTimeout(() => refresh().catch(() => {}));

  return {
    loader(detail) {
      if (detail.stage === 'load') {
        rows.model.textContent = `${name()} · loading`;
        return;
      }
      const voice = detail.file.startsWith('robin_character_lora');
      if (voice) rows.voice.textContent = 'robin · downloading';
      else rows.model.textContent = `${name()} · downloading`;
      rows.from.textContent = voice ? 'hugging face' : `hugging face · ${detail.file.replace(/\.[^.]+$/, '')}`;
    },
    executed(message) {
      const status = message?.robin_status?.[0];
      if (status) show(`${name()} · ready · ${status.loaded}`, 'robin · on disk',
        `${status.downloaded ? 'hugging face' : 'models/robin'} · ${gigabytes(status.size)}`);
    },
  };
}

// The player on Speak, Voice Changer and Load Audio: the badge play button, a waveform to click and seek, the time, and for takes a
// download button.
function makePlayer(what, { download = false } = {}) {
  const play = h('button', { class: 'robin-play', type: 'button', 'aria-label': `Play ${what}`, disabled: true }, icon('play'));
  const wave = h('div', { class: 'robin-wave empty' });
  const time = h('span', { class: 'robin-time' });
  const save = h('a', { class: 'robin-save', 'aria-label': `Download ${what}`, title: `Download ${what}`, hidden: true }, icon('save', 16));
  const row = h('div', { class: 'robin-take' }, play, wave, time, ...(download ? [save] : []));
  let audio = null;
  let seconds = 0;
  let bars = [];
  let frame = 0;

  const paint = () => {
    const at = audio?.duration ? audio.currentTime / audio.duration : 0;
    bars.forEach((bar, i) => bar.classList.toggle('on', i < Math.round(at * bars.length)));
    time.textContent = `${clock(audio?.currentTime)} / ${clock(audio?.duration || seconds)}`;
  };
  const changed = (on) => {
    play.classList.toggle('on', on);
    play.replaceChildren(icon(on ? 'pause' : 'play'));
    play.setAttribute('aria-label', `${on ? 'Pause' : 'Play'} ${what}`);
    cancelAnimationFrame(frame);
    const tick = () => {
      paint();
      frame = requestAnimationFrame(tick);
    };
    if (on) tick();
    else paint();
  };
  const draw = (heights, lit) => {
    if (bars.length !== heights.length) {
      bars = heights.map(() => h('i'));
      wave.replaceChildren(...bars);
    }
    wave.classList.remove('empty');
    heights.forEach((height, i) => {
      bars[i].style.height = `${Math.max(3, Math.round(height * 32))}px`;
      bars[i].classList.toggle('on', lit);
    });
  };
  play.addEventListener('click', () => audio && toggle(audio, changed));
  wave.addEventListener('click', (event) => {
    if (!audio?.duration) return;
    const box = wave.getBoundingClientRect();
    audio.currentTime = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)) * audio.duration;
    paint();
  });

  const player = {
    row,
    // -> the length in seconds, or null when the file can't be played or a newer one replaced it.
    async load(url, { length = 0, name = null, missing = "can't play this one" } = {}) {
      player.stop();
      audio = new Audio(url);
      seconds = length;
      play.disabled = false;
      Object.assign(save, { hidden: !name, href: url, download: name ?? '' });
      paint();
      const samples = await sound(url).catch(() => null);
      if (audio?.src !== new URL(url, location.href).href) return null;
      if (!samples) {
        player.clear(missing);
        return null;
      }
      seconds = samples.length / 48000;
      draw(peaks(samples, 64), false);
      paint();
      return seconds;
    },
    clear(text) {
      player.stop();
      audio = null;
      bars = [];
      play.disabled = true;
      save.hidden = true;
      wave.className = 'robin-wave empty';
      wave.replaceChildren();
      time.textContent = text;
    },
    live(levels, at) {
      player.stop();
      audio = null;
      play.disabled = true;
      draw(levels, true);
      time.textContent = `${clock(at)} / ${clock(MAX_TAKE)}`;
    },
    stop() {
      if (audio && playing?.audio === audio) stopPlaying();
    },
  };
  return player;
}

function takePanel(node) {
  const player = makePlayer('the last take', { download: true });
  player.clear('no take yet');
  addPanel(node, 'robin_take', h('div', { class: 'robin-panel' }, player.row), 58);
  return {
    executed(message) {
      const take = message?.robin_take?.[0];
      if (take) player.load(viewUrl(take), { length: take.seconds, name: `robin_take_seed${take.seed}.flac` });
    },
    stop: player.stop,
  };
}

// -> the name ComfyUI stored the file under, in its input folder.
async function toInput(blob, filename, fields = {}) {
  const body = new FormData();
  body.append('image', blob, filename);
  body.append('type', 'input');
  for (const [key, value] of Object.entries(fields)) body.append(key, value);
  const res = await api.fetchApi('/upload/image', { method: 'POST', body });
  if (!res.ok) throw new Error(res.status === 413 ? 'too large to upload' : `upload failed, ${res.status}`);
  return (await res.json()).name;
}

function wav(samples, rate) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (at, s) => [...s].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
  text(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, 'WAVEfmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => view.setInt16(44 + 2 * i, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
  return new Blob([view], { type: 'audio/wav' });
}

// Keeps the line: from 0.15 s before the first sound to 0.25 s after the last, as the trainer trims clips. -> null when nothing was heard.
function trimSilence(x, rate) {
  const n = rate / 100;
  const db = Array.from({ length: Math.floor(x.length / n) }, (_, k) => {
    let sum = 0;
    for (let i = k * n; i < (k + 1) * n; i++) sum += x[i] * x[i];
    return 10 * Math.log10(sum / n + 1e-18);
  });
  const loud = Math.max(-120, ...db);
  if (loud < -55) return null;
  const floor = Math.max(Math.min(-45, loud - 30), -60);
  const first = db.findIndex((d) => d > floor);
  const last = db.findLastIndex((d) => d > floor);
  return x.subarray(Math.max(0, first * n - Math.round(0.15 * rate)), Math.min(x.length, (last + 1) * n + Math.round(0.25 * rate)));
}

function stamp() {
  const d = new Date();
  const two = (v) => String(v).padStart(2, '0');
  return `${d.getFullYear()}${two(d.getMonth() + 1)}${two(d.getDate())}_${two(d.getHours())}${two(d.getMinutes())}${two(d.getSeconds())}`;
}

// Robin Load Audio: record a line with the microphone, or choose a file; either way it lands in ComfyUI's input folder.
function audioPanel(node) {
  const player = makePlayer('the recording');
  const record = h('button', { class: 'robin-badge', type: 'button' });
  const choose = h('button', { class: 'robin-badge', type: 'button' }, icon('folder'), 'Choose file');
  const reveal = h('button', { class: 'robin-badge', type: 'button', title: 'Open the folder your recordings are saved in' }, icon('open'), 'Open Save Dir');
  const note = h('div', { class: 'robin-hint' });
  const input = h('input', { type: 'file', accept: 'audio/*,video/*', hidden: true });
  addPanel(node, 'robin_audio', h('div', { class: 'robin-panel' }, player.row, h('div', { class: 'robin-pick' }, record, choose, reveal, input), note), 120);
  const list = widget(node, 'audio');
  let session = null;
  const say = (text, full = text) => {
    note.textContent = text;
    note.title = full;
  };
  const idle = () => say(`up to ${MAX_TAKE} s`, `Recordings and files up to ${MAX_TAKE} s, the Voice Changer's limit`);

  const label = (on) => {
    record.classList.toggle('on', on);
    record.replaceChildren(icon(on ? 'stop' : 'record'), on ? 'Stop' : 'Record');
  };
  const busy = (on) => {
    record.disabled = choose.disabled = on;
  };

  async function show() {
    const name = list?.value;
    if (!name) return player.clear('no recording yet');
    const seconds = await player.load(viewUrl({ filename: name, subfolder: '', type: 'input' }), { missing: `can't find ${name}` });
    if (seconds > MAX_TAKE) say(`${clock(seconds)}: over ${MAX_TAKE} s`, `The Voice Changer takes up to ${MAX_TAKE} s: trim this one first.`);
  }
  function use(name) {
    if (!list.options.values.includes(name)) list.options.values.push(name);
    list.value = name;
    idle();
    node.setDirtyCanvas(true, true);
    show();
  }

  async function add(file) {
    busy(true);
    say('adding');
    try {
      const plain = SOUND_FILES.test(file.name);
      use(await toInput(plain ? file : wav(await sound(file), 48000), plain ? file.name : `${file.name.replace(/\.[^.]+$/, '')}.wav`));
    } catch (e) {
      say("couldn't add it", `${file.name}: ${e.message}`);
    } finally {
      busy(false);
    }
  }

  async function finish(chunks, type) {
    try {
      const line = trimSilence(await sound(new Blob(chunks, { type })), 48000);
      if (line) use(await toInput(wav(line, 48000), `recording_${stamp()}.wav`));
      else {
        say('no sound came in', 'Nothing came through the microphone. Check which input the browser is using.');
        show();
      }
    } catch (e) {
      say("didn't save", `The recording didn't save: ${e.message}`);
      show();
    } finally {
      busy(false);
    }
  }

  async function start() {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      say('needs localhost or https', 'Browsers only allow recording on localhost or https. Open ComfyUI that way to record.');
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    } catch (e) {
      if (e.name === 'NotAllowedError') say('microphone blocked', 'The browser blocked the microphone. Allow it for this page and try again.');
      else say('no microphone', e.message);
      return;
    }
    stopPlaying();
    const recorder = new MediaRecorder(stream, { audioBitsPerSecond: 256000 });
    const chunks = [];
    const context = new AudioContext();
    const meter = context.createAnalyser();
    meter.fftSize = 2048;
    context.createMediaStreamSource(stream).connect(meter);
    const frame = new Float32Array(meter.fftSize);
    const levels = new Array(64).fill(0);
    const began = performance.now();
    let tick = 0;
    let last = 0;
    const draw = (now) => {
      const at = (now - began) / 1000;
      if (now - last >= 70) {
        meter.getFloatTimeDomainData(frame);
        let sum = 0;
        for (const v of frame) sum += v * v;
        levels.shift();
        levels.push(Math.min(1, Math.sqrt(4 * Math.sqrt(sum / frame.length))));
        last = now;
      }
      player.live(levels, at);
      if (at >= MAX_TAKE) session?.stop();
      else tick = requestAnimationFrame(draw);
    };
    const release = () => {
      cancelAnimationFrame(tick);
      stream.getTracks().forEach((track) => track.stop());
      context.close();
      session = null;
      label(false);
    };
    recorder.ondataavailable = (event) => event.data.size && chunks.push(event.data);
    recorder.onstop = () => finish(chunks, recorder.mimeType);
    session = {
      stop() {
        release();
        busy(true);
        say('saving');
        recorder.stop();
      },
      cancel() {
        recorder.onstop = null;
        release();
        recorder.stop();
      },
    };
    recorder.start(250);
    label(true);
    choose.disabled = true;
    say('say the line');
    tick = requestAnimationFrame(draw);
  }

  // A page can't open the file manager itself: ComfyUI opens its input folder when it runs on this computer, and otherwise says where it is.
  reveal.addEventListener('click', async () => {
    try {
      const res = await api.fetchApi('/robin/open_recordings', { method: 'POST' });
      const { opened, folder } = await res.json();
      if (opened) return say('opened the save folder', folder);
      const copied = await navigator.clipboard?.writeText(folder).then(() => true, () => false);
      say(`saved on the ComfyUI computer${copied ? ', path copied' : ''}`, `Recordings are saved in ${folder} on the computer running ComfyUI.`);
    } catch (e) {
      say("couldn't open it", e.message);
    }
  });
  record.addEventListener('click', () => (session ? session.stop() : start()));
  choose.addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    const file = input.files[0];
    input.value = '';
    if (file) add(file);
  });
  node.onDragOver = (event) => !!event?.dataTransfer?.types?.includes?.('Files');
  node.onDragDrop = (event) => {
    const file = event?.dataTransfer?.files?.[0];
    if (file) add(file);
    return !!file;
  };
  watch(node, ['audio'], () => {
    idle();
    show();
  });
  label(false);
  idle();
  setTimeout(show);

  return {
    stop() {
      session?.cancel();
      player.stop();
    },
  };
}

function inside(file) {
  return file.webkitRelativePath.split('/').slice(1).join('/');
}

function fingerprint(files) {
  const text = files.map((f) => `${inside(f)}|${f.size}|${f.lastModified}`).sort().join('\n');
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193);
  return (hash >>> 0).toString(16).padStart(8, '0');
}

async function send(file, name) {
  const path = inside(file);
  const subfolder = [UPLOADS, name, path.slice(0, Math.max(0, path.lastIndexOf('/')))].filter(Boolean).join('/');
  await toInput(file, file.name, { subfolder, overwrite: 'true' });
}

// A picked folder goes to input/robin_datasets/<folder name>-<fingerprint of its files>: picking it again sends only what is missing,
// and a different folder with the same name never mixes with it.
async function upload(files, progress) {
  const root = files[0].webkitRelativePath.split('/')[0];
  const name = `${root.replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^[._-]+/, '').slice(0, 48) || 'clips'}-${fingerprint(files)}`;
  const res = await api.fetchApi(`/robin/uploaded?${new URLSearchParams({ name })}`);
  const have = res.ok ? (await res.json()).files : {};
  const todo = files.filter((f) => have[inside(f)] !== f.size);
  const sending = todo.length;
  const failed = [];
  let done = files.length - todo.length;
  progress(done);
  const worker = async () => {
    for (let file; (file = todo.shift()); progress(++done)) {
      await send(file, name).catch((e) => failed.push(`${inside(file)} (${e.message})`));
    }
  };
  await Promise.all([worker(), worker(), worker(), worker()]);
  return { folder: `${UPLOADS}/${name}`, sent: sending - failed.length, failed };
}

function datasetPanel(node) {
  const input = h('input', { type: 'file', webkitdirectory: true, multiple: true, hidden: true });
  const pick = h('button', { class: 'robin-badge', type: 'button', onclick: () => input.click() }, icon('folder'), 'Choose folder');
  const note = h('span', {}, 'or type its path in folder');
  const out = h('pre', { 'aria-live': 'polite' });
  const term = h('div', { class: 'robin-term' }, h('header', {}, h('i'), h('i'), h('i'), h('span', {}, 'REPORT')), out);
  addPanel(node, 'robin_dataset', h('div', { class: 'robin-panel' }, h('div', { class: 'robin-pick' }, pick, note, input), term), 178, true);
  out.addEventListener('wheel', (event) => out.scrollHeight > out.clientHeight && event.stopPropagation(), { passive: true });

  const tone = (line) => (/^\s*ready:/.test(line) ? 'ok' : /^\s*left out:|^There is no folder|^Only .* minutes/.test(line) ? 'bad'
    : /^\s*note/.test(line) ? 'note' : /^\s*voice description|^Choose a folder|^Checking/.test(line) ? 'dim' : '');
  const show = (text) => out.replaceChildren(...String(text).split('\n').map((line) => h('span', { class: tone(line) }, line + '\n')));
  show(EMPTY_REPORT);

  let asked = 0;
  async function check() {
    const folder = String(value(node, 'folder') ?? '');
    const ask = ++asked;
    if (!folder.trim()) return show(EMPTY_REPORT);
    show('Checking the folder...');
    try {
      const res = await api.fetchApi('/robin/check', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ folder, language: value(node, 'language') }),
      });
      const text = res.ok ? (await res.json()).report : `The check failed (${res.status}).`;
      if (ask === asked) show(text);
    } catch (e) {
      if (ask === asked) show(`The check failed: ${e.message}`);
    }
  }
  watch(node, ['folder', 'language'], check);
  setTimeout(() => value(node, 'folder')?.trim() && check());

  input.addEventListener('change', async () => {
    const files = [...input.files].filter((f) => CLIP_FILES.test(f.name) && !inside(f).split('/').some((part) => part.startsWith('.')));
    input.value = '';
    if (!files.length) return show('That folder has no audio clips or .txt transcripts.');
    pick.disabled = true;
    try {
      const { folder, sent, failed } = await upload(files, (done) => (note.textContent = `uploading ${done} / ${files.length}`));
      note.textContent = failed.length ? `${failed.length} of ${files.length} files didn't upload` : sent ? `uploaded ${sent} files` : 'already uploaded';
      widget(node, 'folder').value = folder;
      node.setDirtyCanvas(true, true);
      await check();
      if (failed.length) out.append(h('span', { class: 'bad' }, `\nThese files didn't upload:\n${failed.map((f) => `  ${f}`).join('\n')}\n`));
    } catch (e) {
      note.textContent = 'the upload stopped';
      show(`The upload stopped: ${e.message}`);
    } finally {
      pick.disabled = false;
    }
  });

  return {
    executed(message) {
      if (message?.text) show(message.text.join('\n'));
    },
  };
}

function trainPanel(node) {
  const rows = {};
  const row = (key, label) => h('div', { class: 'row' }, h('span', {}, label), (rows[key] = h('span', {}, DASH)));
  const fill = h('i');
  const more = h('button', { class: 'robin-more', type: 'button' });
  const lcd = h('div', { class: 'robin-lcd' }, row('step', 'step'), row('planner', 'planner loss'), row('decoder', 'decoder loss'), row('pace', 'pace'),
    h('div', { class: 'meter' }, fill));
  const saved = h('div', { class: 'robin-saved' });
  addPanel(node, 'robin_training', h('div', { class: 'robin-panel' }, more, lcd, saved), 180);
  let run = { stage: 'idle', step: 0, samples: new Map() };

  const steps = () => run.total ?? value(node, 'steps');
  const lr = (name) => Number(value(node, name)).toExponential();
  const pace = () => ({
    idle: DASH,
    start: 'starting',
    prepare: `preparing clips · ${run.done} / ${run.todo}`,
    train: run.pace == null ? DASH : `${run.pace.toFixed(1)} s / step · ${span(run.left)} left`,
    done: `done in ${span(run.minutes * 60)}`,
    stopped: 'stopped · queue again to continue',
    failed: 'stopped · see the error',
  })[run.stage];

  function readout() {
    rows.step.textContent = `${run.stage === 'idle' ? DASH : run.step} / ${steps()}`;
    rows.planner.textContent = run.planner == null ? DASH : run.planner.toFixed(2);
    rows.decoder.textContent = run.decoder == null ? DASH : run.decoder.toFixed(3);
    rows.pace.textContent = pace();
    const share = run.stage === 'prepare' ? run.done / run.todo : run.stage === 'done' ? 1 : run.step / steps();
    fill.style.width = `${100 * Math.min(1, share || 0)}%`;
  }

  function mark(entry, on) {
    const label = entry.step ? String(entry.step) : 'base';
    entry.button.classList.toggle('on', on);
    entry.button.replaceChildren(icon(on ? 'pause' : 'play', 10), label);
    entry.button.setAttribute('aria-label', `${on ? 'Pause' : 'Play'} the ${entry.step ? `step ${entry.step}` : 'base'} sample`);
  }

  function chips() {
    const total = steps();
    const every = run.every ?? value(node, 'save_every');
    const plan = [0];
    for (let step = every; step < total && plan.length < 400; step += every) plan.push(step);
    plan.push(total);
    const waiting = plan.filter((step) => !run.samples.has(step)).slice(0, 12);
    const all = [...new Set([...waiting, ...run.samples.keys()])].sort((a, b) => a - b);
    saved.replaceChildren(h('b', {}, 'SAVED'), ...all.map((step) => {
      const entry = run.samples.get(step);
      if (!entry) return h('span', { class: 'robin-wait' }, step ? String(step) : 'base');
      entry.button = h('button', { class: 'robin-badge', type: 'button', onclick: () => toggle(entry.audio, (on) => mark(entry, on)) });
      mark(entry, playing?.audio === entry.audio);
      return entry.button;
    }));
  }

  function keep(sample) {
    run.samples.set(sample.step, { step: sample.step, audio: new Audio(viewUrl(sample)) });
    chips();
    const button = run.samples.get(sample.step).button;
    saved.scrollLeft = Math.max(0, button.offsetLeft + button.offsetWidth + 8 - saved.clientWidth);
  }

  function stop() {
    if ([...run.samples.values()].some((entry) => entry.audio === playing?.audio)) stopPlaying();
  }

  function label() {
    const open = !!node.showAdvanced;
    more.classList.toggle('open', open);
    more.setAttribute('aria-expanded', String(open));
    more.replaceChildren(icon('more', 10), open ? 'hide advanced'
      : `advanced · ranks ${value(node, 'rank_planner')} / ${value(node, 'rank_decoder')} · lr ${lr('lr_planner')} / ${lr('lr_decoder')}`);
  }
  more.addEventListener('click', () => {
    node.toggleAdvanced();
    if (!node.showAdvanced) node.setSize([node.size[0], node.computeSize()[1]]);
  });
  watch(node, ['steps', 'save_every'], () => run.stage === 'idle' && (readout(), chips()));
  watch(node, ['rank_planner', 'rank_decoder', 'lr_planner', 'lr_decoder'], label);
  setTimeout(() => (label(), readout(), chips()));

  return {
    event(detail) {
      if (detail.stage === 'start') {
        stop();
        run = { stage: 'start', step: 0, total: detail.total, every: detail.save_every, samples: new Map() };
        chips();
      } else if (detail.stage === 'prepare') Object.assign(run, { stage: 'prepare', done: detail.step, todo: detail.total });
      else if (detail.stage === 'train') {
        const { step, total, planner, decoder, pace, left } = detail;
        Object.assign(run, { stage: 'train', step, total, planner, decoder, pace, left });
      } else if (detail.stage === 'sample') keep(detail.sample);
      else if (detail.stage === 'done') Object.assign(run, { stage: 'done', step: detail.total, minutes: detail.minutes });
      else run.stage = detail.stage;
      readout();
    },
    executed(message) {
      const done = message?.robin_train?.[0];
      if (!done || run.stage === 'done') return;
      stop();
      run = { stage: 'done', step: done.total, total: done.total, minutes: done.minutes, samples: new Map() };
      for (const sample of done.samples) run.samples.set(sample.step, { step: sample.step, audio: new Audio(viewUrl(sample)) });
      readout();
      chips();
    },
    folded: label,
    stop,
  };
}

const PANELS = { RobinLoader: loaderPanel, RobinSpeak: takePanel, RobinVoiceChanger: takePanel, RobinLoadAudio: audioPanel, RobinCheckDataset: datasetPanel,
  RobinTrainLoRA: trainPanel };

// ComfyUI marks these inputs advanced but still lays them out; they stay folded away until Show Advanced (or the Train node's toggle).
function foldAdvanced(node) {
  const folded = (node.widgets ?? []).filter((w) => w.options?.advanced);
  if (!folded.length) return;
  let open = false;
  for (const w of folded) {
    w.advanced = w.hidden = true;
    w.getOutlineColor = () => '#8a2a25';
  }
  Object.defineProperty(node, 'showAdvanced', {
    configurable: true,
    get: () => open,
    set(value) {
      open = !!value;
      for (const w of folded) w.hidden = !open;
      node.robinUI?.folded?.();
      node.setDirtyCanvas(true, true);
    },
  });
}

function colorSlots(node) {
  for (const slot of [...(node.inputs ?? []), ...(node.outputs ?? [])]) {
    if (slot.type === 'AUDIO') slot.color_on = slot.color_off = CREAM;
  }
}

function pin(colors, type, color) {
  Object.defineProperty(colors, type, { get: () => color, set() {}, enumerable: true, configurable: true });
}

app.registerExtension({
  name: 'Robin.Nodes',

  setup() {
    // Robin's link colors hold through palette changes; dataset links are dashed and audio out of a Robin node is cream.
    const canvas = app.canvas;
    const LGraphCanvas = canvas.constructor;
    pin(LGraphCanvas.link_type_colors, 'ROBIN_MODEL', RED);
    pin(LGraphCanvas.link_type_colors, 'ROBIN_DATASET', AUDIO_LINK);
    pin(canvas.default_connection_color_byType, 'ROBIN_MODEL', RED);
    pin(canvas.default_connection_color_byType, 'ROBIN_DATASET', INK);
    const renderLink = LGraphCanvas.prototype.renderLink;
    LGraphCanvas.prototype.renderLink = function (ctx, a, b, link, skipBorder, flow, color, ...rest) {
      if (color == null && link?.type === 'AUDIO' && this.graph?.getNodeById(link.origin_id)?.type?.startsWith('Robin')) color = AUDIO_LINK;
      if (link?.type !== 'ROBIN_DATASET') return renderLink.call(this, ctx, a, b, link, skipBorder, flow, color, ...rest);
      ctx.save();
      ctx.setLineDash([7, 5]);
      try {
        return renderLink.call(this, ctx, a, b, link, skipBorder, flow, color, ...rest);
      } finally {
        ctx.restore();
      }
    };
    api.addEventListener('robin.loader', ({ detail }) => findNode(detail.node)?.robinUI?.loader?.(detail));
    api.addEventListener('robin.train', ({ detail }) => findNode(detail.node)?.robinUI?.event?.(detail));
  },

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name.startsWith('Robin')) return;
    const proto = nodeType.prototype;
    nodeType.title_text_color = CREAM;
    proto.onDrawTitleBar = drawTitleBar;
    proto.onDrawTitleBox = drawTitleBox;
    proto.onDrawTitleText = drawTitleText;
    chain(proto, 'onDrawForeground', drawBands);
    chain(proto, 'onNodeCreated', function () {
      this.color = RED;
      this.bgcolor = INK;
      colorSlots(this);
      for (const w of this.widgets ?? []) {
        if (w.element instanceof HTMLTextAreaElement) w.element.classList.add('robin-text');
      }
      foldAdvanced(this);
      this.robinUI = PANELS[nodeData.name]?.(this);
      const [width, room] = SIZES[nodeData.name] ?? [0, 0];
      this.setSize([Math.max(this.size[0], width), this.computeSize()[1] + room]);
    });
    chain(proto, 'onConfigure', function () {
      colorSlots(this);
    });
    chain(proto, 'onExecuted', function (message) {
      this.robinUI?.executed?.(message);
    });
    chain(proto, 'onRemoved', function () {
      this.robinUI?.stop?.();
    });
  },
});
