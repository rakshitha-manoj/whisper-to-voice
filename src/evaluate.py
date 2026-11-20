import csv
import glob
import os
import re
import warnings

import numpy as np

FS = 16000
FRAME = 320
HOP = 160
SYSTEMS_RAW = "raw_whisper"
NORMAL_REF = "normal_recording"
N_LISTENERS = 5
SHARED_IDS = ["01", "02"]
LISTENING_NOTE = (
    "Listening test notes\n"
    "====================\n"
    "Fill sheet.csv: one row per listener and item, scores 1 (worst) to 5 (best) for clarity and naturalness.\n"
    "Items are shuffled and blinded: item_XX.wav gives no hint of the system. Do not open key.csv before scoring.\n"
    "Participation is voluntary, listeners are told what the test is for and may stop at any time.\n"
    "Only the scores are stored, no names or personal data; use listener ids such as L1..L5.\n"
    "Audio contains only the project team's own recordings, used with the speakers' consent.\n"
    "After scoring run: python run_eval.py --aggregate-listening results/listening/sheet.csv\n"
)


def load_transcripts(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if m is None:
                warnings.warn(f"transcripts: cannot map line to a sentence id, ignored: {line!r}")
                continue
            out[f"{int(m.group(1)):02d}"] = m.group(2).strip()
    return out


def normalise_text(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def discover_clips(raw_dir="data/raw", shared_ids=None):
    shared_ids = list(SHARED_IDS if shared_ids is None else shared_ids)
    clips = []
    shared = os.path.join(raw_dir, "shared_test.wav")
    if os.path.isfile(shared):
        clips.append(dict(path=shared, speaker="shared", sentence_ids=shared_ids, normal_path=None))
    pattern = re.compile(r"^(?P<spk>[A-Za-z0-9]+)_s(?P<sid>\d+)_whisper\.wav$")
    for p in sorted(glob.glob(os.path.join(raw_dir, "*_whisper.wav"))):
        m = pattern.match(os.path.basename(p))
        if m is None:
            continue
        sid = f"{int(m.group('sid')):02d}"
        normal = os.path.join(raw_dir, f"{m.group('spk')}_s{m.group('sid')}_normal.wav")
        clips.append(dict(path=p, speaker=m.group("spk"), sentence_ids=[sid],
                          normal_path=normal if os.path.isfile(normal) else None))
    return clips


def clip_stem(clip):
    return os.path.splitext(os.path.basename(clip["path"]))[0]


def clip_reference(clip, transcripts):
    return " ".join(transcripts[i] for i in clip["sentence_ids"] if i in transcripts)


def load_asr(model_name="base"):
    import whisper
    return whisper.load_model(model_name, device="cpu")


def transcribe(model, x16k):
    x = np.ascontiguousarray(x16k, dtype=np.float32)
    res = model.transcribe(x, language="en", temperature=0.0, fp16=False,
                           condition_on_previous_text=False, verbose=None)
    return res["text"].strip()


def wer_counts(reference, hypothesis):
    import jiwer
    ref = normalise_text(reference)
    hyp = normalise_text(hypothesis)
    n_ref = len(ref.split())
    if n_ref == 0:
        return 0, 0
    if not hyp:
        return n_ref, n_ref
    out = jiwer.process_words(ref, hyp)
    return out.substitutions + out.deletions + out.insertions, n_ref


def wer_for(model, x, reference):
    hyp = transcribe(model, x)
    errors, n_ref = wer_counts(reference, hyp)
    return (errors / n_ref if n_ref else 0.0), hyp


def _lpc_frames(x, order):
    from src.framing import frame_signal, frame_energy
    from src.lpc import lpc_all
    frames = frame_signal(np.asarray(x, dtype=np.float32), FRAME, HOP)
    energy = frame_energy(frames)
    A, G = lpc_all(frames, order=order)
    return frames, energy, A, G


def lsd_variant(x_in, y_out, order=12, mask=None):
    from src.detect import classify_frames
    from src.lpc import lsd_frames
    frames, energy, A_in, G_in = _lpc_frames(x_in, order)
    _, _, A_out, G_out = _lpc_frames(y_out, order)
    n = min(len(A_in), len(A_out))
    labels = np.asarray(classify_frames(frames, energy))[:n]
    sel = labels != 0
    if mask is not None:
        m = np.zeros(n, dtype=bool)
        mm = np.asarray(mask)
        if mm.dtype == bool:
            m[:min(n, len(mm))] = mm[:n]
        else:
            m[mm[mm < n]] = True
        sel = sel & m
    return lsd_frames(A_in[:n], G_in[:n], A_out[:n], G_out[:n], mask=sel)


def pitch_stats(x, fmin=70, fmax=400):
    import librosa
    f0, voiced, _ = librosa.pyin(np.asarray(x, dtype=np.float32), fmin=fmin, fmax=fmax,
                                 sr=FS, frame_length=1024, hop_length=HOP)
    keep = np.asarray(voiced) & np.isfinite(f0)
    return np.asarray(f0)[keep]


def _write_wav(path, y):
    import soundfile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    soundfile.write(path, y, FS)


def _cell(v):
    return "" if v is None else v


def _load(path):
    from src.framing import load_audio
    x, _ = load_audio(path)
    return x


def run_ablation_eval(clips, results_dir="results", asr=True, model_name="base",
                      transcripts=None, transcripts_path="data/raw/transcripts.txt",
                      model=None, dev=False):
    from src.synth import VARIANTS, _features, render_variant
    os.makedirs(results_dir, exist_ok=True)
    if transcripts is None:
        transcripts = load_transcripts(transcripts_path) if os.path.isfile(transcripts_path) else {}
    if asr and model is None:
        model = load_asr(model_name)
    systems = [SYSTEMS_RAW] + list(VARIANTS)
    rows = []
    for clip in clips:
        stem = clip_stem(clip)
        ref = clip_reference(clip, transcripts)
        feats = _features(clip["path"])
        x = _load(clip["path"])
        outputs = {SYSTEMS_RAW: x}
        for name in VARIANTS:
            y = render_variant(feats, name)
            _write_wav(os.path.join(results_dir, f"{stem}_{name}.wav"), y)
            outputs[name] = y
        for system in systems:
            y = outputs[system]
            row = dict(clip=stem, speaker=clip["speaker"], system=system,
                       lsd=0.0 if system == SYSTEMS_RAW else lsd_variant(x, y),
                       wer=None, errors=None, n_ref=None, hypothesis="")
            if asr and ref:
                hyp = transcribe(model, y)
                errors, n_ref = wer_counts(ref, hyp)
                row.update(wer=errors / n_ref if n_ref else None, errors=errors, n_ref=n_ref, hypothesis=hyp)
            rows.append(row)
        if asr and ref and clip.get("normal_path"):
            hyp = transcribe(model, _load(clip["normal_path"]))
            errors, n_ref = wer_counts(ref, hyp)
            rows.append(dict(clip=stem, speaker=clip["speaker"], system=NORMAL_REF, lsd=None,
                             wer=errors / n_ref if n_ref else None, errors=errors, n_ref=n_ref, hypothesis=hyp))
    agg = aggregate_rows(rows)
    save_ablation(rows, agg, results_dir, dev=dev)
    return rows, agg


def aggregate_rows(rows):
    systems = []
    for r in rows:
        if r["system"] not in systems:
            systems.append(r["system"])
    agg = []
    for s in systems:
        rs = [r for r in rows if r["system"] == s]
        wers = [r["wer"] for r in rs if r["wer"] is not None]
        lsds = [r["lsd"] for r in rs if r["lsd"] is not None]
        errs = [r["errors"] for r in rs if r["errors"] is not None]
        refs = [r["n_ref"] for r in rs if r["n_ref"] is not None]
        agg.append(dict(system=s,
                        wer_mean=float(np.mean(wers)) if wers else None,
                        wer_corpus=(sum(errs) / sum(refs)) if refs and sum(refs) else None,
                        lsd_mean=float(np.mean(lsds)) if lsds else None,
                        n_clips=len(rs)))
    return agg


def _fmt(v, nd=3):
    return "" if v is None else f"{v:.{nd}f}"


def read_listening_scores(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["system"]] = r
    return out


def ablation_markdown(agg, listening=None, dev=False):
    listening = listening or {}
    lines = []
    if dev:
        lines.append("SYNTHETIC DEV DATA (SAPI TTS speech, whisperised by LPC plus noise): pipeline check only, not results.")
        lines.append("")
    lines.append("| System variant | WER | LSD (dB) | Listening clarity | Listening naturalness |")
    lines.append("|---|---|---|---|---|")
    for a in agg:
        ls = listening.get(a["system"], {})
        wer = "" if a["wer_mean"] is None else f"{a['wer_mean']:.3f} (corpus {a['wer_corpus']:.3f})"
        lines.append(f"| {a['system']} | {wer} | {'0 (reference)' if a['system'] == SYSTEMS_RAW else _fmt(a['lsd_mean'], 2)} | "
                     f"{ls.get('clarity', '')} | {ls.get('naturalness', '')} |")
    return "\n".join(lines) + "\n"


def save_ablation(rows, agg, results_dir, dev=False):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "ablation_per_clip.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "speaker", "system", "wer", "errors", "n_ref", "lsd_db", "hypothesis"])
        for r in rows:
            w.writerow([r["clip"], r["speaker"], r["system"], _fmt(r["wer"], 4), _cell(r["errors"]),
                        _cell(r["n_ref"]), _fmt(r["lsd"], 3), r["hypothesis"]])
    with open(os.path.join(results_dir, "ablation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "wer_mean", "wer_corpus", "lsd_db_mean", "n_clips"])
        for a in agg:
            w.writerow([a["system"], _fmt(a["wer_mean"], 4), _fmt(a["wer_corpus"], 4), _fmt(a["lsd_mean"], 3), a["n_clips"]])
    listening = read_listening_scores(os.path.join(results_dir, "listening_scores.csv"))
    md = ablation_markdown(agg, listening, dev=dev)
    with open(os.path.join(results_dir, "ablation.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


def make_listening_sheet(clips, results_dir="results", seed=0, n_listeners=N_LISTENERS):
    from src.synth import VARIANTS, _features, render_variant
    out_dir = os.path.join(results_dir, "listening")
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for clip in clips:
        stem = clip_stem(clip)
        feats = _features(clip["path"])
        entries.append((stem, SYSTEMS_RAW, _load(clip["path"])))
        for name in VARIANTS:
            entries.append((stem, name, render_variant(feats, name)))
    order = np.random.default_rng(seed).permutation(len(entries))
    key = []
    for k, idx in enumerate(order, start=1):
        stem, system, y = entries[int(idx)]
        item = f"item_{k:02d}"
        _write_wav(os.path.join(out_dir, f"{item}.wav"), y)
        key.append((item, stem, system))
    with open(os.path.join(out_dir, "key.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "clip", "system"])
        w.writerows(key)
    with open(os.path.join(out_dir, "sheet.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["listener", "item", "clarity_1to5", "naturalness_1to5"])
        for li in range(1, n_listeners + 1):
            for item, _, _ in key:
                w.writerow([f"L{li}", item, "", ""])
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(LISTENING_NOTE)
    return out_dir


def aggregate_listening(sheet, key, out_path=None):
    with open(key, newline="", encoding="utf-8") as f:
        item_system = {r["item"]: r["system"] for r in csv.DictReader(f)}
    acc = {}
    with open(sheet, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                c = float(r["clarity_1to5"])
                n = float(r["naturalness_1to5"])
            except (ValueError, TypeError, KeyError):
                continue
            s = item_system.get(r["item"])
            if s is None:
                continue
            acc.setdefault(s, []).append((c, n))
    out = {}
    for s, v in acc.items():
        arr = np.array(v)
        out[s] = dict(system=s, clarity=round(float(arr[:, 0].mean()), 3),
                      naturalness=round(float(arr[:, 1].mean()), 3), n=len(v))
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sheet))), "listening_scores.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "clarity", "naturalness", "n"])
        for s in out.values():
            w.writerow([s["system"], s["clarity"], s["naturalness"], s["n"]])
    return out


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    _plt().close(fig)
    return path


def _labels_for(x):
    from src.framing import frame_signal, frame_energy
    from src.detect import classify_frames
    frames = frame_signal(np.asarray(x, dtype=np.float32), FRAME, HOP)
    energy = frame_energy(frames)
    return frames, energy, np.asarray(classify_frames(frames, energy))


def _spec_db(x, n_fft=512, hop=HOP):
    """Hann-window STFT magnitude in dB (25 ms window zero padded to n_fft, 10 ms hop)."""
    import scipy.signal
    f, t, Z = scipy.signal.stft(np.asarray(x, dtype=np.float64), fs=FS, window="hann", nperseg=400,
                                noverlap=400 - hop, nfft=n_fft, boundary=None, padded=False)
    return f, t + 200 / FS, 20 * np.log10(np.abs(Z) + 1e-10)


def _spec_plot(ax, spec, vmax, fmax=4000, dyn_db=70):
    """Draw a spectrogram on a shared dB scale so input and output are comparable."""
    f, t, S = spec
    keep = f <= fmax
    ax.pcolormesh(t, f[keep], S[keep], shading="auto", cmap="magma", vmin=vmax - dyn_db, vmax=vmax)
    ax.set_ylim(0, fmax)


def fig_waveform_spectrogram(x_in, y_out, path, suffix=""):
    plt = _plt()
    _, _, labels = _labels_for(x_in)
    s_in, s_out = _spec_db(x_in), _spec_db(y_out)
    vmax = max(s_in[2].max(), s_out[2].max())
    fig, ax = plt.subplots(5, 1, figsize=(8, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1.4, 0.3, 1, 1.4]))
    ax[0].plot(np.arange(len(x_in)) / FS, x_in, lw=0.4)
    ax[0].set_ylabel("whisper")
    _spec_plot(ax[1], s_in, vmax)
    ax[1].set_ylabel("Hz")
    ax[2].step(np.arange(len(labels)) * HOP / FS, labels, where="post", lw=0.8)
    ax[2].set_yticks([0, 1, 2])
    ax[2].set_ylabel("label")
    ax[3].plot(np.arange(len(y_out)) / FS, y_out, lw=0.4, color="C1")
    ax[3].set_ylabel("output")
    _spec_plot(ax[4], s_out, vmax)
    ax[4].set_ylabel("Hz")
    ax[4].set_xlabel("time (s)")
    ax[0].set_title("Whispered input vs voiced output" + suffix, fontsize=9)
    return _save(fig, path)


def fig_envelope_overlay(x_in, y_out, path, order=12, warp=0.92, suffix=""):
    from src.lpc import lpc_frame, lpc_envelope, preemphasis, warp_poles, _hamming
    plt = _plt()
    fr_in, energy, labels = _labels_for(x_in)
    fr_out, _, _ = _labels_for(y_out)
    cand = np.flatnonzero(labels == 2)
    i = int(cand[np.argmax(energy[cand])]) if len(cand) else int(np.argmax(energy))
    i = min(i, len(fr_out) - 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for k, (fr, title) in enumerate(((fr_in[i], "input (whisper)"), (fr_out[i], "output"))):
        a, g = lpc_frame(fr, order)
        w = _hamming(len(fr))
        spec = 20 * np.log10(np.abs(np.fft.rfft(preemphasis(np.asarray(fr, dtype=np.float64)) * w, 512)) + 1e-12)
        f, env = lpc_envelope(a, g)
        ax[k].plot(f, spec, lw=0.6, color="0.6", label="FFT magnitude")
        ax[k].plot(f, env, lw=1.5, label="LPC envelope")
        if k == 0 and warp is not None:
            _, env_w = lpc_envelope(warp_poles(a, warp), g)
            ax[k].plot(f, env_w, lw=1.2, ls="--", label=f"warped {warp}")
        ax[k].set_title(f"{title}, frame {i}", fontsize=9)
        ax[k].set_xlabel("Hz")
        ax[k].legend(fontsize=7)
    ax[0].set_ylabel("dB")
    fig.suptitle("Vowel frame spectrum and LPC envelope" + suffix, fontsize=9)
    return _save(fig, path)


def fig_f0(x_in, y_out, path, normal=None, suffix=""):
    from src.pitch import build_f0
    plt = _plt()
    _, energy, labels = _labels_for(x_in)
    f0 = build_f0(labels, energy)
    t = np.arange(len(labels)) * HOP / FS
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw=dict(width_ratios=[2, 1]))
    ax[0].plot(t, np.where(f0 > 0, f0, np.nan), lw=1.2)
    ax[0].set_ylabel("Hz")
    ax[0].set_xlabel("time (s)")
    ax2 = ax[0].twinx()
    ax2.step(t, labels, where="post", color="0.6", lw=0.7)
    ax2.set_yticks([0, 1, 2])
    ax2.set_ylabel("label")
    ax[0].set_title("Built f0 contour over label track" + suffix, fontsize=9)
    bins = np.arange(50, 410, 10)
    ax[1].hist(pitch_stats(y_out), bins=bins, alpha=0.6, density=True, label="our output (pyin)")
    if normal is not None and len(normal):
        ax[1].hist(normal, bins=bins, alpha=0.6, density=True, label="normal recording (pyin)")
    ax[1].set_xlabel("f0 (Hz)")
    ax[1].legend(fontsize=7)
    ax[1].set_title("f0 histogram", fontsize=9)
    return _save(fig, path)


def fig_ablation_bar(agg, path, suffix=""):
    plt = _plt()
    names = [a["system"] for a in agg]
    wer = [np.nan if a["wer_mean"] is None else a["wer_mean"] for a in agg]
    lsd = [np.nan if a["lsd_mean"] is None else a["lsd_mean"] for a in agg]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].bar(range(len(names)), wer)
    ax[0].set_title("WER (mean per clip)" + suffix, fontsize=9)
    ax[1].bar(range(len(names)), lsd, color="C1")
    ax[1].set_title("LSD vs input (dB)", fontsize=9)
    for a_ in ax:
        a_.set_xticks(range(len(names)))
        a_.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    return _save(fig, path)


def make_figures(clips, agg, results_dir="results", suffix=""):
    from src.synth import _features, render_variant
    fig_dir = os.path.join(results_dir, "figures")
    paths = [fig_ablation_bar(agg, os.path.join(fig_dir, "fig_ablation_bar.png"), suffix)]
    if not clips:
        return paths
    clip = clips[0]
    x = _load(clip["path"])
    y = render_variant(_features(clip["path"]), "v4_warp")
    normal = pitch_stats(_load(clip["normal_path"])) if clip.get("normal_path") else None
    paths.append(fig_waveform_spectrogram(x, y, os.path.join(fig_dir, "fig_waveform_spectrogram.png"), suffix))
    paths.append(fig_envelope_overlay(x, y, os.path.join(fig_dir, "fig_envelope_overlay.png"), suffix=suffix))
    paths.append(fig_f0(x, y, os.path.join(fig_dir, "fig_f0.png"), normal=normal, suffix=suffix))
    return paths
