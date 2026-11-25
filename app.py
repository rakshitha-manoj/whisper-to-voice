"""Gradio demo: whisper in, voiced speech out. Tabs: Convert, Analysis (modules A and B), Results."""

import glob
import os
import tempfile
import threading

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SHARED_CLIP = os.path.join(ROOT, "data", "raw", "shared_test.wav")
FS = 16000
HOP = 160
VARIANT_CHOICES = ["custom", "v1_robot", "v2_contour", "v3_aspiration", "v4_warp"]
NO_RESULTS = "No results found. Run `python run_eval.py` (or `python run_eval.py --dev`) first."
TOO_SHORT = "Clip is shorter than one 20 ms frame (320 samples at 16 kHz); record or upload a longer clip."
LIVE_NOTE = ("Live mode runs from the command line: `python run_live.py --list-devices`, "
             "then `python run_live.py --in N --out M`.")
ASR_MODELS = ["tiny", "base", "small"]
ASR_NOTE = ("Transcription uses the OpenAI Whisper speech recognition model (ASR) on the CPU. "
            "This is not whispered speech: it is just the name of the recogniser. The first use of a model "
            "downloads it, and each clip then takes a few seconds. The output transcript is of the audio "
            "currently in the voiced output player; if Convert has not been pressed for this clip, the output "
            "is rendered from the current settings instead.")
NO_WHISPER = "openai-whisper is not installed. Run `pip install openai-whisper` and restart the app."
_CACHE = {}
_ASR_CACHE = {}
_ASR_LOCK = threading.RLock()


def demo_clip_path():
    """Path of the shared test recording if present, else a generated pseudo whisper wav."""
    if os.path.exists(SHARED_CLIP):
        return SHARED_CLIP
    from tests.fixtures import write_fixture
    path = os.path.join(tempfile.gettempdir(), "w2v_demo_pseudo_whisper.wav")
    write_fixture(path)
    return path


def get_features(path):
    """Compute framing, labels and LPC once per file."""
    from src.synth import _features
    key = (path, os.path.getmtime(path))
    feats = _CACHE.get(key)
    if feats is None:
        feats = _features(path)
        _CACHE.clear()
        _CACHE[key] = feats
    return feats


def render(path, variant="custom", base_f0=120, decl=0.12, stress=0.25, aspiration=0.15, warp=1.0, volume=1.0):
    from src.synth import _render, render_variant
    feats = get_features(path)
    if len(feats["labels"]) == 0:
        raise ValueError(TOO_SHORT)
    if variant in VARIANT_CHOICES[1:]:
        y = render_variant(feats, variant, volume=volume, base_f0=base_f0)
    else:
        w = None if warp is None or warp >= 1.0 else float(warp)
        y = _render(feats, volume=volume, base_f0=base_f0, decl=decl, stress=stress, aspiration=aspiration, warp=w)
    return y, feats


def get_asr(model_name="base"):
    """Whisper ASR model, loaded once per name. The lock stops two workers loading (or downloading) it together."""
    if model_name not in ASR_MODELS:
        raise ValueError(f"Unknown model {model_name!r}; choose one of {', '.join(ASR_MODELS)}.")
    with _ASR_LOCK:
        model = _ASR_CACHE.get(model_name)
        if model is None:
            from src import evaluate
            try:
                model = evaluate.load_asr(model_name)
            except ImportError as e:
                raise ValueError(NO_WHISPER) from e
            _ASR_CACHE[model_name] = model
        return model


def output_signal(out_audio):
    """Float 16 kHz signal from the Gradio output player value (fs, array); int16 is scaled to +-1."""
    fs, y = out_audio
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / float(np.iinfo(y.dtype).max + 1)
    if int(fs) != FS:
        raise ValueError(f"Output audio is {fs} Hz; expected {FS} Hz. Press Convert again.")
    return y.astype(np.float32)


def transcribe_clip(audio_path, model_name="base", reference="", out_audio=None, shown_input=None,
                    variant="custom", base_f0=120, decl=0.12, stress=0.25, aspiration=0.15, warp=1.0, volume=1.0):
    """Transcribe the whisper input and the voiced output. Returns dict(input, output, wer) (wer None without a reference).

    The output is exactly what the output player holds (so it matches what the user hears) when that player was
    filled from this clip. Otherwise (no Convert yet, or the clip changed since) it is rendered now from the
    current variant and sliders.
    """
    if not audio_path:
        raise ValueError("Upload or record a clip first, or press Load demo clip.")
    from src import evaluate
    from src.framing import load_audio
    x, _ = load_audio(audio_path)
    if out_audio is not None and shown_input == audio_path:
        y = output_signal(out_audio)
    else:
        y, _ = render(audio_path, variant, base_f0, decl, stress, aspiration, warp, volume)
    model = get_asr(model_name)  # after the clip checks, so a bad clip fails before any model load or download
    with _ASR_LOCK:
        text_in = evaluate.transcribe(model, x)
        text_out = evaluate.transcribe(model, y)
    wer = None
    if reference and reference.strip():
        wer = {}
        for key, text in (("input", text_in), ("output", text_out)):
            errors, n_ref = evaluate.wer_counts(reference, text)
            wer[key] = (errors, n_ref)
    return dict(input=text_in, output=text_out, wer=wer)


def transcript_markdown(res):
    """Side by side table of the two transcripts, with WER rows when a reference was given."""
    def cell(t):
        return (t or "(nothing recognised)").replace("|", "/").replace("\n", " ")
    lines = ["| | Input (whisper) | Voiced output |", "|---|---|---|",
             f"| Transcript | {cell(res['input'])} | {cell(res['output'])} |"]
    if res["wer"]:
        def w(key):
            e, n = res["wer"][key]
            return f"{e}/{n} = {100.0 * e / n:.1f}%" if n else "n/a"
        lines.append(f"| WER (errors/reference words) | {w('input')} | {w('output')} |")
    return "\n".join(lines)


def transcribe_ui(audio_path, model_name, reference, out_audio, shown_input, *render_args):
    return transcript_markdown(transcribe_clip(audio_path, model_name, reference, out_audio, shown_input, *render_args))


def spectrogram_figure(x, y, fs=FS):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 3.6))
    axes = fig.subplots(2, 1, sharex=True)
    for ax, sig, title in zip(axes, (x, y), ("Input (whisper)", "Output (voiced)")):
        sig = np.asarray(sig, dtype=np.float64) + 1e-7 * np.random.default_rng(0).standard_normal(len(sig))
        ax.specgram(sig, NFFT=512, Fs=fs, noverlap=384, cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Hz")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def f0_params(variant, decl, stress):
    """(decl, stress, jitter) that render() uses: named variants ignore the decl and stress sliders."""
    from src.synth import VARIANTS
    if variant not in VARIANTS:
        return decl, stress, 0.005
    v = VARIANTS[variant]
    return v.get("decl", 0.12), v.get("stress", 0.25), v.get("jitter", 0.005)


def label_f0_figure(feats, base_f0=120, decl=0.12, stress=0.25, jitter=0.005):
    from matplotlib.figure import Figure
    from src.pitch import build_f0
    labels, energy = feats["labels"], feats["energy"]
    t = np.arange(len(labels)) * HOP / FS
    f0 = build_f0(labels, energy, base_f0=base_f0, decl=decl, stress=stress, jitter=jitter)
    fig = Figure(figsize=(7, 3))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.step(t, labels, where="post")
    ax1.set_yticks([0, 1, 2], ["silence", "consonant", "vowel-like"])
    ax1.set_title("Frame labels", fontsize=9)
    ax2.plot(t, np.where(f0 > 0, f0, np.nan))
    ax2.set_ylabel("f0 (Hz)")
    ax2.set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def convert_clip(audio_path, variant, base_f0, decl, stress, aspiration, warp, volume):
    if not audio_path:
        raise ValueError("Upload or record a clip first, or press Load demo clip.")
    from src.framing import load_audio
    y, feats = render(audio_path, variant, base_f0, decl, stress, aspiration, warp, volume)
    x, _ = load_audio(audio_path)
    n = min(len(x), len(y))
    return (FS, y), spectrogram_figure(x[:n], y[:n]), label_f0_figure(feats, base_f0, *f0_params(variant, decl, stress))


def save_wav_temp(y, name="whisper_to_voice_output.wav", fs=FS):
    """Write y to a fresh temp dir and return the path (for download)."""
    import soundfile
    path = os.path.join(tempfile.mkdtemp(prefix="w2v_"), name)
    soundfile.write(path, np.asarray(y, dtype=np.float32), fs)
    return path


def convert_full(audio_path, variant, base_f0, decl, stress, aspiration, warp, volume):
    """convert_clip plus the input player value and a downloadable wav path."""
    out, spec, lab = convert_clip(audio_path, variant, base_f0, decl, stress, aspiration, warp, volume)
    return out, spec, lab, audio_path, save_wav_temp(out[1])


def compare_variants(audio_path, base_f0, decl, stress, aspiration, warp, volume):
    """Render custom (current sliders) and the four named variants. Returns [(label, (fs, y)), ...]."""
    if not audio_path:
        raise ValueError("Upload or record a clip first, or press Load demo clip.")
    results = []
    for variant in VARIANT_CHOICES:
        y, _ = render(audio_path, variant, base_f0, decl, stress, aspiration, warp, volume)
        label = "custom (sliders)" if variant == "custom" else variant
        results.append((label, (FS, y)))
    return results


def render_settings(variant, base_f0, decl, stress, aspiration, warp):
    """Every parameter render() ends up using for this variant and sliders, as one dict (warp None = off)."""
    from src.synth import VARIANTS
    if variant in VARIANTS:
        v = VARIANTS[variant]
        return dict(base_f0=base_f0, decl=v.get("decl", 0.12), stress=v.get("stress", 0.25),
                    jitter=v.get("jitter", 0.005), aspiration=v.get("aspiration", 0.15), warp=v.get("warp"))
    w = None if warp is None or warp >= 1.0 else float(warp)
    return dict(base_f0=base_f0, decl=decl, stress=stress, jitter=0.005, aspiration=aspiration, warp=w)


def pole_frequencies(a, fs=FS, max_bw=400.0, min_freq=90.0, n_max=3):
    """Formant candidates from LPC coefficients: lowest n_max pole frequencies (Hz) that pass the filters.

    Frequency is the pole angle (angle * fs / 2 pi), bandwidth is -fs / pi * ln(radius). Only poles in the upper
    half plane (positive frequency) with bandwidth below max_bw and frequency above min_freq are kept.
    """
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 2 or not np.any(a[1:]):
        return np.zeros(0)
    z = np.roots(a)
    z = z[(z.imag > 1e-9) & (np.abs(z) > 0)]
    freq = np.angle(z) * fs / (2 * np.pi)
    bw = -fs / np.pi * np.log(np.abs(z))
    keep = (bw >= 0) & (bw < max_bw) & (freq > min_freq)
    return np.sort(freq[keep])[:n_max]


def formant_tracks(A, labels, warp=None, fs=FS):
    """Per frame formant list for non-silent frames (empty array for silence). warp None or 1.0 uses A as is."""
    from src.lpc import warp_poles
    out = []
    for a, lab in zip(A, labels):
        if lab == 0:
            out.append(np.zeros(0))
            continue
        if warp is not None and warp < 1.0:
            a = warp_poles(a, float(warp))
        out.append(pole_frequencies(a, fs))
    return out


def pole_figure(af, idx, warp=1.0, order=12, fs=FS):
    """Poles of the LPC filter of one frame on the z plane, with the unit circle and 1, 2, 4 kHz angle ticks."""
    from matplotlib.figure import Figure
    from src.lpc import lpc_frame, warp_poles
    frames = af["frames"]
    if len(frames) == 0:
        raise ValueError(TOO_SHORT)
    idx = int(min(max(idx, 0), len(frames) - 1))
    a, _ = lpc_frame(frames[idx], order)
    fig = Figure(figsize=(4.6, 4.6))
    ax = fig.subplots()
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="0.5", lw=1)
    for f in (1000, 2000, 4000):
        ang = 2 * np.pi * f / fs
        ax.plot([0, np.cos(ang)], [0, np.sin(ang)], color="0.8", lw=0.8, ls=":")
        ax.annotate(f"{f // 1000} kHz", (1.09 * np.cos(ang), 1.09 * np.sin(ang)), fontsize=7, ha="center", va="center")
    ax.annotate("0 Hz", (1.12, 0), fontsize=7, ha="center", va="center")
    ax.annotate("8 kHz", (-1.14, 0), fontsize=7, ha="center", va="center")
    z = np.roots(a) if np.any(a[1:]) else np.zeros(0)
    ax.plot(z.real, z.imag, "o", color="C0", label="LPC poles")
    if warp < 1.0:
        # a silent frame (a = [1, 0, ...]) has no poles; np.roots would otherwise report 12 at the origin
        zw = np.roots(warp_poles(a, float(warp))) if len(z) else np.zeros(0)
        ax.plot(zw.real, zw.imag, "x", color="C1", ms=7, label=f"warped poles ({warp:.2f})")
    ax.axhline(0, color="0.85", lw=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title(f"Poles of frame {idx} (pole angle = frequency)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def formant_figure(x, A, labels, warp=1.0, idx=None, fs=FS, hop=HOP):
    """F1 to F3 candidates (lowest three narrow-band LPC poles) over the input spectrogram, original and warped, for non-silent frames."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 3.6))
    ax = fig.subplots()
    sig = np.asarray(x, dtype=np.float64) + 1e-7 * np.random.default_rng(0).standard_normal(len(x))
    ax.specgram(sig, NFFT=512, Fs=fs, noverlap=384, cmap="magma")
    t = (np.arange(len(labels)) * hop + 160) / fs
    series = [(formant_tracks(A, labels), "o", "white", "original poles")]
    if warp < 1.0:
        series.append((formant_tracks(A, labels, warp), "^", "cyan", f"warped ({warp:.2f})"))
    for tracks, marker, color, name in series:
        tt = np.concatenate([np.full(len(f), t[i]) for i, f in enumerate(tracks)] + [np.zeros(0)])
        ff = np.concatenate(list(tracks) + [np.zeros(0)])
        ax.plot(tt, ff, marker, color=color, ms=2.5, ls="none", label=name)
    if idx is not None and len(t):
        ax.axvline(t[int(min(max(idx, 0), len(t) - 1))], color="yellow", lw=0.8, ls="--")
    ax.set_ylim(0, 6000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("F1-F3 candidates (lowest 3 narrow-band LPC poles) over input spectrogram", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def module_b_pole_figures(path, idx, warp):
    """(pole figure, formant figure) for the clip, from the cached analysis and conversion features."""
    af = analysis_features(path)
    feats = get_features(path)
    if len(af["frames"]) == 0:
        raise ValueError(TOO_SHORT)
    return pole_figure(af, idx, warp), formant_figure(af["x"], feats["A"], feats["labels"], warp, idx)


def voiced_window(labels, hop=HOP, dur_s=0.07, fs=FS):
    """(start_frame, n_frames) of a window of about dur_s centred in the longest voiced run, or None."""
    lab = np.asarray(labels) == 2
    if not lab.any():
        return None
    d = np.diff(np.concatenate(([0], lab.astype(int), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = int(np.argmax(ends - starts))
    s, e = int(starts[k]), int(ends[k])
    n = max(1, int(round(dur_s * fs / hop)))
    n = min(n, e - s)
    return s + (e - s - n) // 2, n


def _message_figure(text, figsize=(7, 2.5)):
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=10)
    return fig


def excitation_figure(feats, settings):
    """About 70 ms of voiced excitation, pulses only against pulses plus aspiration, from the same f0 as the render."""
    from matplotlib.figure import Figure
    from src.pitch import build_f0, make_excitation
    labels, energy = feats["labels"], feats["energy"]
    win = voiced_window(labels)
    if win is None:
        return _message_figure("No voiced frames in this clip, so there is no voiced excitation to show.")
    s = settings
    f0 = build_f0(labels, energy, base_f0=s["base_f0"], decl=s["decl"], stress=s["stress"], jitter=s["jitter"])
    e_plain = make_excitation(labels, f0, aspiration=0.0)
    e_asp = make_excitation(labels, f0, aspiration=s["aspiration"])
    a, n = win
    lo, hi = a * HOP, min((a + n) * HOP, len(e_plain))
    t = np.arange(lo, hi) / FS * 1000.0
    fig = Figure(figsize=(7, 3.6))
    ax1, ax2 = fig.subplots(2, 1, sharex=True, sharey=True)
    ax1.plot(t, e_plain[lo:hi], lw=0.8)
    ax1.set_title("Rosenberg pulses only (aspiration = 0)", fontsize=9)
    ax2.plot(t, e_asp[lo:hi], lw=0.8, color="C1")
    ax2.set_title(f"Pulses plus aspiration noise (aspiration = {s['aspiration']:.2f})", fontsize=9)
    for ax in (ax1, ax2):
        ax.set_ylabel("Amplitude")
    ax2.set_xlabel("Time (ms)")
    fig.suptitle("Excitation in a voiced region", fontsize=10)
    fig.tight_layout()
    return fig


def f0_components(feats, base_f0, decl, stress):
    """Three f0 contours (jitter 0): base only, base + declination, base + declination + stress (the full contour)."""
    from src.pitch import build_f0
    labels, energy = feats["labels"], feats["energy"]
    return dict(base=build_f0(labels, energy, base_f0=base_f0, decl=0.0, stress=0.0, jitter=0.0),
                declination=build_f0(labels, energy, base_f0=base_f0, decl=decl, stress=0.0, jitter=0.0),
                full=build_f0(labels, energy, base_f0=base_f0, decl=decl, stress=stress, jitter=0.0))


def f0_breakdown_figure(feats, settings):
    from matplotlib.figure import Figure
    s = settings
    c = f0_components(feats, s["base_f0"], s["decl"], s["stress"])
    t = np.arange(len(feats["labels"])) * HOP / FS
    fig = Figure(figsize=(7, 3.2))
    ax = fig.subplots()
    for key, label in (("base", "base only"), ("declination", "+ declination"), ("full", "+ declination + stress (full)")):
        ax.plot(t, np.where(c[key] > 0, c[key], np.nan), label=label, lw=1.6 if key == "full" else 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f0 (Hz)")
    ax.set_title("What builds the f0 contour (voiced frames, jitter off)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def rms_db(sig):
    """Per frame RMS of a signal in dB (framing.frame_signal and frame_energy)."""
    from src.framing import frame_signal, frame_energy
    return 20 * np.log10(frame_energy(frame_signal(np.asarray(sig, dtype=np.float32))) + 1e-6)


def _limit_db_axis(ax, span=70.0):
    """Show the top span dB of the data; digital silence (-120 dB) would otherwise squash the speech."""
    top = max(float(np.max(line.get_ydata())) for line in ax.get_lines())
    ax.set_ylim(top - span, top + 3)


def loudness_figure(x, y, fs=FS, hop=HOP):
    """Waveform overlay (input vs output) and per frame RMS in dB."""
    from matplotlib.figure import Figure
    n = min(len(x), len(y))
    fig = Figure(figsize=(7, 4.8))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    t = np.arange(n) / fs
    ax1.plot(t, x[:n], lw=0.5, color="0.5", label="input (whisper)")
    ax1.plot(t, y[:n], lw=0.5, color="C1", alpha=0.7, label="output (voiced)")
    ax1.set_ylabel("Amplitude")
    for line in ax1.legend(fontsize=7, loc="upper right").get_lines():
        line.set_linewidth(2)
    ax1.set_title("Loudness, input vs output", fontsize=9)
    for sig, name, color in ((x, "input RMS", "0.4"), (y, "output RMS", "C1")):
        e = rms_db(sig[:n])
        ax2.plot(np.arange(len(e)) * hop / fs, e, color=color, label=name)
    ax2.set_ylabel("Frame RMS (dB)")
    ax2.set_xlabel("Time (s)")
    _limit_db_axis(ax2)
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def source_loudness_figures(path, y, variant, base_f0, decl, stress, aspiration, warp):
    """(excitation zoom, f0 breakdown, loudness) for the clip and the rendered output y."""
    from src.framing import load_audio
    if not path:
        raise ValueError("Upload or record a clip first, or press Load demo clip.")
    feats = get_features(path)
    if len(feats["labels"]) == 0:
        raise ValueError(TOO_SHORT)
    s = render_settings(variant, base_f0, decl, stress, aspiration, warp)
    x, _ = load_audio(path)
    return excitation_figure(feats, s), f0_breakdown_figure(feats, s), loudness_figure(x, y)


def source_loudness_ui(path, variant, base_f0, decl, stress, aspiration, warp, volume, out_audio):
    """Gradio wrapper, run after Convert. The output player value only gates it: Gradio peak normalises a numpy
    player to int16, which would shift the output level in the loudness plot. The output is rendered again instead
    (cached features, deterministic seeds), so it is the same signal as the player before that normalisation.
    """
    if not path:
        raise ValueError("Upload or record a clip first, or press Load demo clip.")
    if out_audio is None:
        raise ValueError("Press Convert first.")
    y, _ = render(path, variant, base_f0, decl, stress, aspiration, warp, volume)
    return source_loudness_figures(path, y, variant, base_f0, decl, stress, aspiration, warp)


def comparison_signals(x, results):
    """[(label, signal)] for the input then each compare_variants result, trimmed to a common length."""
    n = min([len(x)] + [len(r[1][1]) for r in results])
    return [("input", np.asarray(x[:n]))] + [(r[0], np.asarray(r[1][1][:n])) for r in results]


def spectrogram_grid_figure(signals, fs=FS):
    """Spectrograms of every signal on one shared dB colour scale."""
    from matplotlib import mlab
    from matplotlib.figure import Figure
    rng = np.random.default_rng(0)
    sigs = [(name, np.asarray(s, dtype=np.float64) + 1e-7 * rng.standard_normal(len(s))) for name, s in signals]
    vmax = max(float(np.max(10 * np.log10(mlab.specgram(s, NFFT=512, Fs=fs, noverlap=384)[0] + 1e-20)))
               for _, s in sigs)
    vmin = vmax - 80.0
    rows = (len(sigs) + 1) // 2
    fig = Figure(figsize=(8, 2.1 * rows + 0.4))
    axes = np.atleast_1d(fig.subplots(rows, 2, sharex=True, sharey=True)).ravel()
    im = None
    for ax, (name, s) in zip(axes, sigs):
        im = ax.specgram(s, NFFT=512, Fs=fs, noverlap=384, cmap="magma", vmin=vmin, vmax=vmax)[3]
        ax.set_title(name, fontsize=8)
    for ax in axes[len(sigs):]:
        ax.axis("off")
    for ax in axes[::2]:
        ax.set_ylabel("Frequency (Hz)")
    for ax in axes[-2:]:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.colorbar(im, ax=list(axes[:len(sigs)]), label="dB (shared scale)", shrink=0.8)
    return fig


def lsd_values(x, signals):
    """{label: mean log spectral distortion (dB) of that signal against x over non-silent frames} via evaluate.lsd_variant."""
    from src import evaluate
    return {name: evaluate.lsd_variant(x[:len(s)], s) for name, s in signals}


def lsd_figure(values):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()
    names = list(values)
    bars = ax.bar(range(len(names)), [values[k] for k in names], color="C0")
    ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(range(len(names)), names, fontsize=7, rotation=15)
    ax.set_ylabel("LSD (dB), lower is closer")
    ax.set_title("Log spectral distortion against the whisper input (non-silent frames)", fontsize=9)
    fig.tight_layout()
    return fig


def ltas_figure(signals, fs=FS):
    """Long term average spectrum (Welch, dB) of every signal on one axis."""
    import scipy.signal
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 3.4))
    ax = fig.subplots()
    for name, s in signals:
        f, p = scipy.signal.welch(np.asarray(s, dtype=np.float64), fs=fs, nperseg=512)
        # custom is dashed and on top: at default sliders it equals v3_aspiration and would otherwise be hidden
        custom = name.startswith("custom")
        ax.plot(f, 10 * np.log10(p + 1e-14), label=name, color="0.2" if name == "input" else None,
                lw=1.8 if name == "input" else 1.0, ls="--" if custom else "-", zorder=3 if custom else 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (dB/Hz)")
    ax.set_title("Long term average spectrum (Welch): spectral tilt, input vs variants", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def comparison_figures(path, results):
    """(spectrogram grid, LSD bars, LTAS) from the compare_variants results; nothing is rendered again."""
    from src.framing import load_audio
    x, _ = load_audio(path)
    sigs = comparison_signals(x, results)
    if np.any(get_features(path)["labels"] != 0):
        lsd_fig = lsd_figure(lsd_values(sigs[0][1], sigs[1:]))
    else:  # lsd_frames returns 0.0 for an empty selection, which would read as a perfect match
        lsd_fig = _message_figure("No non-silent frames in this clip, so LSD is not defined.", figsize=(7, 3))
    return spectrogram_grid_figure(sigs), lsd_fig, ltas_figure(sigs)


def live_run(path, variant, base_f0, decl, stress, aspiration, warp, volume):
    """Run the clip through the live (causal, hop by hop) path with the current settings, plus the offline render.

    Returns dict(x, y_live, y_off, timings (ms per hop), report (live.latency_report)).
    """
    from src import live
    from src.framing import load_audio
    if not path:
        raise ValueError("Upload or record a clip first, or press Load demo clip.")
    y_off, _ = render(path, variant, base_f0, decl, stress, aspiration, warp, volume)
    s = render_settings(variant, base_f0, decl, stress, aspiration, warp)
    x, _ = load_audio(path)
    params = dict(base_f0=s["base_f0"], decl=s["decl"], stress=s["stress"], jitter=s["jitter"],
                  aspiration=s["aspiration"], warp=s["warp"], volume=volume)
    state = live.init_live_state(params)
    y_live = live.process_file_live(x, state=state)
    timings = np.asarray(state["timings"], dtype=np.float64)
    return dict(x=x, y_live=y_live, y_off=y_off, timings=timings, report=live.latency_report(state))


def latency_figure(timings):
    from matplotlib.figure import Figure
    from src.live import BUDGET_MS
    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()
    ax.plot(np.arange(len(timings)), timings, lw=0.8, label="processing time per hop")
    ax.axhline(BUDGET_MS, color="r", ls="--", label=f"{BUDGET_MS:.0f} ms budget (one hop)")
    ax.set_xlabel("Hop index (one hop = 10 ms of audio)")
    ax.set_ylabel("Processing time (ms)")
    ax.set_title("Live path: time to process each hop", fontsize=9)
    ax.set_ylim(0, max(BUDGET_MS * 1.4, float(np.max(timings)) * 1.1 if len(timings) else 0.0))
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.8))
    fig.tight_layout()
    return fig


def live_envelope_figure(y_live, y_off, fs=FS, hop=HOP):
    from matplotlib.figure import Figure
    n = min(len(y_live), len(y_off))
    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()
    for sig, name, style in ((y_off, "offline render (same settings)", "-"), (y_live, "live path", "--")):
        e = rms_db(sig[:n])
        ax.plot(np.arange(len(e)) * hop / fs, e, ls=style, label=name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frame RMS (dB)")
    _limit_db_axis(ax)
    ax.set_title("Live vs offline loudness envelope", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def live_summary(report):
    from src.live import BUDGET_MS
    r = report
    return (f"**Hops:** {r['hops']}  \n"
            f"**Processing time per hop:** mean {r['mean_ms']:.2f} ms, 95th percentile {r['p95_ms']:.2f} ms, "
            f"max {r['max_ms']:.2f} ms (budget {BUDGET_MS:.0f} ms)  \n"
            f"**Hops over budget:** {100 * r['frac_over_budget']:.1f}%  \n"
            f"**Algorithmic delay:** {r['algorithmic_ms']:.0f} ms. This is a model (two hops from the frame "
            f"overlap and buffering), not a measurement. Sound device latency is not measured here; "
            f"run `python run_live.py` with real devices for that.")


def live_ui(path, variant, base_f0, decl, stress, aspiration, warp, volume):
    """Gradio wrapper for the Live tab. Returns (player value, latency figure, envelope figure, summary markdown)."""
    r = live_run(path, variant, base_f0, decl, stress, aspiration, warp, volume)
    return ((FS, r["y_live"]), latency_figure(r["timings"]), live_envelope_figure(r["y_live"], r["y_off"]),
            live_summary(r["report"]))


def analysis_features(path):
    """Frames, RMS, ZCR and voicing strength of a clip, cached per file."""
    from src.framing import load_audio, frame_signal, frame_energy, frame_zcr
    from src.detect import voicing_strength
    key = ("analysis", path, os.path.getmtime(path))
    if key not in _CACHE:
        x, _ = load_audio(path)
        frames = frame_signal(x)
        _CACHE[key] = dict(x=x, frames=frames, energy=frame_energy(frames), zcr=frame_zcr(frames),
                           voicing=voicing_strength(frames))
    return _CACHE[key]


def classify_for_analysis(af, silence_mult=3.0, zcr_thr=None):
    """Re-run classification. zcr_thr of None or 0 keeps the automatic threshold. Returns (labels, zcr_thr, silence_thr)."""
    from src.detect import classify_frames
    zt = float(zcr_thr) if zcr_thr else None
    return classify_frames(af["frames"], af["energy"], zcr=af["zcr"], silence_mult=silence_mult,
                           zcr_thr=zt, return_thresholds=True)


def module_a_figure(af, silence_mult=3.0, zcr_thr=None):
    """RMS, ZCR, voicing strength and frame labels with the thresholds drawn as lines."""
    from matplotlib.figure import Figure
    labels, zt, st = classify_for_analysis(af, silence_mult, zcr_thr)
    t = np.arange(len(labels)) * HOP / FS
    fig = Figure(figsize=(7, 7))
    axes = fig.subplots(4, 1, sharex=True)
    axes[0].plot(t, af["energy"])
    axes[0].axhline(st, color="r", ls="--", label=f"silence threshold {st:.4f}")
    axes[0].set_ylabel("RMS")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].plot(t, af["zcr"])
    axes[1].axhline(zt, color="r", ls="--", label=f"ZCR threshold {zt:.3f}")
    axes[1].set_ylabel("ZCR")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[2].plot(t, af["voicing"])
    axes[2].set_ylabel("Voicing")
    axes[3].step(t, labels, where="post")
    axes[3].set_yticks([0, 1, 2], ["silence", "consonant", "vowel-like"])
    axes[3].set_xlabel("Time (s)")
    axes[0].set_title("Per-frame features and labels", fontsize=9)
    fig.tight_layout()
    return fig


def lpc_frame_data(af, idx, warp=1.0, order=12):
    """Spectrum, LPC envelope, warped envelope (dB) and prediction gain for one frame.

    The LPC fit is lpc_frame (pre-emphasis 0.97, Hamming, Levinson), as in the pipeline.
    The FFT spectrum uses the same pre-emphasised, Hamming windowed frame, so it is directly comparable.
    """
    from src.lpc import preemphasis, lpc_frame, lpc_envelope, warp_poles, prediction_gain_db
    frames = af["frames"]
    if len(frames) == 0:
        raise ValueError(TOO_SHORT)
    idx = int(min(max(idx, 0), len(frames) - 1))
    frame = frames[idx]
    a, g = lpc_frame(frame, order)
    n_fft = 512
    win = np.hamming(len(frame))
    xw = preemphasis(frame) * win
    spec_db = 20 * np.log10(np.abs(np.fft.rfft(xw, n_fft)) / np.sqrt(np.sum(win ** 2)) + 1e-9)
    freqs, env = lpc_envelope(a, g, n_fft=n_fft)
    _, env_w = lpc_envelope(warp_poles(a, float(warp)), g, n_fft=n_fft)
    pg = prediction_gain_db(preemphasis(frame), a)
    return dict(idx=idx, freqs=freqs, spec_db=spec_db, env_db=env, warped_db=env_w, pred_gain_db=pg, gain=g)


def lpc_frame_figure(af, idx, warp=1.0, order=12):
    """FFT magnitude of one frame against its LPC envelope and the pole-warped envelope. Returns (figure, text)."""
    from matplotlib.figure import Figure
    d = lpc_frame_data(af, idx, warp, order)
    fig = Figure(figsize=(7, 3.6))
    ax = fig.subplots()
    ax.plot(d["freqs"], d["spec_db"], color="0.6", label="FFT magnitude")
    ax.plot(d["freqs"], d["env_db"], label=f"LPC envelope (order {order})")
    if warp < 1.0:
        ax.plot(d["freqs"], d["warped_db"], label=f"warped poles ({warp:.2f})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dB")
    ax.set_title(f"Frame {d['idx']}", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, f"Frame {d['idx']}: prediction gain {d['pred_gain_db']:.1f} dB, LPC gain {d['gain']:.4f}"


def analyse(audio_path, silence_mult, zcr_thr, idx, warp):
    """Both analysis figures for a clip. Returns (module A figure, frame figure, text, n_frames)."""
    if not audio_path:
        raise ValueError("Upload or record a clip on the Convert tab first, or press Load demo clip.")
    af = analysis_features(audio_path)
    fig_b, text = lpc_frame_figure(af, idx, warp)
    return module_a_figure(af, silence_mult, zcr_thr), fig_b, text, len(af["frames"])


def results_root():
    return os.path.join(ROOT, "results")


def discover_results(root=None):
    """Folders (results and its direct subfolders) that contain ablation.md."""
    root = root or results_root()
    if not os.path.isdir(root):
        return []
    cands = [root] + sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p))
    return [p for p in cands if os.path.isfile(os.path.join(p, "ablation.md"))]


def results_figures(folder):
    """PNG paths in <folder>/figures (run_eval.py writes figures next to its ablation.md)."""
    return sorted(glob.glob(os.path.join(folder, "figures", "*.png")))


def results_clips(folder):
    """Clip stems that have variant wavs, from names like <stem>_v1_robot.wav."""
    suffix = "_v1_robot.wav"
    return sorted(os.path.basename(p)[:-len(suffix)] for p in glob.glob(os.path.join(folder, "*" + suffix)))


def clip_variant_paths(folder, stem):
    """Existing variant wav paths for a clip, as {variant: path}."""
    out = {}
    for v in VARIANT_CHOICES[1:]:
        p = os.path.join(folder, f"{stem}_{v}.wav")
        if os.path.isfile(p):
            out[v] = p
    return out


def results_markdown(folder):
    """Contents of ablation.md, or a friendly message if there is no folder."""
    if not folder:
        return NO_RESULTS
    with open(os.path.join(folder, "ablation.md"), encoding="utf8") as f:
        return f.read()


def build_app():
    import functools

    import gradio as gr

    def friendly(fn):
        """Show ValueError messages in the UI; Gradio hides plain exception text unless show_error=True."""
        @functools.wraps(fn)
        def wrapper(*args):
            try:
                return fn(*args)
            except ValueError as e:
                raise gr.Error(str(e)) from e
        return wrapper

    folders = discover_results()
    labels = {os.path.relpath(p, ROOT): p for p in folders}

    with gr.Blocks(title="Whisper to Voice") as demo:
        gr.Markdown("## Whisper to Voice\nWhispered speech in, voiced speech out via LPC source-filter resynthesis.")
        with gr.Tabs():
            with gr.Tab("Convert"):
                with gr.Row():
                    with gr.Column():
                        audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Whisper input")
                        demo_btn = gr.Button("Load demo clip")
                        variant = gr.Dropdown(VARIANT_CHOICES, value="custom", label="Variant (custom uses the sliders)")
                        base_f0 = gr.Slider(70, 250, value=120, step=1, label="Base pitch (Hz)")
                        decl = gr.Slider(0, 0.3, value=0.12, step=0.01, label="Intonation (declination)")
                        stress = gr.Slider(0, 0.5, value=0.25, step=0.01, label="Stress")
                        aspiration = gr.Slider(0, 0.3, value=0.15, step=0.01, label="Aspiration")
                        warp = gr.Slider(0.85, 1.0, value=1.0, step=0.01, label="Pole warp (1.0 = off)")
                        volume = gr.Slider(0.2, 2.0, value=1.0, step=0.05, label="Volume")
                        go = gr.Button("Convert", variant="primary")
                        cmp_btn = gr.Button("Compare all variants")
                        gr.Markdown("### Transcribe (speech to text)")
                        asr_model = gr.Dropdown(ASR_MODELS, value="base", label="Whisper ASR model")
                        reference = gr.Textbox(label="Reference sentence (optional, for WER)")
                        tr_btn = gr.Button("Transcribe")
                        gr.Markdown(ASR_NOTE)
                        gr.Markdown(LIVE_NOTE)
                    with gr.Column():
                        with gr.Row():
                            in_audio = gr.Audio(label="Input (A)", type="filepath", interactive=False)
                            # interactive=False: it is also a Transcribe input, which would otherwise make it an upload box
                            out_audio = gr.Audio(label="Voiced output (B)", type="numpy", interactive=False)
                        download = gr.File(label="Download output wav")
                        spec = gr.Plot(label="Spectrograms")
                        lab = gr.Plot(label="Labels and f0")
                        with gr.Accordion("Source and loudness", open=False):
                            gr.Markdown("Filled when Convert runs: excitation zoom, f0 contour breakdown, input vs output loudness.")
                            exc_plot = gr.Plot(label="Excitation (about 70 ms, voiced)")
                            f0_plot = gr.Plot(label="f0 contour breakdown")
                            loud_plot = gr.Plot(label="Loudness")
                        with gr.Accordion("Variant comparison", open=False):
                            gr.Markdown("Filled by Compare all variants (the same renders as the players).")
                            grid_plot = gr.Plot(label="Spectrograms (shared colour scale)")
                            lsd_plot = gr.Plot(label="LSD per variant")
                            ltas_plot = gr.Plot(label="Long term average spectrum")
                        transcripts = gr.Markdown(label="Transcripts")
                        cmp_players = [gr.Audio(label="custom (sliders)" if name == "custom" else name, type="numpy")
                                       for name in VARIANT_CHOICES]
                params = [audio, variant, base_f0, decl, stress, aspiration, warp, volume]
                demo_btn.click(demo_clip_path, outputs=audio)
                converted_from = gr.State(None)

                def convert_tracked(*args):
                    return (*convert_full(*args), args[0])

                source_plots = [exc_plot, f0_plot, loud_plot]
                compare_plots = [grid_plot, lsd_plot, ltas_plot]
                go.click(friendly(convert_tracked), params,
                         [out_audio, spec, lab, in_audio, download, converted_from]).success(
                    friendly(source_loudness_ui), [*params, out_audio], source_plots)
                # A new input clip makes the previous output stale: clear it so player and transcript never disagree
                # (the compare players and the transcript table belong to the old clip too)
                stale = [out_audio, in_audio, download, converted_from, *source_plots, *compare_plots, *cmp_players]
                audio.change(lambda: (None,) * len(stale) + ("",), outputs=[*stale, transcripts])

                def compare(path, _variant, *sliders):
                    results = compare_variants(path, *sliders)
                    return [r[1] for r in results] + list(comparison_figures(path, results))

                cmp_btn.click(friendly(compare), params, cmp_players + compare_plots)
                tr_btn.click(friendly(transcribe_ui), [audio, asr_model, reference, out_audio, converted_from, *params[1:]],
                             transcripts)

            with gr.Tab("Analysis"):
                gr.Markdown("Uses the clip loaded on the Convert tab.")
                an_btn = gr.Button("Analyse clip", variant="primary")
                gr.Markdown("### Frame classification")
                sil_mult = gr.Slider(1, 10, value=3.0, step=0.1, label="Silence multiplier")
                zcr_slider = gr.Slider(0, 0.5, value=0, step=0.01, label="ZCR threshold (0 = automatic)")
                plot_a = gr.Plot(label="Frame features and labels")
                gr.Markdown("### LPC envelope of one frame")
                frame_idx = gr.Slider(0, 100, value=0, step=1, label="Frame index")
                warp_b = gr.Slider(0.85, 1.0, value=0.92, step=0.01, label="Warp factor (1.0 = off)")
                plot_b = gr.Plot(label="LPC envelope")
                gain_txt = gr.Markdown()
                plot_poles = gr.Plot(label="Poles on the unit circle (selected frame)")
                plot_formants = gr.Plot(label="Formant tracks")

                def run_analysis(path, sm, zt, idx, w):
                    fa, fb, text, n = analyse(path, sm, zt, idx, w)
                    return fa, fb, text, gr.Slider(maximum=n - 1), *module_b_pole_figures(path, idx, w)

                def rerun_a(path, sm, zt):
                    return module_a_figure(analysis_features(path), sm, zt) if path else None

                def rerun_b(path, idx, w):
                    if not path:
                        return None, "", None, None
                    return (*lpc_frame_figure(analysis_features(path), idx, w), *module_b_pole_figures(path, idx, w))

                an_btn.click(friendly(run_analysis), [audio, sil_mult, zcr_slider, frame_idx, warp_b],
                             [plot_a, plot_b, gain_txt, frame_idx, plot_poles, plot_formants])
                for s in (sil_mult, zcr_slider):
                    s.release(rerun_a, [audio, sil_mult, zcr_slider], plot_a)
                for s in (frame_idx, warp_b):
                    s.release(friendly(rerun_b), [audio, frame_idx, warp_b], [plot_b, gain_txt, plot_poles, plot_formants])

            with gr.Tab("Live"):
                gr.Markdown("Pushes the clip loaded on the Convert tab through the live (causal, hop by hop) path with the "
                            "current Convert settings. No sound device is used. For real time use run "
                            "`python run_live.py --list-devices`, then `python run_live.py --in N --out M`.")
                live_btn = gr.Button("Run clip through the live path", variant="primary")
                live_audio = gr.Audio(label="Live path output", type="numpy", interactive=False)
                live_summary_md = gr.Markdown()
                live_lat_plot = gr.Plot(label="Per hop processing time")
                live_env_plot = gr.Plot(label="Live vs offline loudness")
                live_btn.click(friendly(live_ui), params, [live_audio, live_lat_plot, live_env_plot, live_summary_md])
                audio.change(lambda: (None,) * 4, outputs=[live_audio, live_lat_plot, live_env_plot, live_summary_md])

            with gr.Tab("Results"):
                if not folders:
                    gr.Markdown(NO_RESULTS)
                else:
                    first = folders[0]
                    first_clips = results_clips(first)
                    folder_dd = gr.Dropdown(list(labels), value=os.path.relpath(first, ROOT), label="Results folder")
                    table = gr.Markdown(results_markdown(first))
                    gallery = gr.Gallery(value=results_figures(first), label="Figures", columns=3)
                    clip_dd = gr.Dropdown(first_clips, value=first_clips[0] if first_clips else None, label="Clip")
                    res_players = [gr.Audio(label=v, type="filepath") for v in VARIANT_CHOICES[1:]]

                    def show_clip(folder_label, stem):
                        paths = clip_variant_paths(labels[folder_label], stem) if stem else {}
                        return [paths.get(v) for v in VARIANT_CHOICES[1:]]

                    def show_folder(folder_label):
                        f = labels[folder_label]
                        clips = results_clips(f)
                        stem = clips[0] if clips else None
                        return [results_markdown(f), results_figures(f), gr.Dropdown(choices=clips, value=stem),
                                *show_clip(folder_label, stem)]

                    folder_dd.change(show_folder, folder_dd, [table, gallery, clip_dd, *res_players])
                    clip_dd.change(show_clip, [folder_dd, clip_dd], res_players)
                    demo.load(show_clip, [folder_dd, clip_dd], res_players)
        gr.Markdown(LIVE_NOTE)
    return demo


if __name__ == "__main__":
    build_app().launch()
