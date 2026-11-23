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
    ax1.set_yticks([0, 1, 2], ["sil", "noise", "voiced"])
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
    axes[3].set_yticks([0, 1, 2], ["silence", "consonant", "vowel"])
    axes[3].set_xlabel("Time (s)")
    axes[0].set_title("Module A: per-frame features and labels", fontsize=9)
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
    ax.set_title(f"Module B: frame {d['idx']}", fontsize=9)
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
                        transcripts = gr.Markdown(label="Transcripts")
                        cmp_players = [gr.Audio(label="custom (sliders)" if name == "custom" else name, type="numpy")
                                       for name in VARIANT_CHOICES]
                params = [audio, variant, base_f0, decl, stress, aspiration, warp, volume]
                demo_btn.click(demo_clip_path, outputs=audio)
                converted_from = gr.State(None)

                def convert_tracked(*args):
                    return (*convert_full(*args), args[0])

                go.click(friendly(convert_tracked), params, [out_audio, spec, lab, in_audio, download, converted_from])
                # A new input clip makes the previous output stale: clear it so player and transcript never disagree
                audio.change(lambda: (None, None, None, None), outputs=[out_audio, in_audio, download, converted_from])

                def compare(path, _variant, *sliders):
                    return [r[1] for r in compare_variants(path, *sliders)]

                cmp_btn.click(friendly(compare), params, cmp_players)
                tr_btn.click(friendly(transcribe_ui), [audio, asr_model, reference, out_audio, converted_from, *params[1:]],
                             transcripts)

            with gr.Tab("Analysis"):
                gr.Markdown("Uses the clip loaded on the Convert tab.")
                an_btn = gr.Button("Analyse clip", variant="primary")
                gr.Markdown("### Module A: frame classification")
                sil_mult = gr.Slider(1, 10, value=3.0, step=0.1, label="Silence multiplier")
                zcr_slider = gr.Slider(0, 0.5, value=0, step=0.01, label="ZCR threshold (0 = automatic)")
                plot_a = gr.Plot(label="Module A")
                gr.Markdown("### Module B: LPC envelope of one frame")
                frame_idx = gr.Slider(0, 100, value=0, step=1, label="Frame index")
                warp_b = gr.Slider(0.85, 1.0, value=0.92, step=0.01, label="Warp factor (1.0 = off)")
                plot_b = gr.Plot(label="Module B")
                gain_txt = gr.Markdown()

                def run_analysis(path, sm, zt, idx, w):
                    fa, fb, text, n = analyse(path, sm, zt, idx, w)
                    return fa, fb, text, gr.Slider(maximum=n - 1)

                def rerun_a(path, sm, zt):
                    return module_a_figure(analysis_features(path), sm, zt) if path else None

                def rerun_b(path, idx, w):
                    if not path:
                        return None, ""
                    return lpc_frame_figure(analysis_features(path), idx, w)

                an_btn.click(friendly(run_analysis), [audio, sil_mult, zcr_slider, frame_idx, warp_b],
                             [plot_a, plot_b, gain_txt, frame_idx])
                for s in (sil_mult, zcr_slider):
                    s.release(rerun_a, [audio, sil_mult, zcr_slider], plot_a)
                for s in (frame_idx, warp_b):
                    s.release(friendly(rerun_b), [audio, frame_idx, warp_b], [plot_b, gain_txt])

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
