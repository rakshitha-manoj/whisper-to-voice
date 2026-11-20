"""Gradio demo: whisper in, voiced speech out (offline file mode, fallback for the live demo)."""

import os
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SHARED_CLIP = os.path.join(ROOT, "data", "raw", "shared_test.wav")
FS = 16000
HOP = 160
VARIANT_CHOICES = ["custom", "v1_robot", "v2_contour", "v3_aspiration", "v4_warp"]
_CACHE = {}


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
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = _features(path)
    return _CACHE[key]


def render(path, variant="custom", base_f0=120, decl=0.12, stress=0.25, aspiration=0.15, warp=1.0, volume=1.0):
    from src.synth import _render, render_variant
    feats = get_features(path)
    if variant in VARIANT_CHOICES[1:]:
        y = render_variant(feats, variant, volume=volume, base_f0=base_f0)
    else:
        w = None if warp is None or warp >= 1.0 else float(warp)
        y = _render(feats, volume=volume, base_f0=base_f0, decl=decl, stress=stress, aspiration=aspiration, warp=w)
    return y, feats


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


def label_f0_figure(feats, base_f0=120, decl=0.12, stress=0.25):
    from matplotlib.figure import Figure
    from src.pitch import build_f0
    labels, energy = feats["labels"], feats["energy"]
    t = np.arange(len(labels)) * HOP / FS
    f0 = build_f0(labels, energy, base_f0=base_f0, decl=decl, stress=stress)
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
    return (FS, y), spectrogram_figure(x[:n], y[:n]), label_f0_figure(feats, base_f0, decl, stress)


def build_app():
    import gradio as gr

    with gr.Blocks(title="Whisper to Voice") as demo:
        gr.Markdown("## Whisper to Voice\nUpload or record a whisper, adjust the controls, and press Convert.")
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
            with gr.Column():
                out_audio = gr.Audio(label="Voiced output", type="numpy")
                spec = gr.Plot(label="Spectrograms")
                lab = gr.Plot(label="Labels and f0")
        demo_btn.click(demo_clip_path, outputs=audio)
        go.click(convert_clip, [audio, variant, base_f0, decl, stress, aspiration, warp, volume],
                 [out_audio, spec, lab])
    return demo


if __name__ == "__main__":
    build_app().launch()
