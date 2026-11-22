# app_a.py
"""Gradio frontend for Module A.
It loads an audio file (or records from mic), runs the existing Module A utilities
(frames, RMS, ZCR, voicing strength, classification, spectrogram) and displays
the results in an interactive web page.

Only the functions in ``src`` are used – no changes to detection, framing or
streaming logic.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import uuid
import gradio as gr
from src.framing import load_audio, frame_signal, frame_energy, frame_zcr
from src.detect import voicing_strength, classify_frames, plot_spectrogram, plot_labels

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

def process_audio(
    audio_path: str | None = None,
    mic_data: np.ndarray | None = None,
    sample_rate: int = 16000,
    silence_multiplier: float = 3.0,
    zcr_threshold: float = 0.33,
) -> dict:
    """Load audio, compute features and generate all required plots.

    Returns a dictionary whose keys are passed to the Gradio interface.
    """
    # ------------------------------------------------------------------
    # Load audio (fallback to mic_data if provided)
    # ------------------------------------------------------------------
    if audio_path is not None:
        x, fs = load_audio(audio_path)
    elif mic_data is not None:
        # mic_data is a 1‑D numpy array (already 16 kHz float32)
        x = mic_data.astype(np.float32)
        fs = sample_rate
    else:
        raise ValueError("No audio source supplied")

    # ------------------------------------------------------------------
    # Frame and compute basic features
    # ------------------------------------------------------------------
    frames = frame_signal(x)
    rms = frame_energy(frames)
    zcr = frame_zcr(frames)
    vs = voicing_strength(frames, fs=fs)

    labels, _, silence_thr = classify_frames(frames, rms, zcr, silence_mult=silence_multiplier, zcr_thr=zcr_threshold, return_thresholds=True)
    noise_floor = silence_thr / silence_multiplier if silence_multiplier > 0 else float("nan")

    # ------------------------------------------------------------------
    # Generate plots – each saved to the results folder and returned as a path
    # ------------------------------------------------------------------
    # 1 – Waveform

    wf_uid = uuid.uuid4().hex[:8]
    wf_path = os.path.join(RESULTS_DIR, f"app_waveform_{wf_uid}.png")
    times = np.arange(len(x)) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(times, x)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(wf_path)
    plt.close()

    # 2 – Zoomed 100 ms view with frame bars

    zoom_uid = uuid.uuid4().hex[:8]
    zoom_path = os.path.join(RESULTS_DIR, f"app_zoom_{zoom_uid}.png")
    zoom_start = 0  # start at 0 s – UI will shift later
    zoom_samples = int(0.1 * fs)
    zoom_end = zoom_start + zoom_samples
    plt.figure(figsize=(10, 3))
    plt.plot(times[zoom_start:zoom_end], x[zoom_start:zoom_end])
    # draw frame boundaries
    frame_len = frames.shape[1]
    hop = 160
    for i, start in enumerate(range(0, zoom_samples - frame_len + 1, hop)):
        plt.axvspan(
            (zoom_start + start) / fs,
            (zoom_start + start + frame_len) / fs,
            color="gray",
            alpha=0.2,
        )
    plt.title("Zoomed 100 ms view – overlapping 320‑sample frames (hop 160)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(zoom_path)
    plt.close()

    # 3 – Label strip (reuse plot_labels helper which also saves the image)

    label_uid = uuid.uuid4().hex[:8]
    label_path = os.path.join(RESULTS_DIR, f"app_labels_{label_uid}.png")
    plot_labels(x, fs, frames, labels, output_path=label_path)

    # 4 – RMS, ZCR and voicing strength curves
    feats_path = os.path.join(RESULTS_DIR, "app_feats.png")
    t_frames = np.arange(len(rms)) * 160 / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t_frames, rms, label="RMS")
    plt.axhline(silence_thr, color="red", linestyle="--", label="Silence thr")
    plt.plot(t_frames, zcr, label="ZCR")
    plt.axhline(zcr_threshold, color="orange", linestyle="--", label="ZCR thr")
    plt.plot(t_frames, vs, label="Voicing strength")
    plt.xlabel("Time (s)")
    plt.title("Features over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(feats_path)
    plt.close()

    # 5 – Spectrogram (reuse helper)
    spec_path = os.path.join(RESULTS_DIR, "app_spectrogram.png")
    plot_spectrogram(x, fs, output_path=spec_path)

    # 6 – Summary table (as markdown)
    total_frames = len(labels)
    perc = {"silence": (labels == 0).sum() / total_frames * 100,
            "consonant": (labels == 1).sum() / total_frames * 100,
            "vowel": (labels == 2).sum() / total_frames * 100}
    table_md = (
        f"**Frames per label**\n"
        f"- Silence: {perc['silence']:.1f}%\n"
        f"- Consonant: {perc['consonant']:.1f}%\n"
        f"- Vowel‑like: {perc['vowel']:.1f}%\n"
        f"**Noise floor (from the first 0.3 s)**: {noise_floor:.6f}\n"
        f"**Silence threshold** (multiplier {silence_multiplier:.2f}): {silence_thr:.6f}\n"
        f"**ZCR threshold**: {zcr_threshold:.3f}\n"
        f"**Mean voicing strength**: {vs.mean():.3f}\n"
    )

    # 7 – Audio player path (Gradio can play the original file)
    audio_player_path = audio_path if audio_path is not None else None

    return {
        "waveform": wf_path,
        "zoom": zoom_path,
        "labels": label_path,
        "features": feats_path,
        "spectrogram": spec_path,
        "summary": table_md,
        "audio": audio_player_path,
    }

# ------------------------------------------------------------------
# Gradio UI definition
# ------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Module A – Whisper‑to‑Voice Front‑end")
    with gr.Tabs():
        with gr.TabItem("Analysis"):
            with gr.Row():
                mic = gr.Audio(label="Mic input (record)", type="numpy")
                file_upload = gr.File(label="Upload wav file", file_types=[".wav"])
                preset = gr.Dropdown(
                    ["data/raw/spk01_s01_whisper.wav", "data/raw/spk01_s01_normal.wav"],
                    label="Choose example",
                    value="data/raw/spk01_s01_whisper.wav",
                )
            silence_mul = gr.Slider(1, 10, value=3, step=0.1, label="Silence threshold multiplier")
            zcr_thr = gr.Slider(0, 0.8, value=0.33, step=0.01, label="ZCR threshold")
            process_btn = gr.Button("Process")

            with gr.Row():
                wf_img = gr.Image(label="Waveform")
                zoom_img = gr.Image(label="Zoomed view")
            label_img = gr.Image(label="Labels")
            feats_img = gr.Image(label="RMS / ZCR / Voicing")
            spec_img = gr.Image(label="Spectrogram")
            summary_md = gr.Markdown()
            audio_player = gr.Audio(label="Audio playback")
            placeholder = gr.Markdown("**Module B and C output (coming)**")

            def run_process(mic_data, uploaded_file, preset_path, sil_mul, zcr_thr):
                # Determine which source to use
                if uploaded_file is not None:
                    path = uploaded_file.name
                elif preset_path is not None:
                    path = preset_path
                else:
                    path = None
                # mic_data comes as (sample_rate, samples)
                if mic_data is not None:
                    # mic_data is a tuple (sample_rate, np.ndarray)
                    mic_arr = mic_data[1].astype(np.float32).flatten()
                else:
                    mic_arr = None
                try:
                    out = process_audio(
                        audio_path=path,
                        mic_data=mic_arr,
                        silence_multiplier=sil_mul,
                        zcr_threshold=zcr_thr,
                    )
                except Exception as e:
                    return (
                        None, None, None, None, None, f"**Error:** {e}", None, None
                    )
                return (
                    out["waveform"],
                    out["zoom"],
                    out["labels"],
                    out["features"],
                    out["spectrogram"],
                    out["summary"],
                    out["audio"],
                    "**Module B and C output (coming)**",
                )

            process_btn.click(
                fn=run_process,
                inputs=[mic, file_upload, preset, silence_mul, zcr_thr],
                outputs=[wf_img, zoom_img, label_img, feats_img, spec_img, summary_md, audio_player, placeholder],
            )
        with gr.TabItem("Whisper vs Normal"):
            # Fixed comparison – use the two example files directly
            whisper_path = "data/raw/spk01_s01_whisper.wav"
            normal_path = "data/raw/spk01_s01_normal.wav"
            # Generate comparison figure once at startup
            def make_comparison():
                # load both files
                x_w, fs_w = load_audio(whisper_path)
                x_n, fs_n = load_audio(normal_path)
                # spectrograms
                def spec_img(x, fs):
                    win_len = int(round(25 * fs / 1000))
                    hop_len = int(round(10 * fs / 1000))
                    f, t, Z = plt.mlab.specgram(x, NFFT=win_len, Fs=fs, noverlap=win_len - hop_len)
                # use existing helper
                spec_w = os.path.join(RESULTS_DIR, "comp_whisper_spec.png")
                spec_n = os.path.join(RESULTS_DIR, "comp_normal_spec.png")
                plot_spectrogram(x_w, fs_w, output_path=spec_w)
                plot_spectrogram(x_n, fs_n, output_path=spec_n)
                # voicing strength curves
                vs_w = voicing_strength(frame_signal(x_w), fs=fs_w)
                vs_n = voicing_strength(frame_signal(x_n), fs=fs_n)
                t_w = np.arange(len(vs_w)) * 160 / fs_w
                t_n = np.arange(len(vs_n)) * 160 / fs_n
                comp_path = os.path.join(RESULTS_DIR, "whisper_vs_normal_comp.png")
                plt.figure(figsize=(12, 5))
                ax1 = plt.subplot(1, 2, 1)
                img1 = plt.imread(spec_w)
                ax1.imshow(img1)
                ax1.set_title("Whisper spectrogram")
                ax1.axis('off')
                ax2 = plt.subplot(1, 2, 2)
                img2 = plt.imread(spec_n)
                ax2.imshow(img2)
                ax2.set_title("Normal spectrogram")
                ax2.axis('off')
                plt.tight_layout()
                plt.savefig(comp_path)
                plt.close()
                # voicing plot
                vs_path = os.path.join(RESULTS_DIR, "voicing_compare.png")
                plt.figure(figsize=(8, 4))
                plt.plot(t_w, vs_w, label='Whisper')
                plt.plot(t_n, vs_n, label='Normal', color='orange')
                plt.xlabel('Time (s)')
                plt.ylabel('Voicing strength')
                plt.title('Voicing strength comparison')
                plt.legend()
                plt.tight_layout()
                plt.savefig(vs_path)
                plt.close()
                return comp_path, vs_path
            if os.path.exists(whisper_path) and os.path.exists(normal_path):
                comp_img, vs_img = make_comparison()
                gr.Image(value=comp_img, label="Spectrograms side‑by‑side")
                gr.Image(value=vs_img, label="Voicing strength curves")
                gr.Markdown("*Whisper (left) vs Normal (right) – spectrograms and voicing strength.*")
            else:
                gr.Markdown("*Add spk01_s01 whisper and normal recordings to data/raw to enable this tab.*")

if __name__ == "__main__":
    demo.launch()
