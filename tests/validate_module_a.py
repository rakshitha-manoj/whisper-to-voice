# tests/validate_module_a.py
"""Validation script for Module A.
Runs the full pipeline on the provided whisper and normal recordings,
produces required plots and prints chosen thresholds.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.framing import load_audio, frame_signal, frame_energy, frame_zcr
from src.detect import voicing_strength, classify_frames, plot_spectrogram, plot_labels

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

def process_file(path, name):
    x, fs = load_audio(path)
    frames = frame_signal(x)
    rms = frame_energy(frames)
    zcr = frame_zcr(frames)
    vs = voicing_strength(frames, fs=fs)
    labels, zcr_thr, silence_thr = classify_frames(frames, rms, zcr)
    # Save plots
    plot_labels(x, fs, frames, labels, output_path=os.path.join(RESULTS_DIR, f"labels_{name}.png"))
    plot_spectrogram(x, fs, output_path=os.path.join(RESULTS_DIR, f"spectrogram_{name}.png"))
    print(f"{name}: ZCR threshold = {zcr_thr:.3f}, silence threshold = {silence_thr:.6f}")
    return {
        'x': x, 'fs': fs, 'frames': frames, 'rms': rms, 'zcr': zcr,
        'vs': vs, 'labels': labels
    }

def synthetic_test():
    # 0.5 s silence, 0.5 s band‑limited noise bursts, 0.5 s silence
    fs = 16000
    silence = np.zeros(int(0.5 * fs), dtype=np.float32)
    noise = np.random.randn(int(0.5 * fs)).astype(np.float32) * 0.2
    # simple band‑pass filter (30‑3000 Hz) using scipy
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [30/(fs/2), 3000/(fs/2)], btype='band')
    noise = filtfilt(b, a, noise)
    signal = np.concatenate([silence, noise, silence])
    frames = frame_signal(signal)
    rms = frame_energy(frames)
    zcr = frame_zcr(frames)
    labels, _, _ = classify_frames(frames, rms, zcr)
    print('Synthetic test label distribution:', dict(zip(*np.unique(labels, return_counts=True))))

def compare_whisper_normal(whisper, normal):
    # Spectrograms side by side and voicing strength overlay
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # Spectrograms
    for ax, (data, title) in zip(axs[0], [((whisper['x'], whisper['fs']), 'Whisper'), ((normal['x'], normal['fs']), 'Normal')]):
        x, fs = data
        from scipy.signal import stft, get_window
        win_len = int(round(25 * fs / 1000))
        hop_len = int(round(10 * fs / 1000))
        f, t, Z = stft(x, fs=fs, window=get_window('hann', win_len), nperseg=win_len,
                       noverlap=win_len - hop_len, nfft=win_len, detrend=False,
                       boundary=None, padded=False)
        S_db = 20 * np.log10(np.abs(Z) + 1e-10)
        im = ax.imshow(S_db, aspect='auto', origin='lower', extent=[t.min(), t.max(), f.min(), f.max()], cmap='viridis')
        ax.set_title(f"Spectrogram – {title}")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(im, ax=ax, label='Amplitude (dB)')
    # Voicing strength over time
    times = np.arange(len(whisper['vs'])) * 160 / whisper['fs']
    axs[1,0].plot(times, whisper['vs'], label='Whisper')
    axs[1,0].set_title('Voicing strength – Whisper')
    axs[1,0].set_xlabel('Time (s)')
    axs[1,0].set_ylabel('Strength')
    axs[1,0].legend()
    times_n = np.arange(len(normal['vs'])) * 160 / normal['fs']
    axs[1,1].plot(times_n, normal['vs'], label='Normal', color='orange')
    axs[1,1].set_title('Voicing strength – Normal')
    axs[1,1].set_xlabel('Time (s)')
    axs[1,1].set_ylabel('Strength')
    axs[1,1].legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'whisper_vs_normal.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved comparison figure to {out_path}")

if __name__ == '__main__':
    whisper_path = os.path.join(DATA_DIR, 'spk01_s01_whisper.wav')
    normal_path = os.path.join(DATA_DIR, 'spk01_s01_normal.wav')
    whisper = process_file(whisper_path, 'whisper')
    normal = process_file(normal_path, 'normal')
    compare_whisper_normal(whisper, normal)
    synthetic_test()
