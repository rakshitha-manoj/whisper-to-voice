"""Detection utilities for Module A: voicing strength, frame classification, plots."""

import os

import numpy as np
import matplotlib.pyplot as plt

from .framing import frame_zcr


def voicing_strength(frames: np.ndarray, fs: int = 16000, lag_min: int = 40, lag_max: int = 228) -> np.ndarray:
    """Normalized autocorrelation peak per frame.

    For each lag k the overlapping parts x[0:N-k] and x[k:N] are compared with
    a normalized cross correlation, so a periodic frame scores near 1 even
    though only N-k samples overlap, and noise scores near 0.
    lag_min/lag_max are in samples; at 16 kHz lags 40..228 cover
    pitch periods from 400 Hz down to about 70 Hz.

    Returns strength, shape (n_frames,).
    """
    n_frames, n = frames.shape
    out = np.zeros(n_frames, dtype=np.float32)
    lags = np.arange(lag_min, min(lag_max, n - 2) + 1)
    for i, frame in enumerate(frames.astype(np.float64)):
        ac = np.correlate(frame, frame, mode='full')[n - 1:]
        csum = np.concatenate(([0.0], np.cumsum(frame ** 2)))
        e_head = csum[n - lags]
        e_tail = csum[n] - csum[lags]
        denom = np.sqrt(e_head * e_tail) + 1e-12
        out[i] = np.max(ac[lags] / denom) if len(lags) else 0.0
    return out


def calibrate_thresholds(energy, zcr, fs=16000, hop=160, calib_s=0.3, silence_mult=3.0, zcr_pct=70):
    """Return (silence_thr, zcr_thr) from the leading noise floor.

    If the first calib_s seconds are louder than the 25th percentile of all
    frame energies (clip starts with speech), the floor falls back to the
    10th percentile of the energies.
    """
    energy = np.asarray(energy, dtype=np.float64)
    zcr = np.asarray(zcr, dtype=np.float64)
    if energy.size == 0:
        return 1e-4, 0.5
    n_early = min(max(int(calib_s * fs // hop), 1), energy.size)
    floor = float(np.median(energy[:n_early]))
    if floor > np.percentile(energy, 25):
        floor = float(np.percentile(energy, 10))
    silence_thr = max(silence_mult * floor, 1e-4)
    non_silent = energy >= silence_thr
    if np.any(non_silent):
        zcr_thr = float(np.percentile(zcr[non_silent], zcr_pct))
    else:
        zcr_thr = 0.5
    return silence_thr, zcr_thr


def label_frame(rms, zcr, silence_thr, zcr_thr) -> int:
    """Label one hop: 0 silence, 1 noisy consonant, 2 vowel-like."""
    if rms < silence_thr:
        return 0
    if zcr > zcr_thr:
        return 1
    return 2


def classify_frames(frames, energy, zcr=None, silence_mult=3.0, zcr_thr=None, return_thresholds=False, hangover=5):
    """Classify frames: 0 silence, 1 noisy consonant, 2 vowel-like.

    Returns int labels of shape (n_frames,), or (labels, zcr_thr, silence_thr)
    when return_thresholds is True. A given zcr_thr overrides the percentile rule.
    """
    energy = np.asarray(energy)
    if zcr is None:
        zcr = frame_zcr(frames)
    silence_thr, auto_zcr_thr = calibrate_thresholds(energy, zcr, silence_mult=silence_mult)
    if zcr_thr is None:
        zcr_thr = auto_zcr_thr
    labels = np.where(energy < silence_thr, 0, np.where(zcr > zcr_thr, 1, 2)).astype(int)

    last = 0
    since = hangover + 1
    raw = labels.copy()
    for i in range(len(raw)):
        if raw[i] != 0:
            last = raw[i]
            since = 0
        else:
            since += 1
            if since <= hangover and last != 0:
                labels[i] = last

    if len(labels) >= 3:
        med = labels.copy()
        med[1:-1] = np.median(np.vstack((labels[:-2], labels[1:-1], labels[2:])), axis=0)
        labels = med.astype(int)

    if return_thresholds:
        return labels, float(zcr_thr), float(silence_thr)
    return labels


def save_handoff(labels, energy, out_dir="data/handoff"):
    """Save labels.npy and energy.npy for teammates."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(os.path.join(out_dir, "energy.npy"), energy)
    return out_dir


def plot_labels(x: np.ndarray, fs: int, frames: np.ndarray, labels: np.ndarray, hop: int = 160, output_path: str = "results/labels.png"):
    """Plot waveform with coloured label markers and save to output_path."""
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    times = np.arange(len(x)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(times, x, label='waveform')
    frame_times = (np.arange(len(labels)) * hop + frames.shape[1] / 2) / fs
    colors = {0: 'red', 1: 'orange', 2: 'green'}
    for lbl, col in colors.items():
        mask = labels == lbl
        plt.scatter(frame_times[mask], np.zeros_like(frame_times[mask]), color=col, marker='|', s=100, label=f'Label {lbl}')
    plt.xlabel('Time (s)')
    plt.title('Waveform with frame-level classification')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_spectrogram(x: np.ndarray, fs: int = 16000, win_ms: float = 25.0, hop_ms: float = 10.0, output_path: str = "results/spectrogram.png"):
    """Compute and plot a Hann‑window spectrogram in dB scale.
    Saves the figure to ``output_path`` and returns the spectrogram matrix,
    frequency bins, and time bins.
    """
    import os
    from scipy.signal import stft, get_window

    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    else:
        os.makedirs('results', exist_ok=True)
    win_len = int(round(win_ms * fs / 1000.0))
    hop_len = int(round(hop_ms * fs / 1000.0))
    window = get_window('hann', win_len, fftbins=True)
    f, t, Z = stft(x, fs=fs, window=window, nperseg=win_len,
                   noverlap=win_len - hop_len, nfft=win_len,
                   detrend=False, boundary=None, padded=False)
    S_db = 20 * np.log10(np.abs(Z) + 1e-10)
    plt.figure(figsize=(8, 4))
    plt.imshow(S_db, aspect='auto', origin='lower',
               extent=[t.min(), t.max(), f.min(), f.max()], cmap='viridis')
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return S_db, f, t
