# src/detect.py
"""Detection utilities for Module A.
Provides voicing strength, frame classification for whispered speech,
and simple plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from .framing import frame_zcr


def voicing_strength(frames: np.ndarray, fs: int = 16000, lag_min: int = 40, lag_max: int = 228) -> np.ndarray:
    """Compute normalized autocorrelation peak for each frame.

    Parameters
    ----------
    frames : np.ndarray, shape (n_frames, frame_len)
    fs : int, default 16000
    lag_min, lag_max : int, lag range in samples (default corresponds to 70‑250 Hz)

    Returns
    -------
    strength : np.ndarray, shape (n_frames,)
    """
    n_frames = frames.shape[0]
    out = np.empty(n_frames, dtype=np.float32)
    for i, frame in enumerate(frames):
        ac_full = np.correlate(frame, frame, mode='full')
        ac = ac_full[frame.shape[0] - 1:]
        zero_lag = ac[0] if ac[0] != 0 else 1e-12
        ac_norm = ac / zero_lag
        start = min(lag_min, len(ac_norm) - 1)
        end = min(lag_max, len(ac_norm) - 1)
        out[i] = np.max(ac_norm[start:end + 1])
    return out


def classify_frames(frames: np.ndarray, energy: np.ndarray, zcr: np.ndarray = None):
    """Classify frames for whispered speech.

    Parameters
    ----------
    frames : np.ndarray
        Frameed audio signal (n_frames, frame_len).
    energy : np.ndarray
        RMS energy per frame.
    zcr : np.ndarray, optional
        Zero‑crossing rate per frame. If omitted it will be computed.

    Returns
    -------
    labels : np.ndarray, shape (n_frames,)
        0 – silence, 1 – noisy consonant, 2 – vowel‑like.
    zcr_thr : float
        Chosen ZCR threshold.
    silence_thr : float
        Silence threshold used.
    """
    # Compute ZCR if not provided
    if zcr is None:
        zcr = frame_zcr(frames)

    fs = 16000
    hop = 160
    # Noise floor from first 0.3 s of audio (approx number of frames)
    frames_early = int(0.3 * fs // hop)
    frames_early = min(frames_early, len(energy))
    noise_floor = np.median(energy[:frames_early]) if frames_early > 0 else 0.0
    silence_thr = max(3 * noise_floor, 1e-4)

    # Determine ZCR threshold from non‑silent frames
    non_silent = energy >= silence_thr
    if np.any(non_silent):
        perc = np.percentile(zcr[non_silent], [10, 30, 50, 70, 90])
        print(f"ZCR percentiles (non‑silent): {perc}")
        zcr_thr = np.percentile(zcr[non_silent], 70)
        print(f"Chosen ZCR threshold: {zcr_thr:.3f}")
    else:
        zcr_thr = 0.5

    # Initial labeling
    labels = np.empty_like(energy, dtype=int)
    labels[energy < silence_thr] = 0
    noisy_mask = (energy >= silence_thr) & (zcr > zcr_thr)
    labels[noisy_mask] = 1
    labels[(energy >= silence_thr) & ~noisy_mask] = 2

    # Hangover: propagate non‑silence forward up to 5 hops after a voiced region
    hangover = 5
    for i in range(1, len(labels)):
        if labels[i - 1] != 0 and labels[i] == 0:
            start = max(0, i - hangover)
            if np.any(labels[start:i] != 0):
                labels[i] = labels[i - 1]

    # Median smoothing (window size 3)
    if len(labels) >= 3:
        med = np.copy(labels)
        med[1:-1] = np.median(np.vstack((labels[:-2], labels[1:-1], labels[2:])), axis=0)
        labels = med.astype(int)

    return labels, zcr_thr, silence_thr
    """Classify frames for whispered speech.

    Parameters
    ----------
    frames : np.ndarray
        Frameed audio signal (n_frames, frame_len).
    energy : np.ndarray
        RMS energy per frame.
    zcr : np.ndarray, optional
        Zero‑crossing rate per frame. If omitted it will be computed.

    Returns
    -------
    labels : np.ndarray, shape (n_frames,)
        0 – silence, 1 – noisy consonant, 2 – vowel‑like.
    """
    # Compute ZCR if not provided
    if zcr is None:
        zcr = frame_zcr(frames)

    fs = 16000
    hop = 160
    # Noise floor from first 0.3 s of audio (approx number of frames)
    frames_early = int(0.3 * fs // hop)
    frames_early = min(frames_early, len(energy))
    noise_floor = np.median(energy[:frames_early]) if frames_early > 0 else 0.0
    silence_thr = max(3 * noise_floor, 1e-4)

    # Determine ZCR threshold from non‑silent frames
    non_silent = energy >= silence_thr
    if np.any(non_silent):
        zcr_thr = np.percentile(zcr[non_silent], 70)
        perc = np.percentile(zcr[non_silent], [10, 30, 50, 70, 90])
        print(f"ZCR percentiles (non‑silent): {perc}")
        print(f"Chosen ZCR threshold: {zcr_thr:.3f}")
    else:
        zcr_thr = 0.5

    # Initial labeling
    labels = np.empty_like(energy, dtype=int)
    labels[energy < silence_thr] = 0
    noisy_mask = (energy >= silence_thr) & (zcr > zcr_thr)
    labels[noisy_mask] = 1
    labels[(energy >= silence_thr) & ~noisy_mask] = 2

    # Hangover: propagate non‑silence forward up to 5 hops after a voiced region
    hangover = 5
    for i in range(1, len(labels)):
        if labels[i - 1] != 0 and labels[i] == 0:
            start = max(0, i - hangover)
            if np.any(labels[start:i] != 0):
                labels[i] = labels[i - 1]

    # Median smoothing (window size 3)
    if len(labels) >= 3:
        med = np.copy(labels)
        med[1:-1] = np.median(np.vstack((labels[:-2], labels[1:-1], labels[2:])), axis=0)
        labels = med.astype(int)

    return labels, zcr_thr, silence_thr
    """Classify frames for whispered speech.

    Returns integer labels per frame:
      0 – silence
      1 – noisy consonant (high ZCR)
      2 – vowel‑like (low ZCR)
    """
    # Compute ZCR per frame
    zcr = frame_zcr(frames)

    # Noise floor from first 0.3 s of audio
    samples_03s = int(0.3 * fs)
    frames_early = max(0, (samples_03s - frames.shape[1]) // hop) + 1
    frames_early = min(frames_03s, len(energy))
    noise_floor = np.median(energy[:frames_early]) if frames_early > 0 else 0.0
    silence_thr = max(3 * noise_floor, 1e-4)  # small absolute floor

    # Determine ZCR threshold from non‑silent frames
    non_silent_mask = energy >= silence_thr
    zcr_non_silent = zcr[non_silent_mask]
    if zcr_non_silent.size == 0:
        zcr_thr = 0.5
    else:
        # Print distribution percentiles for user reference (stdout)
        perc = np.percentile(zcr_non_silent, [10, 30, 50, 70, 90])
        print(f"ZCR percentiles (non‑silent): {perc}")
        # Choose threshold at 70th percentile (adjustable)
        zcr_thr = np.percentile(zcr_non_silent, 70)
        print(f"Chosen ZCR threshold: {zcr_thr:.3f}")

    # Initial labeling
    labels = np.empty_like(energy, dtype=int)
    labels[energy < silence_thr] = 0
    noisy_mask = (energy >= silence_thr) & (zcr > zcr_thr)
    labels[noisy_mask] = 1
    labels[(energy >= silence_thr) & ~noisy_mask] = 2

    # Hangover: extend a non‑silence label forward for up to 5 hops after a voiced region
    hangover = 5
    for i in range(1, len(labels)):
        if labels[i - 1] != 0 and labels[i] == 0:
            # look back up to hangover hops
            start = max(0, i - hangover)
            if np.any(labels[start:i] != 0):
                labels[i] = labels[i - 1]

    # Median smoothing (window size 3)
    if len(labels) >= 3:
        med = np.copy(labels)
        med[1:-1] = np.median(np.vstack((labels[:-2], labels[1:-1], labels[2:])), axis=0)
        labels = med.astype(int)

    return labels, zcr_thr, silence_thr, zcr_thr, silence_thr


def plot_labels(x: np.ndarray, fs: int, frames: np.ndarray, labels: np.ndarray, hop: int = 160, output_path: str = "results/labels.png"):
    """Plot waveform with coloured label strips and save.
    Also saves ``labels.npy`` and ``energy.npy`` for teammates.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    times = np.arange(len(x)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(times, x, label='waveform')
    frame_times = (np.arange(len(labels)) * hop + frames.shape[1] / 2) / fs
    colors = {0: 'red', 1: 'orange', 2: 'green'}
    for lbl, col in colors.items():
        mask = labels == lbl
        plt.scatter(frame_times[mask], np.zeros_like(frame_times[mask]), color=col, marker='|', s=100, label=f'Label {lbl}')
    plt.xlabel('Time (s)')
    plt.title('Waveform with frame‑level classification')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    np.save(os.path.join(os.path.dirname(output_path), 'labels.npy'), labels)
    # also save energy for teammate
    energy = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    np.save(os.path.join(os.path.dirname(output_path), 'energy.npy'), energy)
    print(f"Saved labels plot to {output_path}")
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
    """Compute and plot a Hann‑window spectrogram in dB scale.
    """
    import os
    from scipy.signal import stft, get_window

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    win_len = int(round(win_ms * fs / 1000.0))
    hop_len = int(round(hop_ms * fs / 1000.0))
    window = get_window('hann', win_len, fftbins=True)
    f, t, Z = stft(x, fs=fs, window=window, nperseg=win_len, noverlap=win_len - hop_len, nfft=win_len, detrend=False, boundary=None, padded=False)
    S_db = 20 * np.log10(np.abs(Z) + 1e-10)
    plt.figure(figsize=(8, 4))
    plt.imshow(S_db, aspect='auto', origin='lower', extent=[t.min(), t.max(), f.min(), f.max()], cmap='viridis')
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
