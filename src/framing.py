"""Utility functions for loading audio, framing, and simple frame‑level features.
All functions are pure (no side effects) and accept optional keyword arguments with sensible defaults.
"""

import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavfile
try:
    import soundfile as sf
except Exception:
    sf = None

DC_POLE = 0.995


def dc_block(x, zi=None, r=DC_POLE):
    """Remove DC with the one pole high pass (1 - z^-1) / (1 - r z^-1), cutoff about 13 Hz at 16 kHz.

    Returns (y, zi) so a stream can carry the filter state between hops.
    """
    if zi is None:
        zi = np.zeros(1)
    y, zi = sig.lfilter([1.0, -1.0], [1.0, -r], np.asarray(x, dtype=np.float64), zi=zi)
    return y.astype(np.float32), zi


def load_audio(path: str, target_fs: int = 16000):
    """Load an audio file, force mono, resample to *target_fs* (default 16 kHz), remove DC with dc_block.

    Returns
    -------
    x : np.ndarray, shape (n_samples,)
        Mono float32 waveform in the range [-1, 1].
    fs : int
        Sampling rate (always ``target_fs``).
    """
    if sf is not None:
        x, fs = sf.read(path, dtype="float32")
    else:
        fs, data = wavfile.read(path)
        if data.dtype.kind == 'i':
            max_val = np.iinfo(data.dtype).max
            x = data.astype(np.float32) / max_val
        else:
            x = data.astype(np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if fs != target_fs:
        g = target_fs // np.gcd(int(fs), target_fs)
        r = int(fs) // np.gcd(int(fs), target_fs)
        x = sig.resample_poly(x, up=g, down=r)
        fs = target_fs
    x, _ = dc_block(x)
    return x, fs


def frame_signal(x: np.ndarray, frame_len: int = 320, hop: int = 160):
    """Slice a 1‑D waveform into overlapping frames without windowing.

    Parameters
    ----------
    x : np.ndarray
        Input mono signal.
    frame_len : int, default 320
    hop : int, default 160

    Returns
    -------
    frames : np.ndarray, shape (n_frames, frame_len)
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    n_samples = x.shape[0]
    if n_samples < frame_len:
        return np.empty((0, frame_len), dtype=np.float32)
    return np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop].copy()


def frame_energy(frames: np.ndarray):
    """Root‑mean‑square (RMS) energy per frame.

    Returns an array of shape ``(n_frames,)``.
    """
    return np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))


def frame_zcr(frames: np.ndarray):
    """Zero‑crossing rate per frame as a fraction of sign changes.

    Fraction of the frame_len - 1 adjacent sample pairs whose sign changes (zeros count as positive).
    librosa divides by frame_len instead, so values differ by a factor frame_len / (frame_len - 1).
    Returns ``(n_frames,)``.
    """
    signs = np.sign(frames)
    signs[signs == 0] = 1
    crossings = np.abs(np.diff(signs, axis=1)) > 0
    return crossings.mean(axis=1)
