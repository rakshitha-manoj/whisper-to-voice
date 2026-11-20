import numpy as np
import scipy.signal

_WINDOWS = {}
_EPS_ENERGY = 1e-12


def _hamming(n):
    if n not in _WINDOWS:
        _WINDOWS[n] = np.hamming(n)
    return _WINDOWS[n]


def preemphasis(x, alpha=0.97):
    """First order high pass y[n] = x[n] - alpha*x[n-1], y[0] = x[0]."""
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    if x.size == 0:
        return y
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y


def autocorr(frame, p):
    """Biased autocorrelation r[k] = sum_n x[n] x[n+k] for k = 0..p."""
    x = np.asarray(frame, dtype=np.float64)
    n = len(x)
    r = np.zeros(p + 1)
    for k in range(min(p, n - 1) + 1):
        r[k] = np.dot(x[:n - k], x[k:])
    return r


def levinson(r, p):
    """Levinson-Durbin recursion. Returns a (p+1,) with a[0]=1 for A(z)=1+sum a[k]z^-k, and the final prediction error."""
    a = np.zeros(p + 1)
    a[0] = 1.0
    err = float(r[0])
    if err <= 0.0:
        return a, 0.0
    for i in range(1, p + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / err
        prev = a[1:i].copy()
        a[1:i] = prev + k * prev[::-1]
        a[i] = k
        err *= 1.0 - k * k
        if err <= 0.0:
            err = 0.0
            break
    return a, float(err)


def lpc_frame(frame, order=12):
    """LPC of one frame. Returns (a, gain).

    Pre-emphasis (0.97), Hamming window, autocorrelation, white noise correction
    r[0] *= 1 + 1e-9, Levinson. Gain G = sqrt(err / sum(w**2)), so a unit RMS
    excitation through G/A(z) matches the RMS of the pre-emphasised frame
    (err is the windowed residual energy, sum(w**2) the window energy).
    A silent frame gives a = [1, 0, ...] and gain 0.
    """
    a0 = np.zeros(order + 1)
    a0[0] = 1.0
    x = np.asarray(frame, dtype=np.float64)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return a0, 0.0
    w = _hamming(len(x))
    r = autocorr(preemphasis(x) * w, order)
    if r[0] < _EPS_ENERGY:
        return a0, 0.0
    r[0] = r[0] * (1.0 + 1e-9) + _EPS_ENERGY
    a, err = levinson(r, order)
    gain = float(np.sqrt(max(err, 0.0) / np.sum(w ** 2)))
    if not np.isfinite(gain) or not np.all(np.isfinite(a)):
        return a0, 0.0
    return a, gain


def lpc_all(frames, order=12):
    """LPC for every frame. Returns A (n_frames, order+1) and G (n_frames,)."""
    frames = np.asarray(frames)
    n = len(frames)
    A = np.zeros((n, order + 1))
    A[:, 0] = 1.0
    G = np.zeros(n)
    for i in range(n):
        A[i], G[i] = lpc_frame(frames[i], order)
    return A, G


def residual(frame, a):
    """Prediction residual e = A(z) applied to the frame."""
    return scipy.signal.lfilter(a, [1.0], np.asarray(frame, dtype=np.float64))


def prediction_gain_db(frame, a):
    """10 log10 of frame energy over residual energy."""
    x = np.asarray(frame, dtype=np.float64)
    e = residual(x, a)
    return float(10 * np.log10((np.sum(x ** 2) + 1e-20) / (np.sum(e ** 2) + 1e-20)))


def lpc_envelope(a, gain=1.0, n_fft=512, fs=16000):
    """Magnitude in dB of gain/A(z) at n_fft//2+1 frequencies from 0 to fs/2."""
    w, h = scipy.signal.freqz([1.0], a, worN=n_fft // 2 + 1, fs=fs)
    mag_db = 20 * np.log10(np.abs(gain) * np.abs(h) + 1e-12)
    return w, mag_db


def log_spectral_distortion(a_ref, a_test, gain_ref=1.0, gain_test=1.0, n_fft=512):
    """RMS over frequency of the dB difference between two LPC envelopes."""
    _, d_ref = lpc_envelope(a_ref, gain_ref, n_fft)
    _, d_test = lpc_envelope(a_test, gain_test, n_fft)
    return float(np.sqrt(np.mean((d_ref - d_test) ** 2)))


def lsd_frames(A_ref, G_ref, A_test, G_test, mask=None):
    """Mean LSD over frames; mask is a boolean array (or index array) selecting frames."""
    n = len(A_ref)
    idx = np.arange(n)
    if mask is not None:
        mask = np.asarray(mask)
        idx = idx[mask] if mask.dtype == bool else mask
    if len(idx) == 0:
        return 0.0
    vals = [log_spectral_distortion(A_ref[i], A_test[i], G_ref[i], G_test[i]) for i in idx]
    return float(np.mean(vals))


def envelope_lsd(frame, order=12, n_fft=512):
    """LSD in dB between the LPC envelope (level matched by mean dB) and the FFT magnitude of the pre-emphasised windowed frame."""
    x = np.asarray(frame, dtype=np.float64)
    a, g = lpc_frame(x, order)
    if g == 0.0:
        return 0.0
    y = preemphasis(x) * _hamming(len(x))
    spec_db = 20 * np.log10(np.abs(np.fft.rfft(y, n_fft)) + 1e-12)
    _, env_db = lpc_envelope(a, 1.0, n_fft)
    env_db = env_db + (np.mean(spec_db) - np.mean(env_db))
    return float(np.sqrt(np.mean((spec_db - env_db) ** 2)))


def warp_poles(a, factor):
    """Scale the angle of complex poles by factor (radius kept, real poles unchanged). Returns real array, a[0]=1, same length."""
    a = np.asarray(a, dtype=np.float64)
    out = a.copy()
    if len(a) < 2 or not np.any(a[1:]) or factor == 1.0:
        return out
    roots = np.roots(a)
    new = []
    for z in roots:
        r = min(abs(z), 0.9999)
        if abs(z.imag) <= 1e-9 * max(1.0, abs(z)):
            new.append(complex(np.sign(z.real) * r if z.real != 0 else 0.0, 0.0))
        else:
            theta = np.angle(z) * factor
            new.append(r * np.exp(1j * theta))
    pos = [z for z in new if z.imag > 0]
    pos_conj = [np.conj(z) for z in pos]
    reals = [z for z in new if z.imag == 0]
    neg = [z for z in new if z.imag < 0]
    if len(pos) != len(neg):
        k = min(len(pos), len(neg))
        pos, neg = pos[:k], neg[:k]
        pos_conj = [np.conj(z) for z in pos]
    poly = np.poly(np.array(reals + pos + pos_conj)).real
    res = np.zeros(len(a))
    m = min(len(a), len(poly))
    res[:m] = poly[:m]
    res[0] = 1.0
    return res
