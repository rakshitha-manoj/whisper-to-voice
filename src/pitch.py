import functools

import numpy as np
import scipy.ndimage

F_MIN_EXC = 40.0
F_MAX_EXC = 800.0
TAIL_LEN = 401
PHASE_INIT = -1e-6
PHASE_EPS = 1e-9


def impulse_train(f0, dur, fs=16000):
    """Unit impulses at f0 Hz for dur seconds (C1)."""
    n = int(round(dur * fs))
    e = np.zeros(n)
    if f0 <= 0 or n == 0:
        return e
    idx = np.round(np.arange(0, n, fs / f0)).astype(int)
    e[idx[idx < n]] = 1.0
    return e


def rosenberg_pulse(T0, open_frac=0.40, close_frac=0.16):
    """Glottal flow pulse of length round(T0): raised cosine opening, cosine closing, then closed (C2)."""
    L = int(round(T0))
    pulse = np.zeros(L)
    Tp = int(round(open_frac * L))
    Tn = int(round(close_frac * L))
    n = np.arange(Tp)
    pulse[:Tp] = 0.5 * (1.0 - np.cos(np.pi * n / max(Tp, 1)))
    m = np.arange(min(Tn, L - Tp))
    pulse[Tp:Tp + len(m)] = np.cos(np.pi * m / (2.0 * max(Tn, 1)))
    return pulse


@functools.lru_cache(maxsize=512)
def _unit_pulse(L):
    p = rosenberg_pulse(L)
    rms = np.sqrt(np.mean(p ** 2))
    return p / rms if rms > 0 else p


def _phrases(labels, gap_frames):
    n = len(labels)
    sil = (labels == 0).astype(int)
    d = np.diff(np.concatenate(([0], sil, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    cuts = [int(e) for s, e in zip(starts, ends) if e - s > gap_frames and e < n]
    edges = [0] + cuts + [n]
    out = []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        active = np.flatnonzero(labels[b0:b1] != 0)
        if active.size:
            out.append((b0 + int(active[0]), b0 + int(active[-1])))
    return out


def _smooth_runs(f0, voiced):
    out = f0.copy()
    d = np.diff(np.concatenate(([0], voiced.astype(int), [0])))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        seg = scipy.ndimage.median_filter(f0[s:e], size=5, mode="nearest")
        out[s:e] = scipy.ndimage.uniform_filter1d(seg, size=3, mode="nearest")
    return out


def build_f0(labels, energy, base_f0=120, decl=0.12, stress=0.25, jitter=0.005,
             phrase_gap_s=0.5, hop=160, fs=16000, seed=0):
    """Offline f0 contour per frame: per phrase linear declination, energy stress, jitter, smoothing (C4)."""
    labels = np.asarray(labels)
    energy = np.asarray(energy, dtype=np.float64)
    n = len(labels)
    f0 = np.zeros(n)
    voiced = labels == 2
    if not np.any(voiced):
        return f0.astype(np.float32)
    rng = np.random.default_rng(seed)
    log_e = np.log(energy + 1e-8)
    gap_frames = phrase_gap_s * fs / hop
    for first, last in _phrases(labels, gap_frames):
        idx = np.flatnonzero(voiced[first:last + 1]) + first
        if idx.size == 0:
            continue
        span = max(last - first, 1)
        fall = decl * (idx - first) / span
        dev = np.clip(log_e[idx] - log_e[idx].mean(), -1.0, 1.0)
        jit = 1.0 + jitter * rng.uniform(-1.0, 1.0, idx.size)
        f0[idx] = base_f0 * (1.0 - fall) * (1.0 + stress * dev) * jit
    f0 = _smooth_runs(f0, voiced)
    f0 = np.where(voiced, np.clip(f0, 70.0, 400.0), 0.0)
    return f0.astype(np.float32)


def init_f0_state(seed=0):
    """State for the causal f0 tracker."""
    return dict(t=0.0, started=False, sil=0.0, mean=0.0, count=0, prev=0.0,
                rng=np.random.default_rng(seed))


def f0_step(label, energy, state, base_f0=120, decl=0.12, stress=0.25, jitter=0.005,
            phrase_gap_s=0.5, phrase_dur_s=2.0, smooth=0.5, hop=160, fs=16000):
    """Causal f0 for one hop: declination from running phrase time, stress vs running mean energy."""
    dt = hop / fs
    if label == 0:
        state["sil"] += dt
        state["prev"] = 0.0
        if state["sil"] > phrase_gap_s:
            state.update(t=0.0, started=False, mean=0.0, count=0)
        elif state["started"]:
            state["t"] += dt
        return 0.0, state
    state["sil"] = 0.0
    state["started"] = True
    t = state["t"]
    state["t"] += dt
    if label != 2:
        state["prev"] = 0.0
        return 0.0, state
    log_e = float(np.log(energy + 1e-8))
    state["count"] = min(state["count"] + 1, 200)
    state["mean"] += (log_e - state["mean"]) / state["count"]
    dev = min(max(log_e - state["mean"], -1.0), 1.0)
    fall = decl * min(t / phrase_dur_s, 1.0)
    jit = 1.0 + jitter * state["rng"].uniform(-1.0, 1.0)
    raw = base_f0 * (1.0 - fall) * (1.0 + stress * dev) * jit
    f = raw if state["prev"] <= 0 else state["prev"] + smooth * (raw - state["prev"])
    f = min(max(f, 70.0), 400.0)
    state["prev"] = f
    return f, state


def build_f0_causal(labels, energy, base_f0=120, decl=0.12, stress=0.25, jitter=0.005,
                    phrase_gap_s=0.5, phrase_dur_s=2.0, hop=160, fs=16000, seed=0):
    """Run f0_step over a whole label track (for testing the causal tracker)."""
    state = init_f0_state(seed)
    out = np.zeros(len(labels), dtype=np.float32)
    for i in range(len(labels)):
        out[i], state = f0_step(labels[i], energy[i], state, base_f0, decl, stress, jitter,
                                phrase_gap_s, phrase_dur_s, hop=hop, fs=fs)
    return out


def init_excitation_state(seed=0):
    """State for excitation_step: running phase, previous f0, pulse tail, RNG."""
    return dict(phase=PHASE_INIT, prev_f0=0.0, tail=np.zeros(TAIL_LEN),
                rng=np.random.default_rng(seed))


def _unit_noise(rng, n):
    x = rng.standard_normal(n)
    r = np.sqrt(np.mean(x ** 2))
    return x / r if r > 0 else x


def _voiced_hop(f0, state, hop, fs, aspiration):
    f0 = min(max(float(f0), F_MIN_EXC), F_MAX_EXC)
    prev = state["prev_f0"]
    if prev > 0:
        f_s = prev + (f0 - prev) * np.arange(1, hop + 1) / hop
    else:
        f_s = np.full(hop, f0)
    ph = state["phase"] + np.cumsum(f_s / fs)
    k = np.floor(ph + PHASE_EPS)
    k_prev = np.floor(state["phase"] + PHASE_EPS)
    onsets = np.flatnonzero(np.diff(k, prepend=k_prev) > 0)
    buf = np.zeros(hop + TAIL_LEN)
    buf[:TAIL_LEN] += state["tail"]
    for n in onsets:
        L = int(round(fs / f_s[n]))
        buf[n:n + L] += _unit_pulse(L)
    state["phase"] = float(ph[-1] - k[-1])
    state["prev_f0"] = f0
    state["tail"] = buf[hop:].copy()
    out = buf[:hop]
    a = min(max(aspiration, 0.0), 1.0)
    if a > 0:
        out = ((1.0 - a) * out + a * _unit_noise(state["rng"], hop)) / np.sqrt((1.0 - a) ** 2 + a ** 2)
    return out


def excitation_step(label, f0, state, hop=160, fs=16000, aspiration=0.15):
    """One hop of excitation. 2 with f0>0: pulses plus aspiration; other nonzero labels: white noise; 0: zeros.
    Phase and the pulse tail are reset on any non-pulse hop, so a pulse cut by the end of a voiced run is dropped
    and silence is exactly zero."""
    if label == 2 and f0 > 0:
        return _voiced_hop(f0, state, hop, fs, aspiration).astype(np.float32), state
    state["phase"] = PHASE_INIT
    state["prev_f0"] = 0.0
    state["tail"] = np.zeros(TAIL_LEN)
    if label == 0:
        return np.zeros(hop, dtype=np.float32), state
    return _unit_noise(state["rng"], hop).astype(np.float32), state


def make_excitation(labels, f0, hop=160, fs=16000, aspiration=0.15, seed=0):
    """Excitation for a whole label track by looping excitation_step (C5)."""
    n = len(labels)
    e = np.zeros(n * hop, dtype=np.float32)
    state = init_excitation_state(seed)
    for i in range(n):
        e[i * hop:(i + 1) * hop], state = excitation_step(labels[i], f0[i], state, hop, fs, aspiration)
    return e
