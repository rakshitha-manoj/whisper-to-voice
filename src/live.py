"""Causal per-hop whisper-to-voice pipeline. State dict carried between hops."""

import os
import time

import numpy as np

from .detect import calibrate_thresholds, label_frame
from .framing import dc_block, frame_zcr
from .lpc import lpc_frame, warp_poles
from .pitch import init_excitation_state, init_f0_state, f0_step, excitation_step
from .synth import init_output_state, output_step, synth_step

BUDGET_MS = 10.0
LIMIT_KNEE = 0.9


def default_params():
    """Parameters read every hop, so a GUI can change them while streaming."""
    return dict(base_f0=120, decl=0.12, stress=0.25, jitter=0.005, aspiration=0.15,
                warp=None, volume=1.0, order=12, calib_s=0.3, silence_mult=3.0,
                zcr_pct=70, hangover=5, zcr_thr=None, zcr_init=0.12, zcr_window_s=3.0, zcr_min_s=0.5)


def init_live_state(params=None, fs=16000, hop=160):
    p = default_params()
    if params:
        p.update(params)
    return dict(params=p, fs=fs, hop=hop,
                prev=np.zeros(hop, dtype=np.float32),
                calib_rms=[], calib_zcr=[], thresholds=None, dc_zi=np.zeros(1),
                speech_zcr=[], zcr_live=float(p["zcr_init"]), zcr_updates=0,
                last_nonsilent=0, since_nonsilent=10 ** 6,
                f0_state=init_f0_state(0), exc_state=init_excitation_state(0),
                zi=np.zeros(p["order"]), out_state=init_output_state(),
                timings=[], hop_count=0, last_label=0, last_f0=0.0)


def _live_zcr_threshold(rms, zcr, silence_thr, state):
    """ZCR threshold for live labelling.

    Calibration hears only silence, so the offline rule (a percentile of the non silent ZCRs) cannot be applied
    at stream start. Start from zcr_init, then track the zcr_pct percentile of the ZCRs of the last zcr_window_s
    seconds of non silent frames once zcr_min_s seconds of speech were heard. params zcr_thr fixes it instead.
    """
    p = state["params"]
    if p.get("zcr_thr") is not None:
        return float(p["zcr_thr"])
    hops_per_s = state["fs"] / state["hop"]
    if rms >= silence_thr:
        state["speech_zcr"].append(float(zcr))
        state["zcr_updates"] += 1
        keep = int(p["zcr_window_s"] * hops_per_s)
        if len(state["speech_zcr"]) > keep:
            del state["speech_zcr"][:len(state["speech_zcr"]) - keep]
        if len(state["speech_zcr"]) >= p["zcr_min_s"] * hops_per_s and state["zcr_updates"] % 10 == 0:
            state["zcr_live"] = float(np.percentile(state["speech_zcr"], p["zcr_pct"]))
    return state["zcr_live"]


def _soft_limit(y):
    a = np.abs(y)
    over = a > LIMIT_KNEE
    if not np.any(over):
        return y
    room = 1.0 - LIMIT_KNEE
    out = y.copy()
    out[over] = np.sign(y[over]) * (LIMIT_KNEE + room * np.tanh((a[over] - LIMIT_KNEE) / room))
    return out


def _frame_features(prev, block):
    frame = np.concatenate((prev, block)).astype(np.float32)
    rms = float(np.sqrt(np.mean(frame ** 2)))
    zcr = float(frame_zcr(frame[None, :])[0])
    return frame, rms, zcr


def _causal_hangover(label, state, hangover):
    if label != 0:
        state["last_nonsilent"] = label
        state["since_nonsilent"] = 0
        return label
    state["since_nonsilent"] += 1
    if state["since_nonsilent"] <= hangover and state["last_nonsilent"] != 0:
        return state["last_nonsilent"]
    return 0


def process_hop_live(block, state):
    """One hop in, one hop out. Zeros while calibrating, then the full causal chain."""
    t0 = time.perf_counter()
    p = state["params"]
    hop, fs = state["hop"], state["fs"]
    block, state["dc_zi"] = dc_block(np.asarray(block, dtype=np.float32).reshape(-1), state["dc_zi"])
    frame, rms, zcr = _frame_features(state["prev"], block)
    n_calib = int(p["calib_s"] * fs // hop)
    out = np.zeros(hop, dtype=np.float32)

    if state["thresholds"] is None:
        if state["hop_count"] >= 1:
            state["calib_rms"].append(rms)
            state["calib_zcr"].append(zcr)
        if state["hop_count"] + 1 >= n_calib:
            state["thresholds"] = calibrate_thresholds(
                np.array(state["calib_rms"]), np.array(state["calib_zcr"]), fs, hop,
                p["calib_s"], p["silence_mult"], p["zcr_pct"])
    else:
        silence_thr = state["thresholds"][0]
        zcr_thr = _live_zcr_threshold(rms, zcr, silence_thr, state)
        raw = label_frame(rms, zcr, silence_thr, zcr_thr)
        label = _causal_hangover(raw, state, int(p["hangover"]))
        a, g = lpc_frame(frame, order=int(p["order"]))
        if p["warp"] is not None:
            a = warp_poles(a, p["warp"])
        f0, state["f0_state"] = f0_step(label, rms, state["f0_state"], p["base_f0"], p["decl"],
                                        p["stress"], p["jitter"], hop=hop, fs=fs)
        e, state["exc_state"] = excitation_step(label, f0, state["exc_state"], hop, fs, p["aspiration"])
        if len(state["zi"]) != len(a) - 1:
            state["zi"] = np.zeros(len(a) - 1)
        y, state["zi"] = synth_step(a, float(g), e.astype(np.float64), state["zi"])
        y, state["out_state"] = output_step(y, rms, state["out_state"], p["volume"])
        out = _soft_limit(np.nan_to_num(y)).astype(np.float32)
        state["last_label"], state["last_f0"] = int(label), float(f0)

    state["prev"] = block.copy()
    state["hop_count"] += 1
    state["timings"].append((time.perf_counter() - t0) * 1000.0)
    return out, state


def process_file_live(x, params=None, fs=16000, hop=160, state=None):
    """Stream a whole signal through process_hop_live; output has the input length."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if state is None:
        state = init_live_state(params, fs, hop)
    n = len(x)
    y = np.zeros(n, dtype=np.float32)
    for i in range(0, n, hop):
        block = x[i:i + hop]
        if len(block) < hop:
            block = np.pad(block, (0, hop - len(block)))
        out, state = process_hop_live(block, state)
        y[i:i + hop] = out[:n - i]
    process_file_live.last_state = state
    return y


def latency_report(state, stream_latency_s=None):
    t = np.asarray(state["timings"], dtype=np.float64)
    hop_ms = 1000.0 * state["hop"] / state["fs"]
    algo_ms = 2.0 * hop_ms
    rep = dict(hops=int(t.size),
               mean_ms=float(t.mean()) if t.size else 0.0,
               p95_ms=float(np.percentile(t, 95)) if t.size else 0.0,
               max_ms=float(t.max()) if t.size else 0.0,
               frac_over_budget=float(np.mean(t > BUDGET_MS)) if t.size else 0.0,
               algorithmic_ms=algo_ms, device_ms=None, total_ms=algo_ms)
    if stream_latency_s is not None:
        lat = np.atleast_1d(np.asarray(stream_latency_s, dtype=np.float64))
        rep["device_ms"] = float(lat.sum() * 1000.0)
        rep["total_ms"] = algo_ms + rep["device_ms"]
    return rep


def plot_latency(state, path="results/latency.png"):
    from matplotlib.figure import Figure

    t = np.asarray(state["timings"], dtype=np.float64)
    fig = Figure(figsize=(5, 2.6))
    ax = fig.subplots()
    ax.plot(t, lw=0.8, label="hop time")
    ax.axhline(BUDGET_MS, color="r", ls="--", label="10 ms budget")
    ax.set_xlabel("hop")
    ax.set_ylabel("ms")
    ax.set_title("Per-hop processing time")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=120)
    return path
