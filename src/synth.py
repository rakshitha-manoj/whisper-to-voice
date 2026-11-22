import os

import numpy as np
import scipy.signal

GAIN_CAP = 50.0
RAMP_RATIO = 2.0
SILENT_RMS = 1e-12
DEEMPH = 0.97
OUT_DC_POLE = 0.99
PEAK = 0.99

VARIANTS = {
    "v1_robot": dict(decl=0.0, stress=0.0, jitter=0.0, aspiration=0.0, warp=None),
    "v2_contour": dict(aspiration=0.0, warp=None),
    "v3_aspiration": dict(aspiration=0.15, warp=None),
    "v4_warp": dict(aspiration=0.15, warp=0.92),
}


def synth_step(a, g, e_hop, zi):
    """Filter one hop of excitation through g / A(z), carrying the filter state (C6)."""
    y, zi = scipy.signal.lfilter([g], a, e_hop, zi=zi)
    return y, zi


def synthesize(A, G, e, labels, energy, hop=160):
    """Time varying all pole synthesis by looping synth_step."""
    n = len(labels)
    zi = np.zeros(A.shape[1] - 1)
    y = np.zeros(n * hop, dtype=np.float32)
    for i in range(n):
        y_hop, zi = synth_step(A[i], float(G[i]), e[i * hop:(i + 1) * hop].astype(np.float64), zi)
        y[i * hop:(i + 1) * hop] = y_hop
    return y


def init_output_state():
    """State for output_step: radiation and de-emphasis filter memories, last gain."""
    return dict(zi=np.zeros(1), zd=np.zeros(1), zdc=np.zeros(1), gain=None)


def output_step(y_hop, target_rms, state, volume=1.0):
    """Lip radiation 1 - 0.98 z^-1 and de-emphasis 1 / (1 - 0.97 z^-1), then RMS match to target_rms * volume with a linear gain ramp.

    The LPC filter was fitted on pre-emphasised speech, so its envelope carries an extra +6 dB/octave;
    de-emphasis removes it. Without it the output is too bright and the high band hides the harmonics.
    Radiation and de-emphasis together keep a DC gain of about 0.67, and the Rosenberg flow pulse has a large mean,
    so a DC blocker (cutoff about 25 Hz) follows; otherwise the RMS match would spend loudness on inaudible DC.
    """
    y_rad, state["zi"] = scipy.signal.lfilter([1.0, -0.98], [1.0], y_hop, zi=state["zi"])
    y_rad, state["zd"] = scipy.signal.lfilter([1.0], [1.0, -DEEMPH], y_rad, zi=state["zd"])
    y_rad, state["zdc"] = scipy.signal.lfilter([1.0, -1.0], [1.0, -OUT_DC_POLE], y_rad, zi=state["zdc"])
    rms = float(np.sqrt(np.mean(y_rad ** 2)))
    if rms < SILENT_RMS:
        return y_rad.astype(np.float32), state
    g_new = min(target_rms * volume / rms, GAIN_CAP)
    g_prev = g_new if state["gain"] is None else min(max(state["gain"], g_new / RAMP_RATIO), g_new * RAMP_RATIO)
    n = len(y_hop)
    gains = g_prev + (g_new - g_prev) * np.arange(1, n + 1) / n
    state["gain"] = g_new
    return (y_rad * gains).astype(np.float32), state


def output_stage(y, energy_in, hop=160, volume=1.0):
    """Radiation plus loudness following the input energy, by looping output_step (C7)."""
    state = init_output_state()
    out = np.zeros(len(energy_in) * hop, dtype=np.float32)
    for i in range(len(energy_in)):
        out[i * hop:(i + 1) * hop], state = output_step(
            y[i * hop:(i + 1) * hop].astype(np.float64), float(energy_in[i]), state, volume)
    return out


def _features(path_in, order=12):
    from src.framing import load_audio, frame_signal, frame_energy
    from src.detect import classify_frames
    from src.lpc import lpc_all

    x, fs = load_audio(path_in)
    frames = frame_signal(x)
    energy = frame_energy(frames)
    labels = classify_frames(frames, energy)
    A, G = lpc_all(frames, order=order)
    return dict(fs=fs, n=len(x), energy=energy, labels=np.asarray(labels), A=A, G=G)


def _render(feats, volume=1.0, base_f0=120, decl=0.12, stress=0.25, jitter=0.005,
            aspiration=0.15, warp=None, hop=160):
    from src.pitch import build_f0, make_excitation

    labels, energy, A, G = feats["labels"], feats["energy"], feats["A"], feats["G"]
    if warp is not None:
        from src.lpc import warp_poles
        A = np.array([warp_poles(a, warp) for a in A]).reshape(A.shape)
    f0 = build_f0(labels, energy, base_f0=base_f0, decl=decl, stress=stress, jitter=jitter, hop=hop)
    e = make_excitation(labels, f0, hop=hop, aspiration=aspiration)
    y_raw = synthesize(A, G, e, labels, energy, hop=hop)
    y = output_stage(y_raw, energy, hop=hop, volume=volume)
    n = feats.get("n", len(y))
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    return peak_protect(y)


def peak_protect(y, peak=PEAK):
    """Scale the whole signal down if any sample would clip when written to a wav."""
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y * (peak / m)).astype(np.float32) if m > peak else y.astype(np.float32)


def convert(path_in, path_out, volume=1.0, base_f0=120, decl=0.12, stress=0.25,
            aspiration=0.15, warp=None, order=12):
    """Whisper wav in, voiced wav out. Returns (y, fs)."""
    import soundfile

    feats = _features(path_in, order)
    y = _render(feats, volume=volume, base_f0=base_f0, decl=decl, stress=stress,
                aspiration=aspiration, warp=warp)
    soundfile.write(path_out, y, feats["fs"])
    return y, feats["fs"]


def render_variant(feats, name, volume=1.0, base_f0=120):
    """Render one named ablation variant from precomputed features."""
    return _render(feats, volume=volume, base_f0=base_f0, **VARIANTS[name])


def run_ablation(path_in, results_dir="results", prefix=None, order=12):
    """Write the four ablation variants as {prefix}_{variant}.wav; returns {variant: (y, path)}."""
    import soundfile

    if prefix is None:
        prefix = os.path.splitext(os.path.basename(path_in))[0]
    os.makedirs(results_dir, exist_ok=True)
    feats = _features(path_in, order)
    outputs = {}
    for name in VARIANTS:
        y = render_variant(feats, name)
        path = os.path.join(results_dir, f"{prefix}_{name}.wav")
        soundfile.write(path, y, feats["fs"])
        outputs[name] = (y, path)
        print(f"Saved {path}")
    return outputs
