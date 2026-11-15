import numpy as np


def _rosenberg_pulse(T0):
    T0 = int(round(T0))
    pulse = np.zeros(T0, dtype=np.float64)
    Tp = int(round(0.40 * T0))
    Tn = int(round(0.16 * T0))
    for n in range(Tp):
        pulse[n] = 0.5 * (1.0 - np.cos(np.pi * n / Tp))
    for n in range(Tn):
        idx = Tp + n
        if idx < T0:
            pulse[idx] = np.cos(np.pi * n / (2.0 * Tn))
    diff = np.diff(pulse, prepend=0.0)
    peak = np.max(np.abs(diff))
    if peak > 0:
        diff = diff / peak
    return diff


def build_f0(labels, energy, base_f0=120, decl=0.12, stress=0.25):
    n_frames = len(labels)
    f0 = np.zeros(n_frames, dtype=np.float64)

    voiced = labels == 2
    if not np.any(voiced):
        return f0.astype(np.float32)

    voiced_energy = energy[voiced]
    mean_energy = np.mean(voiced_energy) if np.mean(voiced_energy) > 0 else 1.0

    n_voiced = np.sum(voiced)
    decl_curve = np.linspace(0.0, decl, n_voiced)

    rng = np.random.default_rng(seed=42)
    jitter = rng.uniform(-0.005, 0.005, size=n_voiced)

    voiced_indices = np.where(voiced)[0]
    for k, i in enumerate(voiced_indices):
        stress_factor = stress * (energy[i] / mean_energy - 1.0)
        raw = base_f0 * (1.0 - decl_curve[k] + stress_factor + jitter[k])
        f0[i] = np.clip(raw, 70.0, 400.0)

    smoothed = f0.copy()
    for i in range(1, n_frames - 1):
        if voiced[i]:
            smoothed[i] = np.median(f0[max(0, i - 2):i + 3][voiced[max(0, i - 2):i + 3]])
            if smoothed[i] == 0:
                smoothed[i] = f0[i]

    kernel = np.array([0.25, 0.5, 0.25])
    for i in range(1, n_frames - 1):
        if voiced[i]:
            win = smoothed[max(0, i - 1):i + 2]
            smoothed[i] = np.convolve(win, kernel[:len(win)], mode='valid')[0] if len(win) == 3 else smoothed[i]

    smoothed = np.clip(smoothed, 0.0, 400.0)
    smoothed[~voiced] = 0.0

    return smoothed.astype(np.float32)


def make_excitation(labels, f0, hop=160, fs=16000):
    n_frames = len(labels)
    n_samples = n_frames * hop
    e = np.zeros(n_samples, dtype=np.float32)

    phase = 0.0
    rng = np.random.default_rng(seed=0)

    for i in range(n_frames):
        start = i * hop
        end = start + hop
        label = labels[i]
        frame_f0 = float(f0[i]) if f0[i] > 0 else 0.0

        if label == 0:
            phase = 0.0
            continue

        if label == 1:
            noise = rng.standard_normal(hop).astype(np.float32)
            rms = np.sqrt(np.mean(noise ** 2))
            if rms > 0:
                noise /= rms
            e[start:end] = noise
            phase = 0.0
            continue

        if label == 2 and frame_f0 > 0:
            T0 = fs / frame_f0
            pulse = _rosenberg_pulse(T0)
            pulse_len = len(pulse)

            aspiration_frac = rng.uniform(0.10, 0.20)
            noise = rng.standard_normal(hop).astype(np.float32)

            voiced_buf = np.zeros(hop, dtype=np.float64)
            t = phase
            while t < hop:
                idx = int(round(t))
                if idx < hop:
                    end_idx = min(idx + pulse_len, hop)
                    copy_len = end_idx - idx
                    voiced_buf[idx:end_idx] += pulse[:copy_len]
                t += T0

            phase = t - hop

            voiced_rms = np.sqrt(np.mean(voiced_buf ** 2))
            if voiced_rms > 0:
                voiced_buf /= voiced_rms

            noise_rms = np.sqrt(np.mean(noise ** 2))
            if noise_rms > 0:
                noise /= noise_rms

            mixed = (1.0 - aspiration_frac) * voiced_buf + aspiration_frac * noise.astype(np.float64)
            mix_rms = np.sqrt(np.mean(mixed ** 2))
            if mix_rms > 0:
                mixed /= mix_rms

            e[start:end] = mixed.astype(np.float32)
        else:
            noise = rng.standard_normal(hop).astype(np.float32)
            noise_rms = np.sqrt(np.mean(noise ** 2))
            if noise_rms > 0:
                noise /= noise_rms
            e[start:end] = noise
            phase = 0.0

    return e
