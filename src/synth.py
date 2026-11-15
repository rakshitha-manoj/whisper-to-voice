import numpy as np
import scipy.signal


def synthesize(A, G, e, labels, energy, hop=160):
    n_frames = len(labels)
    order = A.shape[1] - 1
    y = np.zeros(n_frames * hop, dtype=np.float32)
    zi = np.zeros(order, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop
        end = start + hop

        a = A[i].astype(np.float64)
        g = float(G[i])
        exc = e[start:end].astype(np.float64)

        frame_out, zi = scipy.signal.lfilter([g], a, exc, zi=zi)
        y[start:end] = frame_out.astype(np.float32)

    return y


def _radiation_filter(y):
    return scipy.signal.lfilter([1.0, -0.98], [1.0], y).astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def output_stage(y, energy_in, hop=160, volume=1.0):
    n_frames = len(energy_in)
    y_rad = _radiation_filter(y)
    out = np.zeros_like(y_rad)
    for i in range(n_frames):
        start = i * hop
        end = start + hop
        seg = y_rad[start:end]
        seg_rms = _rms(seg)
        target_rms = float(energy_in[i]) * volume
        if seg_rms > 1e-8:
            out[start:end] = seg * (target_rms / seg_rms)
        else:
            out[start:end] = seg
    return out


def convert(path_in, path_out, volume=1.0):
    import soundfile
    from src.framing import load_audio, frame_signal, frame_energy
    from src.detect import classify_frames
    from src.lpc import lpc_all, warp_poles
    from src.pitch import build_f0, make_excitation

    x, fs = load_audio(path_in)
    frames = frame_signal(x)
    energy = frame_energy(frames)
    labels = classify_frames(frames, energy)
    A, G = lpc_all(frames, order=12)
    f0 = build_f0(labels, energy, base_f0=120)
    e = make_excitation(labels, f0)
    y_raw = synthesize(A, G, e, labels, energy)
    y_out = output_stage(y_raw, energy, volume=volume)
    soundfile.write(path_out, y_out, fs)
    return y_out, fs


def _run_variant(x, fs, frames, energy, labels, A, G, name, results_dir,
                 fixed_f0=False, with_contour=False, with_aspiration=False,
                 with_warp=False, hop=160):
    import soundfile
    from src.lpc import warp_poles
    from src.pitch import build_f0, make_excitation

    n_frames = len(labels)

    if fixed_f0:
        f0 = np.where(labels == 2, 120.0, 0.0).astype(np.float32)
    elif with_contour:
        f0 = build_f0(labels, energy, base_f0=120)
    else:
        f0 = build_f0(labels, energy, base_f0=120)

    if with_aspiration:
        e = make_excitation(labels, f0, hop=hop, fs=fs)
    else:
        from src.pitch import _rosenberg_pulse
        rng = np.random.default_rng(0)
        e = np.zeros(n_frames * hop, dtype=np.float32)
        phase = 0.0
        for i in range(n_frames):
            start = i * hop
            end_s = start + hop
            lbl = labels[i]
            frame_f0 = float(f0[i]) if f0[i] > 0 else 0.0
            if lbl == 0:
                phase = 0.0
            elif lbl == 1:
                noise = rng.standard_normal(hop).astype(np.float32)
                r = np.sqrt(np.mean(noise ** 2))
                if r > 0:
                    noise /= r
                e[start:end_s] = noise
                phase = 0.0
            elif lbl == 2 and frame_f0 > 0:
                T0 = fs / frame_f0
                pulse = _rosenberg_pulse(T0)
                plen = len(pulse)
                buf = np.zeros(hop, dtype=np.float64)
                t = phase
                while t < hop:
                    idx = int(round(t))
                    if idx < hop:
                        eidx = min(idx + plen, hop)
                        buf[idx:eidx] += pulse[:eidx - idx]
                    t += T0
                phase = t - hop
                rms_b = np.sqrt(np.mean(buf ** 2))
                if rms_b > 0:
                    buf /= rms_b
                e[start:end_s] = buf.astype(np.float32)
            else:
                noise = rng.standard_normal(hop).astype(np.float32)
                r = np.sqrt(np.mean(noise ** 2))
                if r > 0:
                    noise /= r
                e[start:end_s] = noise
                phase = 0.0

    A_use = A.copy()
    if with_warp:
        A_use = np.array([warp_poles(A[i], factor=0.92) for i in range(len(A))])

    y_raw = synthesize(A_use, G, e, labels, energy, hop=hop)
    y_out = output_stage(y_raw, energy, hop=hop)

    import os
    os.makedirs(results_dir, exist_ok=True)
    out_path = f"{results_dir}/{name}.wav"
    soundfile.write(out_path, y_out, fs)
    return y_out, out_path


def run_ablation(path_in, results_dir="results"):
    from src.framing import load_audio, frame_signal, frame_energy
    from src.detect import classify_frames
    from src.lpc import lpc_all

    x, fs = load_audio(path_in)
    frames = frame_signal(x)
    energy = frame_energy(frames)
    labels = classify_frames(frames, energy)
    A, G = lpc_all(frames, order=12)

    variants = [
        dict(name="v1_robot",      fixed_f0=True,  with_contour=False, with_aspiration=False, with_warp=False),
        dict(name="v2_contour",    fixed_f0=False, with_contour=True,  with_aspiration=False, with_warp=False),
        dict(name="v3_aspiration", fixed_f0=False, with_contour=True,  with_aspiration=True,  with_warp=False),
        dict(name="v4_warp",       fixed_f0=False, with_contour=True,  with_aspiration=True,  with_warp=True),
    ]

    outputs = {}
    for v in variants:
        y, path = _run_variant(x, fs, frames, energy, labels, A, G,
                               results_dir=results_dir, **v)
        outputs[v["name"]] = (y, path)
        print(f"Saved {path}")

    return outputs


def evaluate_wer(path_in, results_dir="results", transcripts_path="data/raw/transcripts.txt"):
    import jiwer
    import os

    try:
        import whisper as openai_whisper
    except ImportError:
        print("openai-whisper not installed, skipping WER evaluation")
        return {}

    model = openai_whisper.load_model("base")

    with open(transcripts_path) as f:
        ref_lines = [l.strip() for l in f if l.strip()]
    reference = " ".join(w for line in ref_lines for w in line.split()[1:])

    def transcribe(path):
        result = model.transcribe(path)
        return result["text"].strip()

    scores = {}
    hyp_in = transcribe(path_in)
    wer_in = jiwer.wer(reference, hyp_in)
    scores["input"] = wer_in
    print(f"WER input: {wer_in:.3f}")

    variant_names = ["v1_robot", "v2_contour", "v3_aspiration", "v4_warp"]
    for name in variant_names:
        wav_path = f"{results_dir}/{name}.wav"
        if os.path.exists(wav_path):
            hyp = transcribe(wav_path)
            w = jiwer.wer(reference, hyp)
            scores[name] = w
            print(f"WER {name}: {w:.3f}")

    return scores
