import numpy as np
import scipy.signal

FS = 16000

VOWELS = {
    "a": [(700, 130), (1220, 70), (2600, 160)],
    "i": [(270, 60), (2290, 90), (3010, 150)],
    "u": [(300, 60), (870, 80), (2240, 120)],
}


def formant_filter(formants, fs=FS):
    a = np.array([1.0])
    for f, bw in formants:
        r = np.exp(-np.pi * bw / fs)
        theta = 2 * np.pi * f / fs
        a = np.convolve(a, [1.0, -2 * r * np.cos(theta), r * r])
    return a


def pulse_train(n, f0=120.0, fs=FS):
    e = np.zeros(n)
    e[(np.arange(0, n, fs / f0)).astype(int)] = 1.0
    return e


def _segment(kind, dur, rng, voiced, fs=FS):
    n = int(dur * fs)
    if kind == "sil":
        return 0.002 * rng.standard_normal(n)
    if kind == "s":
        b, a = scipy.signal.butter(4, 4000, "highpass", fs=fs)
        return 0.05 * scipy.signal.lfilter(b, a, rng.standard_normal(n))
    src = pulse_train(n, fs=fs) if voiced else rng.standard_normal(n)
    y = scipy.signal.lfilter([1.0], formant_filter(VOWELS[kind], fs), src)
    y = y * np.hanning(n) ** 0.25
    return 0.1 * y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def pseudo_whisper(voiced=False, seed=0):
    rng = np.random.default_rng(seed)
    plan = [("sil", 0.4), ("a", 0.25), ("s", 0.12), ("i", 0.25), ("sil", 0.6),
            ("u", 0.25), ("s", 0.1), ("a", 0.3), ("sil", 0.3)]
    x = np.concatenate([_segment(k, d, rng, voiced) for k, d in plan])
    return x.astype(np.float32), FS


def write_fixture(path, voiced=False, seed=0):
    import soundfile
    x, fs = pseudo_whisper(voiced=voiced, seed=seed)
    soundfile.write(path, x, fs)
    return path
