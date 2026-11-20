import os
import sys
import time

import numpy as np
import scipy.linalg
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import lpc
from tests.fixtures import FS, VOWELS, formant_filter, pseudo_whisper
from src.mocks import mock_frames


def _fixture_frames():
    x, _ = pseudo_whisper()
    return mock_frames(x)


def _vowel_frame(vowel="a", seed=3, n=320):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(4000)
    y = scipy.signal.lfilter([1.0], formant_filter(VOWELS[vowel]), e)
    return y[2000:2000 + n]


def _peaks(freqs, db, k):
    idx, _ = scipy.signal.find_peaks(db)
    idx = idx[np.argsort(db[idx])[::-1]][:k]
    return np.sort(freqs[idx])


def test_preemphasis():
    x = np.random.default_rng(0).standard_normal(50)
    y = lpc.preemphasis(x, 0.9)
    assert y[0] == x[0]
    assert np.allclose(y[1:], x[1:] - 0.9 * x[:-1])


def test_autocorr_matches_numpy():
    rng = np.random.default_rng(1)
    for _ in range(10):
        x = rng.standard_normal(320)
        r = lpc.autocorr(x, 16)
        full = np.correlate(x, x, "full")[len(x) - 1:]
        assert np.allclose(r, full[:17])
        assert np.isclose(r[0], np.sum(x ** 2))


def test_levinson_matches_toeplitz():
    rng = np.random.default_rng(2)
    frames = list(_fixture_frames()[::5]) + [rng.standard_normal(320) for _ in range(20)]
    worst = 0.0
    for f in frames:
        if np.sum(f ** 2) < 1e-6:
            continue
        y = lpc.preemphasis(f) * np.hamming(320)
        for p in range(4, 25, 4):
            r = lpc.autocorr(y, p)
            r[0] = r[0] * (1 + 1e-9)
            a, err = lpc.levinson(r, p)
            ref = scipy.linalg.solve_toeplitz(r[:p], -r[1:p + 1])
            worst = max(worst, np.max(np.abs(a[1:] - ref)))
            assert a[0] == 1.0
            assert np.isclose(err, r[0] + np.dot(a[1:], r[1:]), rtol=1e-6)
    print("max levinson error", worst)
    assert worst < 1e-8


def test_poles_inside_unit_circle():
    for f in _fixture_frames():
        a, g = lpc.lpc_frame(f)
        assert np.all(np.isfinite(a)) and np.isfinite(g)
        assert np.all(np.abs(np.roots(a)) < 1.0)


def test_silent_frames():
    a, g = lpc.lpc_frame(np.zeros(320))
    assert g == 0.0
    assert a[0] == 1.0 and np.all(a[1:] == 0.0)
    assert len(a) == 13


def test_lpc_all_shapes():
    A, G = lpc.lpc_all(np.zeros((0, 320)))
    assert A.shape == (0, 13) and G.shape == (0,)
    F = _fixture_frames()
    A, G = lpc.lpc_all(F, order=10)
    assert A.shape == (len(F), 11) and G.shape == (len(F),)
    assert np.all(A[:, 0] == 1.0)
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(G))


def test_speed():
    F = _fixture_frames()
    t = time.perf_counter()
    for f in F:
        lpc.lpc_frame(f)
    ms = (time.perf_counter() - t) / len(F) * 1e3
    print("ms per frame", ms)
    assert ms < 1.0


def test_prediction_gain():
    f = _vowel_frame("a")
    a, g = lpc.lpc_frame(f)
    pg = lpc.prediction_gain_db(f, a)
    print("prediction gain dB", pg)
    assert pg > 0
    assert np.sum(lpc.residual(f, a) ** 2) < np.sum(f ** 2)


def test_gain_matches_rms():
    f = _vowel_frame("a")
    a, g = lpc.lpc_frame(f)
    rng = np.random.default_rng(5)
    e = rng.standard_normal(40000)
    y = scipy.signal.lfilter([g], a, e)
    target = lpc.preemphasis(f)
    ratio = np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(target ** 2))
    print("gain rms ratio", ratio)
    assert 0.5 < ratio < 2.0


def _deemph_db(freqs):
    _, h = scipy.signal.freqz([1.0], [1.0, -0.97], worN=freqs, fs=FS)
    return 20 * np.log10(np.abs(h))


def test_envelope_peaks_near_formants():
    errs = []
    for seed in range(5):
        f = _vowel_frame("a", seed=seed)
        a, g = lpc.lpc_frame(f, 12)
        w, db = lpc.lpc_envelope(a, g)
        p = _peaks(w, db + _deemph_db(w), 2)
        errs.append((abs(p[0] - 700) / 700, abs(p[1] - 1220) / 1220))
    errs = np.array(errs)
    print("formant rel errors (de-emphasised envelope) mean", errs.mean(axis=0), "max", errs.max(axis=0))
    assert np.all(errs.mean(axis=0) < 0.15)


def test_envelope_vs_librosa():
    import librosa
    corrs = []
    for v in VOWELS:
        f = _vowel_frame(v, seed=4)
        y = lpc.preemphasis(f) * np.hamming(len(f))
        a, g = lpc.lpc_frame(f, 12)
        a_lib = librosa.lpc(y, order=12)
        _, d1 = lpc.lpc_envelope(a, g)
        _, d2 = lpc.lpc_envelope(a_lib, 1.0)
        corrs.append(np.corrcoef(d1, d2)[0, 1])
    print("librosa envelope correlations", corrs)
    assert min(corrs) > 0.9


def test_warp_identity_and_stability():
    a, _ = lpc.lpc_frame(_vowel_frame("a"))
    assert np.allclose(lpc.warp_poles(a, 1.0), a, atol=1e-10)
    for fac in np.linspace(0.85, 1.0, 7):
        w = lpc.warp_poles(a, fac)
        assert w.shape == a.shape and w.dtype.kind == "f"
        assert w[0] == 1.0
        assert np.all(np.abs(np.roots(w)) < 1.0)


def test_warp_moves_formants():
    f = _vowel_frame("a")
    a, g = lpc.lpc_frame(f)
    w = lpc.warp_poles(a, 0.9)
    f0, d0 = lpc.lpc_envelope(a)
    f1, d1 = lpc.lpc_envelope(w)
    p0 = _peaks(f0, d0, 2)
    p1 = _peaks(f1, d1, 2)
    print("peaks", p0, p1)
    assert np.all(p1 < p0)
    assert np.allclose(p1 / p0, 0.9, atol=0.06)


def test_warp_degenerate():
    a = np.zeros(13)
    a[0] = 1.0
    assert np.array_equal(lpc.warp_poles(a, 0.9), a)


def test_lsd():
    a, g = lpc.lpc_frame(_vowel_frame("a"))
    b, h = lpc.lpc_frame(_vowel_frame("i"))
    assert lpc.log_spectral_distortion(a, a, g, g) == 0.0
    assert lpc.log_spectral_distortion(a, b, g, h) > 0.5
    A = np.stack([a, a])
    B = np.stack([a, b])
    G = np.array([g, g])
    H = np.array([g, h])
    assert lpc.lsd_frames(A, G, A, G) == 0.0
    assert lpc.lsd_frames(A, G, B, H) > 0
    assert lpc.lsd_frames(A, G, B, H, mask=np.array([True, False])) == 0.0


def test_envelope_lsd_reasonable():
    f = _vowel_frame("a")
    v4 = lpc.envelope_lsd(f, 4)
    v16 = lpc.envelope_lsd(f, 16)
    assert 0 < v16 < v4
    assert lpc.envelope_lsd(np.zeros(320)) == 0.0
