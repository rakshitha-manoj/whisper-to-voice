import sys
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.signal
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pitch, synth
from src.mocks import mock_frames, mock_energy
from tests.fixtures import write_fixture, formant_filter, VOWELS

FS = 16000
HOP = 160


def est_f0(x, lo=70, hi=400):
    x = x - np.mean(x)
    n = len(x)
    spec = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(np.abs(spec) ** 2)[:n]
    lags = np.arange(int(FS / hi), int(FS / lo))
    return FS / lags[np.argmax(acf[lags])]


def hop_rms(x, hop=HOP):
    n = len(x) // hop
    return np.sqrt(np.mean(x[:n * hop].astype(np.float64).reshape(n, hop) ** 2, axis=1))


def two_phrase_labels():
    labels = np.concatenate([np.zeros(10), np.full(150, 2), np.zeros(60), np.full(150, 2), np.zeros(10)]).astype(int)
    return labels


def test_impulse_train_harmonics():
    e = pitch.impulse_train(120, 1.0, FS)
    assert len(e) == FS
    E = np.abs(np.fft.rfft(e))
    freqs = np.fft.rfftfreq(len(e), 1 / FS)
    for h in range(1, 6):
        k = np.argmin(np.abs(freqs - 120 * h))
        assert E[k - 3:k + 4].max() > 0.8 * E.max()
        assert E[k - 3:k + 4].max() > 5 * E[k + int(60 * 1.0):k + int(60 * 1.0) + 3].max()
    assert abs(e.sum() - 120) <= 1


def test_rosenberg_pulse_shape():
    for T0 in (80, 133.3, 200):
        p = pitch.rosenberg_pulse(T0)
        L = int(round(T0))
        assert len(p) == L
        assert p[0] == 0.0
        assert abs(np.argmax(p) - 0.40 * L) <= 1
        assert p.max() == pytest.approx(1.0)
        closed = int(round(0.40 * L)) + int(round(0.16 * L))
        assert np.all(p[closed:] == 0.0)
        assert p.min() >= 0.0


@pytest.mark.parametrize("f0", [120.0, 200.0])
def test_excitation_f0_autocorrelation(f0):
    n = 100
    labels = np.full(n, 2)
    e = pitch.make_excitation(labels, np.full(n, f0), HOP, FS)
    est = est_f0(e[HOP * 10:HOP * 70])
    assert abs(est - f0) / f0 < 0.03


def test_excitation_glide_tracked():
    n = 150
    labels = np.full(n, 2)
    f0 = np.linspace(100, 150, n)
    e = pitch.make_excitation(labels, f0, HOP, FS)
    start = est_f0(e[HOP * 5:HOP * 30])
    end = est_f0(e[HOP * 120:HOP * 145])
    assert abs(start - np.mean(f0[5:30])) / np.mean(f0[5:30]) < 0.05
    assert abs(end - np.mean(f0[120:145])) / np.mean(f0[120:145]) < 0.05
    assert end > start * 1.25


@pytest.mark.parametrize("f0", [125.0, 120.0])
def test_hopwise_equals_single_pass(f0):
    n = 60
    labels = np.full(n, 2)
    fz = np.full(n, f0)
    hopwise = pitch.make_excitation(labels, fz, HOP, FS, aspiration=0.0)
    state = pitch.init_excitation_state()
    one, _ = pitch.excitation_step(2, f0, state, hop=n * HOP, aspiration=0.0)
    np.testing.assert_allclose(hopwise, one, atol=1e-6)


def test_hopwise_matches_independent_train():
    n = 40
    e = pitch.make_excitation(np.full(n, 2), np.full(n, 125.0), HOP, FS, aspiration=0.0)
    p = pitch.rosenberg_pulse(128)
    p = p / np.sqrt(np.mean(p ** 2))
    ref = np.zeros(n * HOP + 128)
    for k in range(0, n * HOP, 128):
        ref[k:k + 128] += p
    np.testing.assert_allclose(e, ref[:n * HOP], atol=1e-5)


@pytest.mark.parametrize("aspiration", [0.0, 0.15])
def test_no_hop_rate_modulation(aspiration):
    n = 200
    e = pitch.make_excitation(np.full(n, 2), np.full(n, 120.0), HOP, FS, aspiration=aspiration)
    r = hop_rms(e)[2:]
    assert r.min() > 0.75 and r.max() < 1.25


def test_build_f0_basic_properties():
    labels = two_phrase_labels()
    labels[100:110] = 1
    rng = np.random.default_rng(0)
    energy = 0.05 * np.exp(0.5 * rng.standard_normal(len(labels)))
    f0 = pitch.build_f0(labels, energy)
    assert f0.shape == labels.shape
    assert np.all(f0[labels != 2] == 0)
    v = f0[labels == 2]
    assert np.all((v >= 70) & (v <= 400))
    runs = labels == 2
    both = runs[1:] & runs[:-1]
    jumps = np.abs(np.diff(f0))[both] / f0[1:][both]
    assert jumps.max() < 0.15


def test_build_f0_declination_and_reset():
    labels = two_phrase_labels()
    energy = np.full(len(labels), 0.05)
    f0 = pitch.build_f0(labels, energy, stress=0.0, jitter=0.0)
    p1 = f0[10:160]
    p2 = f0[220:370]
    assert p1[:10].mean() > p1[-10:].mean() * 1.08
    assert p2[:10].mean() > p2[-10:].mean() * 1.08
    assert abs(p2[0] - p1[0]) / p1[0] < 0.02
    assert p2[0] > p1[-1] * 1.08
    assert np.all(np.diff(p1) <= 1e-4)


def test_build_f0_short_gap_does_not_reset():
    labels = np.concatenate([np.full(100, 2), np.zeros(30), np.full(100, 2)]).astype(int)
    f0 = pitch.build_f0(labels, np.full(230, 0.05), stress=0.0, jitter=0.0)
    assert f0[130] < f0[0] * 0.95


def test_build_f0_louder_is_higher():
    n = 200
    labels = np.full(n, 2)
    log_e = 0.5 * np.sin(2 * np.pi * np.arange(n) / 60)
    f0 = pitch.build_f0(labels, 0.05 * np.exp(log_e), decl=0.0, jitter=0.0)
    assert np.corrcoef(f0[5:-5], log_e[5:-5])[0, 1] > 0.9


def test_build_f0_flat_when_neutral():
    labels = two_phrase_labels()
    rng = np.random.default_rng(1)
    f0 = pitch.build_f0(labels, 0.05 * np.exp(rng.standard_normal(len(labels))), base_f0=150, decl=0, stress=0, jitter=0)
    np.testing.assert_allclose(f0[labels == 2], 150.0, atol=1e-3)
    assert np.all(f0[labels != 2] == 0)


def test_build_f0_causal_properties():
    labels = two_phrase_labels()
    labels[100:110] = 1
    rng = np.random.default_rng(0)
    energy = 0.05 * np.exp(0.5 * rng.standard_normal(len(labels)))
    f0 = pitch.build_f0_causal(labels, energy)
    assert np.all(f0[labels != 2] == 0)
    v = f0[labels == 2]
    assert np.all((v >= 70) & (v <= 400))

    energy = np.full(len(labels), 0.05)
    f0 = pitch.build_f0_causal(labels, energy, stress=0.0, jitter=0.0)
    p1, p2 = f0[10:100], f0[220:370]
    assert p1[:5].mean() > p1[-5:].mean() * 1.03
    assert abs(p2[0] - f0[10]) / f0[10] < 0.02
    assert p2[0] > f0[159] * 1.05

    flat = pitch.build_f0_causal(labels, energy, base_f0=130, decl=0, stress=0, jitter=0)
    np.testing.assert_allclose(flat[labels == 2], 130.0, atol=1e-3)

    n = 200
    log_e = 0.5 * np.sin(2 * np.pi * np.arange(n) / 60)
    f0 = pitch.build_f0_causal(np.full(n, 2), 0.05 * np.exp(log_e), decl=0.0, jitter=0.0)
    assert np.corrcoef(f0[20:], log_e[20:])[0, 1] > 0.5


def test_excitation_unit_rms_per_label():
    labels = np.tile(np.repeat([0, 1, 2], 10), 4)
    f0 = np.where(labels == 2, 130.0, 0.0)
    e = pitch.make_excitation(labels, f0, HOP, FS)
    r = hop_rms(e)
    assert len(e) == len(labels) * HOP
    assert np.all(e[np.repeat(labels == 0, HOP)] == 0)
    assert np.all(np.abs(r[labels == 1] - 1) < 1e-4)
    assert np.all(np.abs(r[labels == 2][5:] - 1) < 0.3)
    assert abs(np.mean(r[labels == 2]) - 1) < 0.1


def test_label2_without_f0_is_noise():
    state = pitch.init_excitation_state()
    e, _ = pitch.excitation_step(2, 0.0, state)
    assert abs(np.sqrt(np.mean(e.astype(float) ** 2)) - 1) < 1e-4


def test_aspiration_changes_excitation_and_is_reproducible():
    labels = np.full(50, 2)
    f0 = np.full(50, 120.0)
    a = pitch.make_excitation(labels, f0, aspiration=0.0)
    b = pitch.make_excitation(labels, f0, aspiration=0.15)
    c = pitch.make_excitation(labels, f0, aspiration=0.15)
    assert not np.allclose(a, b)
    np.testing.assert_array_equal(b, c)
    assert np.abs(hop_rms(b)[2:] - 1).max() < 0.25


def test_synthesize_hopwise_equals_lfilter():
    a = formant_filter([(700, 130), (1220, 70)])
    a = np.concatenate([a, np.zeros(13 - len(a))]) if len(a) < 13 else a
    n = 60
    A = np.tile(a, (n, 1))
    G = np.full(n, 0.7)
    rng = np.random.default_rng(0)
    e = rng.standard_normal(n * HOP)
    y = synth.synthesize(A, G, e, np.full(n, 2), np.ones(n), HOP)
    ref = scipy.signal.lfilter([0.7], a, e)
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-6)


def test_output_stage_follows_input_loudness():
    n = 300
    a = formant_filter([(700, 130), (1220, 70)])
    A = np.tile(a, (n, 1))
    labels = np.full(n, 2)
    energy = 0.02 + 0.1 * (1 + np.sin(2 * np.pi * np.arange(n) / 100))
    e = pitch.make_excitation(labels, np.full(n, 120.0), HOP, FS)
    y = synth.synthesize(A, np.ones(n), e, labels, energy, HOP)
    out = synth.output_stage(y, energy, HOP)
    r = hop_rms(out)
    ratio = r[2:] / energy[2:]
    assert ratio.min() > 0.5 and ratio.max() < 2.0
    assert np.corrcoef(np.log(r[2:]), np.log(energy[2:]))[0, 1] > 0.9


def test_output_step_radiation_and_deemphasis_applied_once():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(2 * HOP)
    rad = scipy.signal.lfilter([1.0], [1.0, -synth.DEEMPH], scipy.signal.lfilter([1.0, -0.98], [1.0], y))
    state = synth.init_output_state()
    o1, state = synth.output_step(y[:HOP], 0.1, state)
    g1 = 0.1 / np.sqrt(np.mean(rad[:HOP] ** 2))
    np.testing.assert_allclose(o1, rad[:HOP] * g1, rtol=1e-4, atol=1e-6)
    o2, state = synth.output_step(y[HOP:], 0.2, state)
    g2 = 0.2 / np.sqrt(np.mean(rad[HOP:] ** 2))
    ramp = g1 + (g2 - g1) * np.arange(1, HOP + 1) / HOP
    np.testing.assert_allclose(o2, rad[HOP:] * ramp, rtol=1e-4, atol=1e-6)


def test_output_step_digital_silence_then_onset_is_bounded():
    state = synth.init_output_state()
    z, state = synth.output_step(np.zeros(HOP), 0.002, state)
    assert np.all(z == 0)
    rng = np.random.default_rng(0)
    o, state = synth.output_step(rng.standard_normal(HOP) * 3, 0.1, state)
    assert np.abs(o).max() < 1.0


def _patch_pipeline(monkeypatch):
    import librosa
    import src.detect
    import src.framing
    import src.lpc

    def load_audio(path):
        x, fs = soundfile.read(path, dtype="float32")
        return x, fs

    def frame_signal(x, frame_len=320, hop=160):
        return mock_frames(x, frame_len, hop)

    def classify_frames(frames, energy):
        zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
        return np.where(energy < 0.01, 0, np.where(zcr > 0.25, 1, 2))

    def lpc_all(frames, order=12):
        A = np.zeros((len(frames), order + 1))
        G = np.zeros(len(frames))
        for i, f in enumerate(frames):
            a = librosa.lpc(f.astype(np.float64) + 1e-9 * np.arange(len(f)) % 1e-6, order=order)
            A[i] = a
            G[i] = np.sqrt(np.mean(scipy.signal.lfilter(a, [1.0], f.astype(np.float64)) ** 2)) + 1e-9
        return A, G

    def warp_poles(a, factor):
        return np.asarray(a) * factor ** np.arange(len(a))

    monkeypatch.setattr(src.framing, "load_audio", load_audio)
    monkeypatch.setattr(src.framing, "frame_signal", frame_signal)
    monkeypatch.setattr(src.framing, "frame_energy", mock_energy)
    monkeypatch.setattr(src.detect, "classify_frames", classify_frames)
    monkeypatch.setattr(src.lpc, "lpc_all", lpc_all)
    monkeypatch.setattr(src.lpc, "warp_poles", warp_poles)


def test_convert_end_to_end(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch)
    src_wav = write_fixture(tmp_path / "in.wav")
    out_wav = tmp_path / "out.wav"
    y, fs = synth.convert(str(src_wav), str(out_wav))
    x, _ = soundfile.read(src_wav)
    assert fs == FS
    assert len(y) == (len(x) // HOP) * HOP
    assert np.all(np.isfinite(y))
    assert np.abs(y).max() < 2.0
    assert np.abs(y).max() > 0.01
    back, _ = soundfile.read(out_wav)
    assert len(back) == len(y)
    energy = mock_energy(mock_frames(x.astype(np.float32), 320, 160))
    r = hop_rms(y)
    loud = energy > 0.03
    assert np.corrcoef(np.log(r[loud] + 1e-9), np.log(energy[loud]))[0, 1] > 0.5
    y2, _ = synth.convert(str(src_wav), str(out_wav), warp=0.92, aspiration=0.0)
    assert np.all(np.isfinite(y2))


def test_run_ablation_writes_four_wavs(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch)
    src_wav = write_fixture(tmp_path / "clip7.wav")
    res = tmp_path / "res"
    out = synth.run_ablation(str(src_wav), results_dir=str(res))
    assert set(out) == set(synth.VARIANTS)
    names = sorted(p.name for p in res.glob("*.wav"))
    assert names == sorted(f"clip7_{v}.wav" for v in synth.VARIANTS)
    out2 = synth.run_ablation(str(src_wav), results_dir=str(res), prefix="other")
    assert all(Path(p).name.startswith("other_") for _, p in out2.values())
    ys = [y for y, _ in out.values()]
    assert not np.allclose(ys[0], ys[1])
    assert not np.allclose(ys[2], ys[3])


def test_step_functions_fast_enough_for_live():
    n = 1000
    a = formant_filter([(700, 130), (1220, 70), (2600, 160)])
    a = np.concatenate([a, np.zeros(13 - len(a))])
    es, zs, os_ = pitch.init_excitation_state(), np.zeros(12), synth.init_output_state()
    fs_ = pitch.init_f0_state()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(HOP)
    t0 = time.perf_counter()
    for i in range(n):
        f0, fs_ = pitch.f0_step(2, 0.05, fs_)
        e, es = pitch.excitation_step(2, f0, es)
        y, zs = synth.synth_step(a, 0.5, e.astype(np.float64), zs)
        out, os_ = synth.output_step(y, 0.05, os_)
    ms = (time.perf_counter() - t0) / n * 1e3
    print(f"ms per hop: {ms:.3f}")
    assert ms < 2.0
