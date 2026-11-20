import os
import sys

import numpy as np
import librosa
import soundfile as sf
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.framing import load_audio, frame_signal, frame_energy, frame_zcr
from src.detect import (voicing_strength, classify_frames, calibrate_thresholds,
                        label_frame, save_handoff, plot_spectrogram)
from tests.fixtures import pseudo_whisper

FS = 16000
PLAN = [("sil", 0.4), ("a", 0.25), ("s", 0.12), ("i", 0.25), ("sil", 0.6),
        ("u", 0.25), ("s", 0.1), ("a", 0.3), ("sil", 0.3)]
EXPECTED = {"sil": 0, "s": 1, "a": 2, "i": 2, "u": 2}


@pytest.fixture(scope="module")
def fixture_feats():
    x, fs = pseudo_whisper()
    frames = frame_signal(x)
    return x, frames, frame_energy(frames), frame_zcr(frames)


def segment_slices(margin=3):
    out = []
    t = 0.0
    for kind, dur in PLAN:
        a = int(round(t * 100)) + margin
        b = int(round((t + dur) * 100)) - margin
        out.append((kind, a, b))
        t += dur
    return out


def test_load_audio_stereo_44k(tmp_path):
    fs_in = 44100
    t = np.arange(int(1.5 * fs_in)) / fs_in
    left = 0.4 * np.sin(2 * np.pi * 440 * t)
    right = 0.2 * np.sin(2 * np.pi * 440 * t)
    path = str(tmp_path / "stereo.wav")
    sf.write(path, np.stack([left, right], axis=1), fs_in)
    x, fs = load_audio(path)
    assert fs == 16000
    assert x.dtype == np.float32
    assert x.ndim == 1
    assert abs(len(x) - 1.5 * 16000) <= 2
    assert np.abs(x).max() <= 1.0


def test_frame_signal_shape_and_librosa():
    x, _ = pseudo_whisper()
    frames = frame_signal(x)
    n = 1 + (len(x) - 320) // 160
    assert frames.shape == (n, 320)
    assert frames.dtype == np.float32
    ref = librosa.util.frame(x, frame_length=320, hop_length=160).T
    assert np.array_equal(frames, ref)


def test_frame_signal_is_copy():
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    frames = frame_signal(x)
    before = frames.copy()
    x[:] = 0
    assert np.array_equal(frames, before)
    assert not np.shares_memory(frames, x)


def test_frame_signal_short_input():
    assert frame_signal(np.zeros(100, dtype=np.float32)).shape == (0, 320)
    assert frame_signal(np.zeros(320, dtype=np.float32)).shape == (1, 320)


def test_energy_and_zcr_vs_librosa(fixture_feats):
    x, frames, energy, zcr = fixture_feats
    rms_ref = librosa.feature.rms(y=x, frame_length=320, hop_length=160, center=False)[0]
    zcr_ref = librosa.feature.zero_crossing_rate(x, frame_length=320, hop_length=160, center=False)[0]
    assert np.allclose(energy, rms_ref, atol=1e-6), "frame_energy must equal librosa rms"
    assert np.allclose(zcr * 319 / 320, zcr_ref, atol=1e-6), (
        "frame_zcr divides crossings by 319 pairs, librosa by 320 samples, so scale by 319/320")
    assert np.abs(zcr - zcr_ref).max() < 0.5 / 320 + 1e-6 + 0.01


def test_voicing_strength_sine_vs_noise():
    t = np.arange(8000) / FS
    sine = np.sin(2 * np.pi * 120 * t).astype(np.float32)
    noise = np.random.default_rng(1).standard_normal(8000).astype(np.float32)
    v_sine = voicing_strength(frame_signal(sine))
    v_noise = voicing_strength(frame_signal(noise))
    assert np.all(v_sine > 0.95)
    assert np.all(v_noise < 0.45)
    assert np.median(v_noise) < 0.35


def test_voicing_strength_whisper_vs_voiced():
    xw, _ = pseudo_whisper()
    xv, _ = pseudo_whisper(voiced=True)
    vw = voicing_strength(frame_signal(xw))
    vv = voicing_strength(frame_signal(xv))
    vowels = [(a, b) for k, a, b in segment_slices() if k in ("a", "i", "u")]
    mw = np.mean([vw[a:b].mean() for a, b in vowels])
    mv = np.mean([vv[a:b].mean() for a, b in vowels])
    assert mv > mw + 0.1


def test_classify_contract_labels_only(fixture_feats):
    _, frames, energy, zcr = fixture_feats
    labels = classify_frames(frames, energy)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (len(frames),)
    assert np.issubdtype(labels.dtype, np.integer)
    assert set(np.unique(labels)) <= {0, 1, 2}
    assert np.array_equal(labels, classify_frames(frames, energy, zcr))


def test_classify_return_thresholds(fixture_feats):
    _, frames, energy, zcr = fixture_feats
    out = classify_frames(frames, energy, zcr, return_thresholds=True)
    assert len(out) == 3
    labels, zcr_thr, silence_thr = out
    assert labels.shape == (len(frames),)
    assert 0 < silence_thr < 0.05
    assert 0 < zcr_thr < 1


def test_classify_fixture_segments(fixture_feats):
    _, frames, energy, _ = fixture_feats
    labels = classify_frames(frames, energy)
    for kind, a, b in segment_slices():
        frac = np.mean(labels[a:b] == EXPECTED[kind])
        assert frac > 0.75, f"segment {kind} at frames {a}-{b}: only {frac:.2f} correct"


def test_classify_explicit_zcr_threshold(fixture_feats):
    _, frames, energy, _ = fixture_feats
    all_noisy = classify_frames(frames, energy, zcr_thr=0.0)
    assert not np.any(all_noisy == 2)
    none_noisy = classify_frames(frames, energy, zcr_thr=1.0)
    assert not np.any(none_noisy == 1)


def test_clip_starting_with_speech_not_all_silence():
    x, _ = pseudo_whisper()
    y = x[int(0.4 * FS):]
    frames = frame_signal(y)
    energy = frame_energy(frames)
    labels = classify_frames(frames, energy)
    assert np.mean(labels == 0) < 0.5
    assert np.mean(labels[2:20] != 0) > 0.8


def test_calibrate_guard_on_speech_start():
    x, _ = pseudo_whisper()
    y = x[int(0.4 * FS):]
    frames = frame_signal(y)
    energy = frame_energy(frames)
    zcr = frame_zcr(frames)
    silence_thr, zcr_thr = calibrate_thresholds(energy, zcr)
    assert silence_thr < 0.03
    naive = 3 * np.median(energy[:int(0.3 * FS // 160)])
    assert naive > silence_thr


def test_calibrate_empty_and_leading_silence(fixture_feats):
    assert calibrate_thresholds(np.array([]), np.array([])) == (1e-4, 0.5)
    _, _, energy, zcr = fixture_feats
    silence_thr, zcr_thr = calibrate_thresholds(energy, zcr)
    assert 0.004 < silence_thr < 0.01


def test_label_frame_matches_pre_smoothing(fixture_feats):
    _, frames, energy, zcr = fixture_feats
    silence_thr, zcr_thr = calibrate_thresholds(energy, zcr)
    per_hop = np.array([label_frame(e, z, silence_thr, zcr_thr) for e, z in zip(energy, zcr)])
    vec = np.where(energy < silence_thr, 0, np.where(zcr > zcr_thr, 1, 2))
    assert np.array_equal(per_hop, vec)
    labels, zt, st = classify_frames(frames, energy, zcr, return_thresholds=True)
    assert zt == pytest.approx(zcr_thr)
    assert st == pytest.approx(silence_thr)
    assert np.mean(labels == per_hop) > 0.9
    assert label_frame(0.0, 0.9, 0.1, 0.5) == 0
    assert label_frame(1.0, 0.9, 0.1, 0.5) == 1
    assert label_frame(1.0, 0.1, 0.1, 0.5) == 2


def test_hangover_is_bounded():
    n = 60
    energy = np.full(n, 0.001)
    energy[20:30] = 0.5
    zcr = np.full(n, 0.05)
    labels = classify_frames(np.zeros((n, 320), dtype=np.float32), energy, zcr)
    assert np.all(labels[:20] == 0)
    assert np.all(labels[36:] == 0)


def test_save_handoff(tmp_path):
    labels = np.array([0, 1, 2])
    energy = np.array([0.1, 0.2, 0.3])
    out = save_handoff(labels, energy, out_dir=str(tmp_path / "h"))
    assert np.array_equal(np.load(os.path.join(out, "labels.npy")), labels)
    assert np.array_equal(np.load(os.path.join(out, "energy.npy")), energy)


def test_plot_spectrogram_shape(tmp_path):
    x, fs = pseudo_whisper()
    S, f, t = plot_spectrogram(x, fs, output_path=str(tmp_path / "s.png"))
    assert S.shape == (len(f), len(t))
    assert os.path.exists(tmp_path / "s.png")


def test_load_audio_removes_dc_and_classifier_survives_offset(tmp_path):
    import soundfile
    x, fs = pseudo_whisper()
    base = classify_frames(frame_signal(x), frame_energy(frame_signal(x)))
    p = tmp_path / "dc.wav"
    soundfile.write(p, np.clip(x + 0.3, -1, 1), fs)
    y, _ = load_audio(str(p))
    assert abs(float(np.mean(y[4000:]))) < 0.01
    lab = classify_frames(frame_signal(y), frame_energy(frame_signal(y)))
    assert np.mean(lab == base) > 0.8


def test_dc_block_starts_in_steady_state_and_carries_state():
    from src.framing import dc_block
    x = np.full(2000, 0.3, dtype=np.float32) + 0.01 * np.random.default_rng(0).standard_normal(2000).astype(np.float32)
    y, _ = dc_block(x)
    assert np.max(np.abs(y[:400])) < 0.05
    one, _ = dc_block(x, np.zeros(1))
    parts, zi = [], np.zeros(1)
    for a, b in ((0, 0), (0, 1), (1, 160), (160, 160), (160, 2000)):
        yp, zi = dc_block(x[a:b], zi)
        parts.append(yp)
    np.testing.assert_allclose(np.concatenate(parts), one, atol=1e-6)
