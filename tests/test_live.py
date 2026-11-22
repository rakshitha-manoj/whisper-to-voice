import os
import subprocess
import sys

import numpy as np
import pytest

from src.detect import voicing_strength
from src.framing import frame_energy, frame_signal
from src.live import (default_params, init_live_state, latency_report, plot_latency,
                      process_file_live, process_hop_live)
from src.synth import convert
from tests.fixtures import pseudo_whisper, write_fixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = 16000
HOP = 160
VOWEL_SPANS = [(0.4, 0.65), (0.77, 1.02), (1.62, 1.87), (1.97, 2.27)]


@pytest.fixture(scope="module")
def whisper():
    return pseudo_whisper()[0]


@pytest.fixture(scope="module")
def live_run(whisper):
    state = init_live_state()
    y = process_file_live(whisper, state=state)
    return y, state


def _vowel_mask(n_frames):
    t = np.arange(n_frames) * HOP / FS + 0.01
    m = np.zeros(n_frames, dtype=bool)
    for a, b in VOWEL_SPANS:
        m |= (t >= a + 0.03) & (t <= b - 0.05)
    return m


def _f0_autocorr(seg):
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    lo, hi = FS // 400, FS // 70
    return FS / (lo + int(np.argmax(ac[lo:hi])))


def test_output_basic(whisper, live_run):
    y, _ = live_run
    assert len(y) == len(whisper)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 1.0
    assert np.all(y[:int(0.3 * FS)] == 0)
    assert np.max(np.abs(y)) > 0.01


def test_loudness_follows_input(whisper, live_run):
    y, _ = live_run
    ein = frame_energy(frame_signal(whisper))
    eout = frame_energy(frame_signal(y))
    mask = np.arange(len(ein)) * HOP / FS > 0.35
    mask &= ein > 0.02
    c = np.corrcoef(np.log(ein[mask]), np.log(eout[mask] + 1e-8))[0, 1]
    assert c > 0.85


def test_output_is_voiced(whisper, live_run):
    y, _ = live_run
    vin = voicing_strength(frame_signal(whisper))
    vout = voicing_strength(frame_signal(y))
    m = _vowel_mask(len(vin))
    assert vout[m].mean() > vin[m].mean() + 0.1


def test_timing(live_run):
    _, state = live_run
    rep = latency_report(state, stream_latency_s=(0.01, 0.02))
    print(f"\nper-hop ms mean {rep['mean_ms']:.3f} p95 {rep['p95_ms']:.3f} max {rep['max_ms']:.3f}")
    assert rep["mean_ms"] < 2.0
    assert rep["p95_ms"] < 5.0
    assert rep["algorithmic_ms"] == pytest.approx(20.0)
    assert rep["total_ms"] == pytest.approx(50.0)


def test_plot_latency(live_run, tmp_path):
    _, state = live_run
    path = plot_latency(state, str(tmp_path / "lat.png"))
    assert os.path.getsize(path) > 0


def test_params_change_midstream(whisper):
    state = init_live_state(dict(base_f0=100, jitter=0.0, stress=0.0, decl=0.0))
    y = np.zeros(len(whisper), dtype=np.float32)
    switch = int(1.5 * FS)
    for i in range(0, len(whisper) - HOP + 1, HOP):
        if i == switch:
            state["params"]["base_f0"] = 200
        out, state = process_hop_live(whisper[i:i + HOP], state)
        y[i:i + HOP] = out
    early = _f0_autocorr(y[int(0.5 * FS):int(0.62 * FS)])
    late = _f0_autocorr(y[int(2.0 * FS) + 800:int(2.0 * FS) + 800 + 2000])
    assert late > early * 1.5


def test_live_vs_offline_envelope(whisper, tmp_path):
    wav = write_fixture(str(tmp_path / "w.wav"))
    off, _ = convert(wav, str(tmp_path / "o.wav"))
    live = process_file_live(whisper)
    eo = frame_energy(frame_signal(off))
    el = frame_energy(frame_signal(live))
    el = el[1:]
    eo = eo[:len(el)]
    n = min(len(el), len(eo))
    mask = np.arange(n) * HOP / FS > 0.35
    c = np.corrcoef(np.log(eo[:n][mask] + 1e-6), np.log(el[:n][mask] + 1e-6))[0, 1]
    print(f"\nlive vs offline log-RMS envelope correlation {c:.3f}")
    assert c > 0.8


def test_imports_without_device():
    import app
    import run_live
    import src.stream
    assert callable(app.build_app) and callable(run_live.main) and callable(src.stream.run_stream)


def test_run_live_file_subprocess(tmp_path):
    wav = write_fixture(str(tmp_path / "in.wav"))
    out = tmp_path / "out.wav"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run_live.py"), "--file", wav, str(out)],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 1000


def test_list_devices_subprocess():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run_live.py"), "--list-devices"],
                       capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stderr


def test_live_zcr_threshold_adapts_to_offline_value_and_finds_consonants():
    from src.detect import classify_frames
    from src.framing import frame_signal, frame_energy
    from src.live import init_live_state, process_hop_live
    x, fs = pseudo_whisper()
    _, zcr_off, _ = classify_frames(frame_signal(x), frame_energy(frame_signal(x)), return_thresholds=True)
    st = init_live_state()
    labels = []
    for i in range(len(x) // 160):
        _, st = process_hop_live(x[i * 160:(i + 1) * 160], st)
        labels.append(st["last_label"])
    assert abs(st["zcr_live"] - zcr_off) < 0.3 * zcr_off
    assert labels.count(1) >= 20


def test_live_fixed_zcr_threshold_param_overrides():
    from src.live import init_live_state, process_hop_live
    x, _ = pseudo_whisper()
    st = init_live_state(dict(zcr_thr=0.9))
    labels = []
    for i in range(len(x) // 160):
        _, st = process_hop_live(x[i * 160:(i + 1) * 160], st)
        labels.append(st["last_label"])
    assert st["zcr_live"] == 0.12
    assert labels.count(1) == 0


def test_live_zcr_threshold_keeps_adapting_when_window_not_multiple_of_ten():
    from src.live import init_live_state, process_hop_live
    rng = np.random.default_rng(0)
    low = np.convolve(rng.standard_normal(16000 * 6), np.ones(8) / 8, "same") * 0.1
    high = np.diff(rng.standard_normal(16000 * 6 + 1)) * 0.1
    x = np.concatenate([np.zeros(4800), low, high]).astype(np.float32)
    st = init_live_state(dict(zcr_window_s=2.55))
    first = None
    for i in range(len(x) // 160):
        _, st = process_hop_live(x[i * 160:(i + 1) * 160], st)
        if i == (4800 + len(low)) // 160:
            first = st["zcr_live"]
    assert st["zcr_live"] > first + 0.2


def test_device_from_args_keeps_a_lone_in_or_out():
    from types import SimpleNamespace
    import run_live
    pick = lambda i, o: run_live.device_from_args(SimpleNamespace(in_idx=i, out_idx=o))
    assert pick(None, None) is None
    assert pick(1, 3) == (1, 3)
    assert pick(1, None) == (1, None)
    assert pick(None, 3) == (None, 3)


def test_stream_callback_survives_processing_error(monkeypatch):
    import src.stream as stream
    state = init_live_state()
    real = stream.process_hop_live

    def boom(block, st):
        raise RuntimeError("boom")

    monkeypatch.setattr(stream, "process_hop_live", boom)
    cb = stream.make_callback(state)
    indata = np.full((160, 1), 0.1, dtype=np.float32)
    outdata = np.ones((160, 1), dtype=np.float32)
    cb(indata, outdata, 160, None, None)
    assert not outdata.any()
    assert state["error_count"] == 1 and "boom" in state["first_error"]
    monkeypatch.setattr(stream, "process_hop_live", real)
    cb(indata, outdata, 160, None, None)
    assert state["error_count"] == 1 and state["hop_count"] == 1
