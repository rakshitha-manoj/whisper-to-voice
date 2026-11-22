import numpy as np
import pytest

import app
from tests.fixtures import write_fixture


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return write_fixture(str(tmp_path_factory.mktemp("app") / "w.wav"))


def test_convert_clip_audio(clip):
    (fs, y), spec, lab = app.convert_clip(clip, "custom", 120, 0.12, 0.25, 0.15, 1.0, 1.0)
    assert fs == 16000 and len(y) > 1600 and np.all(np.isfinite(y))
    assert spec is not None and lab is not None


def test_convert_full_download(clip):
    import soundfile
    out, _, _, in_path, dl = app.convert_full(clip, "v4_warp", 120, 0.12, 0.25, 0.15, 0.9, 1.0)
    y, fs = soundfile.read(dl)
    assert in_path == clip and fs == 16000 and len(y) == len(out[1])


def test_convert_clip_requires_audio():
    with pytest.raises(ValueError):
        app.convert_clip(None, "custom", 120, 0.12, 0.25, 0.15, 1.0, 1.0)


def test_compare_variants(clip):
    res = app.compare_variants(clip, 120, 0.12, 0.25, 0.15, 1.0, 1.0)
    assert len(res) == 5
    assert [r[0] for r in res][1:] == ["v1_robot", "v2_contour", "v3_aspiration", "v4_warp"]
    assert all(len(r[1][1]) > 0 and np.all(np.isfinite(r[1][1])) for r in res)


def test_module_a_figure_and_thresholds(clip):
    af = app.analysis_features(clip)
    fig = app.module_a_figure(af, 3.0, None)
    assert len(fig.axes) == 4
    labels, zt, st = app.classify_for_analysis(af, 3.0, 0.2)
    assert zt == pytest.approx(0.2) and set(np.unique(labels)) <= {0, 1, 2}
    _, _, st_hi = app.classify_for_analysis(af, 8.0, None)
    assert st_hi > st


@pytest.mark.parametrize("which,warp", [("first", 1.0), ("last", 0.85), ("mid", 0.92)])
def test_lpc_frame_figure(clip, which, warp):
    af = app.analysis_features(clip)
    n = len(af["frames"])
    idx = {"first": 0, "last": n - 1, "mid": n // 2}[which]
    fig, text = app.lpc_frame_figure(af, idx, warp)
    d = app.lpc_frame_data(af, idx, warp)
    assert fig.axes and "prediction gain" in text
    assert np.all(np.isfinite(d["env_db"])) and np.all(np.isfinite(d["warped_db"]))
    if warp == 1.0:
        assert np.allclose(d["env_db"], d["warped_db"])


def test_analyse(clip):
    fa, fb, text, n = app.analyse(clip, 3.0, 0, 5, 0.9)
    assert n > 5 and fa is not None and fb is not None and text


def test_results_discovery_empty(tmp_path):
    assert app.discover_results(str(tmp_path)) == []
    assert app.discover_results(str(tmp_path / "missing")) == []
    assert app.results_figures(str(tmp_path)) == []
    assert "run_eval.py" in app.results_markdown(None)


def test_results_discovery_populated(tmp_path):
    sub = tmp_path / "dev_eval"
    (sub / "figures").mkdir(parents=True)
    (sub / "ablation.md").write_text("| a |\n|---|\n| 1 |\n")
    (sub / "figures" / "f.png").write_bytes(b"x")
    for v in ("v1_robot", "v2_contour"):
        (sub / f"clip1_whisper_{v}.wav").write_bytes(b"x")
    assert app.discover_results(str(tmp_path)) == [str(sub)]
    assert len(app.results_figures(str(sub))) == 1
    assert app.results_clips(str(sub)) == ["clip1_whisper"]
    assert set(app.clip_variant_paths(str(sub), "clip1_whisper")) == {"v1_robot", "v2_contour"}
    assert "| 1 |" in app.results_markdown(str(sub))


def test_build_app_constructs():
    assert app.build_app() is not None


def test_too_short_clip_raises_friendly(tmp_path):
    import soundfile
    p = str(tmp_path / "short.wav")
    soundfile.write(p, np.zeros(80, dtype=np.float32), 16000)
    for call in (lambda: app.convert_clip(p, "v4_warp", 120, 0.12, 0.25, 0.15, 0.9, 1.0),
                 lambda: app.compare_variants(p, 120, 0.12, 0.25, 0.15, 0.9, 1.0),
                 lambda: app.analyse(p, 3.0, 0, 0, 1.0)):
        with pytest.raises(ValueError, match="shorter than one"):
            call()


def test_f0_params_follow_variant():
    assert app.f0_params("custom", 0.2, 0.4) == (0.2, 0.4, 0.005)
    assert app.f0_params("v1_robot", 0.2, 0.4) == (0.0, 0.0, 0.0)
    assert app.f0_params("v3_aspiration", 0.2, 0.4) == (0.12, 0.25, 0.005)
