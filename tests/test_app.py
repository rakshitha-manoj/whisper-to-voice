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


def test_output_players_are_not_inputs_for_the_user():
    """The output player feeds Transcribe, which makes Gradio infer it as interactive unless pinned."""
    import gradio as gr
    demo = app.build_app()
    players = {b.label: b for b in demo.blocks.values() if isinstance(b, gr.Audio)}
    assert players["Voiced output (B)"].interactive is False
    assert players["Input (A)"].interactive is False


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


class FakeAsr:
    """Stands in for a Whisper model: returns fixed text and records what it was given."""
    def __init__(self, text="the cat sat"):
        self.text, self.calls = text, []

    def transcribe(self, x, **kw):
        self.calls.append(len(x))
        return {"text": f" {self.text} "}


@pytest.fixture
def fake_asr(monkeypatch):
    from src import evaluate
    loads = []

    def fake_load(name="base"):
        loads.append(name)
        return FakeAsr()
    monkeypatch.setattr(evaluate, "load_asr", fake_load)
    monkeypatch.setattr(app, "_ASR_CACHE", {})
    return loads


def test_transcribe_both_signals(clip, fake_asr):
    res = app.transcribe_clip(clip, "base", "")
    assert res["input"] == "the cat sat" and res["output"] == "the cat sat" and res["wer"] is None
    md = app.transcript_markdown(res)
    assert "the cat sat" in md and "WER" not in md


def test_transcribe_wer_against_reference(clip, fake_asr):
    res = app.transcribe_clip(clip, "base", "The cat sat down.")
    assert res["wer"] == {"input": (1, 4), "output": (1, 4)}
    assert "1/4 = 25.0%" in app.transcript_markdown(res)
    res = app.transcribe_clip(clip, "base", "the cat sat")
    assert res["wer"]["input"] == (0, 3)


def test_transcribe_uses_output_player_value(clip, fake_asr):
    fs_y = (16000, (np.ones(8000) * 1000).astype(np.int16))
    app.transcribe_clip(clip, "base", "", out_audio=fs_y, shown_input=clip)
    model = app.get_asr("base")
    assert model.calls[1] == 8000          # second call is the player signal, not a re-render
    app.transcribe_clip(clip, "base", "", out_audio=fs_y, shown_input="other.wav")
    assert model.calls[3] != 8000          # stale player: output re-rendered from current settings


def test_transcribe_requires_clip(fake_asr):
    with pytest.raises(ValueError, match="Upload or record"):
        app.transcribe_clip(None, "base", "")


def test_transcribe_whisper_missing(clip, monkeypatch):
    from src import evaluate

    def boom(name="base"):
        raise ImportError("no whisper")
    monkeypatch.setattr(evaluate, "load_asr", boom)
    monkeypatch.setattr(app, "_ASR_CACHE", {})
    with pytest.raises(ValueError, match="pip install openai-whisper"):
        app.transcribe_clip(clip, "base", "")


def test_asr_model_cache_reuses_object(fake_asr):
    a = app.get_asr("tiny")
    assert app.get_asr("tiny") is a and fake_asr == ["tiny"]
    assert app.get_asr("base") is not a
    with pytest.raises(ValueError, match="Unknown model"):
        app.get_asr("huge")


def _whisper_base_available():
    import importlib.util
    import os
    return (importlib.util.find_spec("whisper") is not None
            and os.path.isfile(os.path.expanduser("~/.cache/whisper/base.pt")))


@pytest.mark.skipif(not _whisper_base_available(), reason="openai-whisper or cached base model missing")
def test_transcribe_real_base_does_not_crash(clip):
    res = app.transcribe_clip(clip, "base", "hello world")
    assert isinstance(res["input"], str) and isinstance(res["output"], str)
    assert res["wer"]["input"][1] == 2
