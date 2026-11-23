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


# ---- graphs: Module B poles and formants, Module C source and loudness, variant comparison, live path ----

def _texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts]


@pytest.mark.parametrize("warp", [1.0, 0.9])
@pytest.mark.parametrize("which", ["first", "last"])
def test_pole_figure(clip, which, warp):
    af = app.analysis_features(clip)
    idx = 0 if which == "first" else len(af["frames"]) - 1
    fig = app.pole_figure(af, idx, warp)
    ax = fig.axes[0]
    assert "kHz" in " ".join(_texts(fig))
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert ("warped poles (0.90)" in labels) == (warp < 1.0)
    assert "LPC poles" in labels


def test_pole_figure_rejects_empty_clip():
    with pytest.raises(ValueError):
        app.pole_figure(dict(frames=np.zeros((0, 320))), 0, 1.0)


def test_pole_frequencies_are_plausible(clip):
    feats = app.get_features(clip)
    tracks = app.formant_tracks(feats["A"], feats["labels"])
    assert len(tracks) == len(feats["labels"])
    assert sum(len(t) > 0 for t in tracks) > 20
    for t, lab in zip(tracks, feats["labels"]):
        assert len(t) <= 3
        assert np.all((t > 90) & (t < 8000))
        assert np.all(np.diff(t) >= 0)
        if lab == 0:
            assert len(t) == 0
    warped = app.formant_tracks(feats["A"], feats["labels"], 0.9)
    assert all(len(t) <= 3 and np.all((t > 90) & (t < 8000)) for t in warped)


def test_pole_frequencies_known_resonance():
    from tests.fixtures import formant_filter
    a = formant_filter([(700, 130), (1220, 70), (2600, 160)])
    f = app.pole_frequencies(a)
    assert np.allclose(f, [700, 1220, 2600], atol=5)
    assert len(app.pole_frequencies(np.array([1.0, 0, 0, 0]))) == 0


@pytest.mark.parametrize("warp", [1.0, 0.9])
def test_formant_figure(clip, warp):
    fig = app.module_b_pole_figures(clip, 30, warp)[1]
    labels = [ln.get_label() for ln in fig.axes[0].get_lines()]
    assert "original poles" in labels and (f"warped ({warp:.2f})" in labels) == (warp < 1.0)


def test_excitation_figure_voiced(clip):
    feats = app.get_features(clip)
    s = app.render_settings("custom", 120, 0.12, 0.25, 0.15, 1.0)
    fig = app.excitation_figure(feats, s)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        x = ax.get_lines()[0].get_xdata()
        assert 60 <= x[-1] - x[0] + 0.0625 <= 85  # about 70 ms in ms


def test_excitation_figure_without_voiced_frames():
    feats = dict(labels=np.ones(50, dtype=int), energy=np.ones(50))
    fig = app.excitation_figure(feats, app.render_settings("custom", 120, 0.12, 0.25, 0.15, 1.0))
    assert any("No voiced" in t for t in _texts(fig))


def test_source_loudness_on_clip_without_voiced_frames(tmp_path):
    import soundfile
    p = str(tmp_path / "quiet.wav")
    soundfile.write(p, (0.001 * np.random.default_rng(1).standard_normal(8000)).astype(np.float32), 16000)
    y, _ = app.render(p)
    figs = app.source_loudness_figures(p, y, "custom", 120, 0.12, 0.25, 0.15, 1.0)
    assert len(figs) == 3
    # every frame is silent, so LSD has no frames: a message, not five 0.00 dB bars that read as a perfect match
    lsd_fig = app.comparison_figures(p, app.compare_variants(p, 120, 0.12, 0.25, 0.15, 1.0, 1.0))[1]
    assert any("not defined" in t for t in _texts(lsd_fig)) and not lsd_fig.axes[0].patches


def test_f0_components_full_contour_matches_build_f0(clip):
    from src.pitch import build_f0
    feats = app.get_features(clip)
    c = app.f0_components(feats, 130, 0.2, 0.3)
    ref = build_f0(feats["labels"], feats["energy"], base_f0=130, decl=0.2, stress=0.3, jitter=0.0)
    assert np.array_equal(c["full"], ref)
    voiced = feats["labels"] == 2
    assert np.all(c["base"][voiced] == 130) and np.all(c["base"][~voiced] == 0)
    assert np.all(c["declination"][voiced] <= 130 + 1e-3)
    fig = app.f0_breakdown_figure(feats, app.render_settings("custom", 130, 0.2, 0.3, 0.15, 1.0))
    assert len(fig.axes[0].get_lines()) == 3


def test_render_settings_follow_variant():
    assert app.render_settings("custom", 100, 0.2, 0.4, 0.1, 0.9) == dict(
        base_f0=100, decl=0.2, stress=0.4, jitter=0.005, aspiration=0.1, warp=0.9)
    s = app.render_settings("v1_robot", 100, 0.2, 0.4, 0.1, 0.9)
    assert (s["decl"], s["stress"], s["jitter"], s["aspiration"], s["warp"]) == (0.0, 0.0, 0.0, 0.0, None)
    assert app.render_settings("v4_warp", 100, 0.2, 0.4, 0.1, 1.0)["warp"] == 0.92


def test_loudness_figure_output_follows_input(clip):
    from src.framing import load_audio
    x, _ = load_audio(clip)
    y, _ = app.render(clip)
    fig = app.loudness_figure(x, y)
    assert len(fig.axes) == 2
    ex, ey = app.rms_db(x), app.rms_db(y)
    loud = ex > ex.max() - 30
    assert np.abs(ex[loud] - ey[:len(ex)][loud]).mean() < 3.0


def test_source_loudness_ui_needs_convert(clip):
    with pytest.raises(ValueError):
        app.source_loudness_ui(clip, "custom", 120, 0.12, 0.25, 0.15, 1.0, 1.0, None)
    out, *_ = app.convert_clip(clip, "custom", 120, 0.12, 0.25, 0.15, 1.0, 0.5)
    # what Gradio hands back from a numpy output player: peak normalised int16
    pcm = (out[1] / np.abs(out[1]).max() * 32767).astype(np.int16)
    figs = app.source_loudness_ui(clip, "custom", 120, 0.12, 0.25, 0.15, 1.0, 0.5, (16000, pcm))
    assert len(figs) == 3
    # the loudness plot shows the rendered level (volume 0.5 included), not the normalised player level
    out_rms = [ln for ln in figs[2].axes[1].get_lines() if ln.get_label() == "output RMS"][0].get_ydata()
    assert np.allclose(out_rms, app.rms_db(out[1]), atol=1e-4)


@pytest.fixture(scope="module")
def cmp_results(clip):
    return app.compare_variants(clip, 120, 0.12, 0.25, 0.15, 0.92, 1.0)


def test_comparison_grid_lsd_ltas(clip, cmp_results):
    from src.framing import load_audio
    x, _ = load_audio(clip)
    sigs = app.comparison_signals(x, cmp_results)
    assert [n for n, _ in sigs] == ["input", "custom (sliders)", "v1_robot", "v2_contour", "v3_aspiration", "v4_warp"]
    assert len({len(s) for _, s in sigs}) == 1
    grid = app.spectrogram_grid_figure(sigs)
    clims = {tuple(ax.images[0].get_clim()) for ax in grid.axes if ax.images}
    assert len(clims) == 1 and sum(1 for ax in grid.axes if ax.images) == 6
    lsd = app.lsd_values(sigs[0][1], sigs[1:])
    assert list(lsd) == [n for n, _ in sigs[1:]] and all(np.isfinite(v) and v > 0 for v in lsd.values())
    assert len(app.lsd_figure(lsd).axes[0].patches) == 5
    assert len(app.ltas_figure(sigs).axes[0].get_lines()) == 6
    assert len(app.comparison_figures(clip, cmp_results)) == 3


def test_live_run_and_summary(clip):
    r = app.live_run(clip, "custom", 120, 0.12, 0.25, 0.15, 0.92, 1.0)
    n_hops = int(np.ceil(len(r["x"]) / 160))
    assert len(r["timings"]) == n_hops == r["report"]["hops"]
    assert len(r["y_live"]) == len(r["x"]) and np.all(np.isfinite(r["y_live"]))
    for k in ("mean_ms", "p95_ms", "max_ms", "frac_over_budget", "algorithmic_ms"):
        assert k in r["report"]
    out, lat, env, md = app.live_ui(clip, "custom", 120, 0.12, 0.25, 0.15, 0.92, 1.0)
    assert out[0] == 16000 and "Algorithmic delay" in md and "not measured" in md
    assert lat.axes[0].lines[1].get_ydata()[0] == 10.0 and len(env.axes[0].get_lines()) == 2


def test_live_run_requires_clip():
    with pytest.raises(ValueError):
        app.live_run(None, "custom", 120, 0.12, 0.25, 0.15, 1.0, 1.0)


def test_live_tab_player_not_interactive():
    import gradio as gr
    demo = app.build_app()
    players = {b.label: b for b in demo.blocks.values() if isinstance(b, gr.Audio)}
    assert players["Live path output"].interactive is False
