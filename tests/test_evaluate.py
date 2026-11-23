import csv
import os

import numpy as np
import pytest
import soundfile

from src import evaluate as ev
from src.synth import VARIANTS
from tests.fixtures import pseudo_whisper, write_fixture


def test_load_transcripts_with_odd_line(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("01 The birch canoe.\n02 Glue the sheet.\ns01 Hey, how are you?", encoding="utf-8")
    with pytest.warns(UserWarning):
        tr = ev.load_transcripts(str(p))
    assert tr == {"01": "The birch canoe.", "02": "Glue the sheet."}


def test_load_real_transcripts():
    if not os.path.isfile("data/raw/transcripts.txt"):
        pytest.skip("no transcripts")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tr = ev.load_transcripts("data/raw/transcripts.txt")
    assert set(tr) == {f"{i:02d}" for i in range(1, 21)}
    assert all(tr[k].strip() for k in tr)


def test_normalise_text():
    assert ev.normalise_text("  Hello,   World!  It's a blue-green  day. ") == "hello world its a blue green day"


def test_wer_counts():
    assert ev.wer_counts("the cat sat", "The cat sat.") == (0, 3)
    assert ev.wer_counts("the cat sat", "") == (3, 3)
    assert ev.wer_counts("the cat sat", "the dog sat") == (1, 3)


def test_discover_clips(tmp_path):
    for name in ["shared_test.wav", "spk01_s03_whisper.wav", "spk01_s03_normal.wav",
                 "spk01_s04_whisper.wav", "junk.wav"]:
        (tmp_path / name).write_bytes(b"")
    clips = ev.discover_clips(str(tmp_path))
    by_stem = {ev.clip_stem(c): c for c in clips}
    assert set(by_stem) == {"shared_test", "spk01_s03_whisper", "spk01_s04_whisper"}
    assert by_stem["shared_test"]["sentence_ids"] == ["01", "02"]
    assert by_stem["spk01_s03_whisper"]["sentence_ids"] == ["03"]
    assert by_stem["spk01_s03_whisper"]["normal_path"].endswith("spk01_s03_normal.wav")
    assert by_stem["spk01_s04_whisper"]["normal_path"] is None
    assert ev.discover_clips(str(tmp_path / "nothing")) == []


def test_lsd_variant():
    x, _ = pseudo_whisper()
    assert ev.lsd_variant(x, x) == pytest.approx(0.0, abs=1e-9)
    y = np.random.default_rng(0).standard_normal(len(x)).astype(np.float32) * 0.05
    assert ev.lsd_variant(x, y) > 0.5


@pytest.fixture
def clip(tmp_path):
    p = write_fixture(str(tmp_path / "spk01_s01_whisper.wav"))
    return dict(path=p, speaker="spk01", sentence_ids=["01"], normal_path=None)


def test_run_ablation_eval_no_asr(clip, tmp_path):
    res = str(tmp_path / "res")
    rows, agg = ev.run_ablation_eval([clip], res, asr=False, transcripts={"01": "the birch canoe"})
    systems = [a["system"] for a in agg]
    assert systems == ["raw_whisper"] + list(VARIANTS) and len(systems) == 5
    assert agg[0]["lsd_mean"] == 0.0
    assert all(a["lsd_mean"] > 0 for a in agg[1:])
    for f in ["ablation_per_clip.csv", "ablation.csv", "ablation.md"]:
        assert os.path.isfile(os.path.join(res, f))
    for v in VARIANTS:
        assert os.path.isfile(os.path.join(res, f"spk01_s01_whisper_{v}.wav"))
    md = open(os.path.join(res, "ablation.md"), encoding="utf-8").read()
    assert "| System variant | WER | LSD (dB) | Listening clarity | Listening naturalness |" in md
    assert "v4_warp" in md


def test_run_ablation_eval_monkeypatched_asr(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "transcribe", lambda model, x: "the birch canoe")
    rows, agg = ev.run_ablation_eval([clip], str(tmp_path / "r"), asr=True, model=object(),
                                     transcripts={"01": "The birch canoe slid"})
    assert agg[0]["wer_mean"] == pytest.approx(0.25)
    assert agg[0]["wer_corpus"] == pytest.approx(0.25)


def test_listening_sheet_and_aggregate(clip, tmp_path):
    res = str(tmp_path / "res")
    d = ev.make_listening_sheet([clip], res, seed=1)
    key = list(csv.DictReader(open(os.path.join(d, "key.csv"), encoding="utf-8")))
    assert len(key) == 5
    assert {k["system"] for k in key} == {"raw_whisper"} | set(VARIANTS)
    for k in key:
        assert os.path.isfile(os.path.join(d, k["item"] + ".wav"))
        x, _ = soundfile.read(os.path.join(d, k["item"] + ".wav"))
        assert len(x) > 0
    sheet = list(csv.DictReader(open(os.path.join(d, "sheet.csv"), encoding="utf-8")))
    assert {r["listener"] for r in sheet} == {f"L{i}" for i in range(1, 6)}
    assert len(sheet) == 25
    assert os.path.isfile(os.path.join(d, "README.txt"))

    item_of = {k["system"]: k["item"] for k in key}
    filled = os.path.join(d, "filled.csv")
    with open(filled, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["listener", "item", "clarity_1to5", "naturalness_1to5"])
        w.writerow(["L1", item_of["v4_warp"], 4, 2])
        w.writerow(["L2", item_of["v4_warp"], 2, 4])
        w.writerow(["L1", item_of["raw_whisper"], 1, ""])
        w.writerow(["L3", item_of["v1_robot"], 5, 5])
    out = ev.aggregate_listening(filled, os.path.join(d, "key.csv"), os.path.join(res, "listening_scores.csv"))
    assert out["v4_warp"]["clarity"] == pytest.approx(3.0)
    assert out["v4_warp"]["naturalness"] == pytest.approx(3.0)
    assert out["v4_warp"]["n"] == 2
    assert out["v1_robot"]["clarity"] == 5
    assert "raw_whisper" not in out
    assert os.path.isfile(os.path.join(res, "listening_scores.csv"))
    ev.run_ablation_eval([clip], res, asr=False, transcripts={})
    assert "3.0" in open(os.path.join(res, "ablation.md"), encoding="utf-8").read()
