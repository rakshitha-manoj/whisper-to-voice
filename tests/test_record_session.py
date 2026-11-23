import os
import time
import zipfile

import numpy as np
import pytest
import soundfile

from tests.fixtures import FS, pseudo_whisper
from tools import record_session as rs

TRANSCRIPT = "\n".join(f"{i:02d} Sentence number {i}." for i in range(1, 21)) + "\n"


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    (d / "transcripts.txt").write_text(TRANSCRIPT, encoding="utf-8")
    return d


def clean_take(seconds=5.0):
    x, _ = pseudo_whisper()
    n = int(seconds * FS)
    out = np.zeros(n, dtype=np.float32)
    out[:len(x)] = x[:n]
    return out


class Fakes:
    """Scripted rec/play/ask. `answers` is consumed by ask(); Enter is the empty string."""

    def __init__(self, answers, take=None):
        self.answers = list(answers)
        self.take = take
        self.recs = 0
        self.plays = 0
        self.prompts = []
        self.out = []

    def rec(self, seconds, on_start=None):
        self.recs += 1
        return clean_take(seconds) if self.take is None else self.take

    def play(self, x):
        self.plays += 1

    def ask(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def run(self, argv, data_dir):
        argv = list(argv) + ["--data-dir", str(data_dir), "--zip-dir", str(data_dir)]
        return rs.main(argv, rec=self.rec, play=self.play, ask=self.ask, sleep=lambda s: None, out=self.out.append)


def test_records_all_names_for_speaker(data_dir):
    f = Fakes(["", "k"] * 40 + ["n"])
    f.run(["--speaker", "2"], data_dir)
    names = sorted(p.name for p in data_dir.glob("*.wav"))
    expected = sorted(f"spk02_s{i:02d}_{st}.wav" for i in range(1, 21) for st in ("whisper", "normal"))
    assert names == expected
    assert not list(data_dir.glob("*.part"))
    assert any("40 of 40" in line for line in f.out)
    assert rs.speaker_id("02") == rs.speaker_id(" spk2 ") == rs.speaker_id("SPK02") == "02"
    for bad in ["abc", "0", "100", "spk", "s02"]:
        with pytest.raises(ValueError):
            rs.speaker_id(bad)
    with pytest.raises(SystemExit):
        Fakes([]).run(["--speaker", "abc"], data_dir)


def test_keep_writes_16k_mono_right_length(data_dir, monkeypatch):
    f = Fakes(["", "k", "q"])
    f.run(["--speaker", "02", "--only", "07", "--seconds", "3"], data_dir)
    info = soundfile.info(str(data_dir / "spk02_s07_whisper.wav"))
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert info.frames == 3 * 16000

    real_write = soundfile.write

    def interrupted_write(*a, **k):
        real_write(*a, **k)  # the .part exists, then Ctrl-C arrives
        raise KeyboardInterrupt
    monkeypatch.setattr(soundfile, "write", interrupted_write)
    f = Fakes(["", "k"])
    f.run(["--speaker", "02", "--only", "08"], data_dir)  # Ctrl-C while saving
    assert any("Interrupted" in line for line in f.out)
    assert sorted(p.name for p in data_dir.glob("spk*")) == ["spk02_s07_whisper.wav"]  # no .part left


def test_redo_then_keep_writes_once(data_dir):
    f = Fakes(["", "r", "", "p", "k", "q"])
    f.run(["--speaker", "02", "--only", "07"], data_dir)
    assert f.recs == 2
    assert f.plays == 3  # after each take plus one "play again"
    assert [p.name for p in data_dir.glob("spk*.wav")] == ["spk02_s07_whisper.wav"]

    def slow_rec(seconds, on_start=None, started=True):  # the "speak now" cue counts from when capture starts
        if started:
            on_start()
        time.sleep(rs.LEAD_IN_HINT_S + 0.3)
        return clean_take(seconds)
    for started in (True, False):
        f = Fakes(["", "s"])
        f.rec = lambda s, cb=None: slow_rec(s, cb, started)
        f.run(["--speaker", "02", "--only", "08", "--styles", "whisper"], data_dir)
        assert any("speak now" in line for line in f.out) == started


def test_skip_writes_nothing(data_dir):
    f = Fakes(["", "s", "s", "q"])  # skip after take, skip at the prompt, then quit
    f.run(["--speaker", "02", "--only", "07"], data_dir)
    assert not list(data_dir.glob("spk*"))


def test_existing_skipped_unless_redo(data_dir):
    path = data_dir / "spk02_s07_whisper.wav"
    rs.save_take(str(path), np.zeros(1600, dtype=np.float32))
    f = Fakes(["", "k", "q"])
    f.run(["--speaker", "02", "--only", "07", "--styles", "whisper"], data_dir)
    assert f.recs == 0
    assert any("already recorded" in line for line in f.out)
    assert soundfile.info(str(path)).frames == 1600
    f = Fakes(["", "k", "n"])
    f.run(["--speaker", "02", "--only", "07", "--styles", "whisper", "--redo"], data_dir)
    assert f.recs == 1
    assert soundfile.info(str(path)).frames == 5 * 16000


def test_only_from_and_styles(data_dir):
    f = Fakes(["", "k"] * 5 + ["n"])
    f.run(["--speaker", "02", "--only", "07", "12", "--styles", "normal"], data_dir)
    assert sorted(p.name for p in data_dir.glob("spk*.wav")) == ["spk02_s07_normal.wav", "spk02_s12_normal.wav"]
    t = {f"{i:02d}": "x" for i in range(1, 21)}
    assert rs.select_ids(t, start="18") == ["18", "19", "20"]
    assert rs.select_ids(t, only=["3", "07"]) == ["03", "07"]
    with pytest.raises(ValueError):
        rs.select_ids(t, only=["99"])


def test_quality_checks():
    clean = clean_take()
    assert rs.check_take(clean) == []
    assert rs.check_take(clean + 0.02) == []  # a DC offset alone is not a problem
    assert rs.check_take(clean * 0.04) == []  # soft but clean whisper (about -47 dBFS speech level) is accepted
    assert any("Clipping" in w for w in rs.check_take(np.clip(clean * 20, -1, 1)))
    assert any("Too quiet" in w for w in rs.check_take(clean * 0.01))
    assert any("Too quiet" in w for w in rs.check_take(np.zeros(5 * FS, dtype=np.float32)))
    early = clean_take()
    early[: FS // 4] = clean_take()[int(0.4 * FS):int(0.4 * FS) + FS // 4]  # vowel in the first 0.25 s
    assert any("first 0.3 s" in w for w in rs.check_take(early))
    cut = clean_take()
    cut[-FS // 4:] = clean_take()[int(0.4 * FS):int(0.4 * FS) + FS // 4]
    assert any("cut off" in w for w in rs.check_take(cut))


def test_warnings_shown_but_keep_allowed(data_dir):
    f = Fakes(["", "k", "q"], take=np.zeros(5 * FS, dtype=np.float32))
    f.run(["--speaker", "02", "--only", "07"], data_dir)
    assert any("WARNING" in line and "Too quiet" in line for line in f.out)
    assert (data_dir / "spk02_s07_whisper.wav").is_file()


def test_quit_midway_no_partial_files_and_summary(data_dir):
    f = Fakes(["", "k", "", "q"])
    f.run(["--speaker", "02", "--only", "07", "08"], data_dir)
    assert [p.name for p in data_dir.iterdir() if p.name != "transcripts.txt"] == ["spk02_s07_whisper.wav"]
    text = "\n".join(f.out)
    assert "1 of 40" in text and "07 normal" in text and "08 both" in text


def test_zip_contains_only_that_speaker(data_dir):
    for name in ["spk02_s01_whisper.wav", "spk02_s01_normal.wav", "spk01_s01_whisper.wav", "shared_test.wav"]:
        rs.save_take(str(data_dir / name), np.zeros(1600, dtype=np.float32))
    f = Fakes([])
    f.run(["--speaker", "02", "--only", "01", "--zip"], data_dir)
    with zipfile.ZipFile(data_dir / "spk02_recordings.zip") as z:
        assert sorted(z.namelist()) == ["spk02_s01_normal.wav", "spk02_s01_whisper.wav"]


def test_shared_mode(data_dir):
    f = Fakes(["", "k"])
    f.run(["--shared"], data_dir)
    path = data_dir / "shared_test.wav"
    assert soundfile.info(str(path)).frames == 12 * 16000
    text = "\n".join(f.out)
    assert "Sentence number 1." in text and "Sentence number 2." in text and "git add data/raw/shared_test.wav" in text
    assert not list(data_dir.glob("spk*"))
    f = Fakes([""])
    f.rec = lambda seconds, on_start=None: (_ for _ in ()).throw(KeyboardInterrupt)
    f.run(["--shared", "--redo"], data_dir)  # Ctrl-C while recording the shared clip
    assert "Interrupted" in f.out[-1] and not any("Speaker" in line for line in f.out)
