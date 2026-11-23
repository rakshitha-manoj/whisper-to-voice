"""Guided recording session: all transcript sentences, whispered and normal, for one speaker.

Usage (from anywhere, works on Windows and Mac):
    python tools/record_session.py --speaker 02
    python tools/record_session.py --shared           # shared_test.wav (Hafsa only)
See docs/RECORDING.md for the full instructions.
"""
import argparse
import os
import sys
import threading
import time
import zipfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluate import load_transcripts  # noqa: E402
from src.framing import frame_energy, frame_signal  # noqa: E402

FS = 16000
STYLES = ("whisper", "normal")
SHARED_IDS = ("01", "02")
DEFAULT_SECONDS = 5.0
SHARED_SECONDS = 12.0
LEAD_IN_HINT_S = 0.5

# Quality check thresholds. Speech level = 95th percentile of frame RMS (robust to short bursts).
CLIP_PEAK = 0.99          # a sample this close to full scale has almost certainly clipped
MIN_SPEECH_RMS = 0.002    # about -54 dBFS: catches a silent or wrong mic (or no Mac permission, which records zeros).
                          # Not higher: a clean headset whisper at default gain can sit around -45 to -50 dBFS, and
                          # poor signal to noise is caught by the lead-in check below.
LEAD_IN_S = 0.3           # the pipeline calibrates its silence threshold from the first 0.3 s
LEAD_IN_RATIO = 0.3       # lead-in RMS above 0.3 x speech RMS: noise or talking started too early
TAIL_S = 0.3              # last 0.3 s should be silent again
TAIL_RATIO = 0.5          # tail RMS above 0.5 x speech RMS: the sentence was probably cut off

WHISPER_HELP = "WHISPER: no voice at all, only breath. Normal speaking pace. Put your hand on your throat: no buzz."
NORMAL_HELP = "NORMAL: your usual speaking voice and pitch, like talking to a friend. Not a flat list voice."


def check_take(x, fs=FS):
    """Return a list of warning strings for a recorded take (empty list means it looks fine)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    warns = []
    if x.size < fs // 2:
        return ["Recording is far too short."]
    peak = float(np.max(np.abs(x)))
    if peak >= CLIP_PEAK:
        warns.append(f"Clipping (peak {peak:.2f}). Move the mic further away or speak softer, then redo.")
    x = x - np.mean(x)  # judge the signal without any DC offset (load_audio removes it too)
    e = frame_energy(frame_signal(x))
    speech = float(np.percentile(e, 95))
    if speech < MIN_SPEECH_RMS:
        warns.append(f"Too quiet (speech level {speech:.4f}, minimum {MIN_SPEECH_RMS}). Check that the script uses "
                     "your headset mic and that it is not muted, raise the microphone input volume in the system "
                     "sound settings, or move the mic closer.")
        return warns
    lead = float(np.sqrt(np.mean(x[: int(LEAD_IN_S * fs)] ** 2)))
    if lead > LEAD_IN_RATIO * speech:
        warns.append("Noise or speech in the first 0.3 s: you started too early, or the room noise is loud compared "
                     "to your voice. Stay silent for half a second after pressing Enter; if it keeps appearing, "
                     "make the room quieter or move the mic closer.")
    tail = float(np.sqrt(np.mean(x[-int(TAIL_S * fs):] ** 2)))
    if tail > TAIL_RATIO * speech:
        warns.append("Sound in the last 0.3 s: the sentence may be cut off. Start a little sooner or speak faster.")
    return warns


def speaker_id(s):
    """'2', '02' and 'spk02' all give '02'. Raises ValueError for anything else (ids 01 to 99)."""
    t = str(s).strip().lower()
    t = t[3:] if t.startswith("spk") else t
    if not t.isdigit() or not 1 <= int(t) <= 99:
        raise ValueError(f"speaker id must be a number from 01 to 99, for example 02 (got {s!r})")
    return f"{int(t):02d}"


def clip_path(data_dir, speaker, sid, style):
    return os.path.join(data_dir, f"spk{speaker}_s{sid}_{style}.wav")


def save_take(path, x):
    """Write 16 kHz mono PCM_16 to a temp file, then rename, so a crash never leaves a half file."""
    import soundfile
    tmp = path + ".part"
    try:
        soundfile.write(tmp, np.asarray(x, dtype=np.float32).reshape(-1), FS, subtype="PCM_16", format="WAV")
        os.replace(tmp, path)
    except BaseException:  # including Ctrl-C: never leave the .part behind
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def default_rec(seconds, on_start=None):
    """Record `seconds` of 16 kHz mono. on_start() is called once capture is running: opening the stream
    can take about half a second (measured 0.3 to 0.6 s with Windows MME), so cues must not start earlier."""
    import sounddevice
    x = sounddevice.rec(int(seconds * FS), samplerate=FS, channels=1, dtype="float32")
    if on_start is not None:
        on_start()
    sounddevice.wait()
    return x[:, 0]


def default_play(x):
    import sounddevice
    sounddevice.play(np.asarray(x, dtype=np.float32), FS)
    sounddevice.wait()


def record_one(path, title, text, help_line, seconds, rec, play, ask, sleep, out=print):
    """Record one file with the keep/redo loop. Returns 'kept', 'skipped' or 'quit'."""
    out("")
    out("=" * 60)
    out(title)
    out(f'    "{text}"')
    out(help_line)
    while True:
        a = ask("Press Enter to record (stay silent for half a second first), s = skip, q = quit: ").strip().lower()
        if a in ("s", "q"):
            return "skipped" if a == "s" else "quit"
        if a != "":
            continue
        for n in (3, 2, 1):
            out(f"  {n}...")
            sleep(1.0)
        out("  Recording... stay silent")
        timer = threading.Timer(LEAD_IN_HINT_S, lambda: out("  ...speak now"))
        try:  # rec(seconds, on_start) starts the cue once capture is really running
            x = np.asarray(rec(seconds, timer.start), dtype=np.float32).reshape(-1)
        finally:
            timer.cancel()
        out("  Done. Playing it back...")
        play(x)
        for w in check_take(x):
            out(f"  WARNING: {w}")
        while True:
            c = ask("[k]eep  [r]edo  [p]lay again  [s]kip  [q]uit: ").strip().lower()[:1]
            if c == "p":
                play(x)
            elif c in ("k", "r", "s", "q"):
                break
        if c == "k":
            save_take(path, x)
            out(f"  Saved {os.path.basename(path)}")
            return "kept"
        if c == "s":
            return "skipped"
        if c == "q":
            return "quit"
        # c == "r": record again


def _sid(i):
    if not str(i).strip().isdigit():
        raise ValueError(f"sentence id must be a number such as 07 (got {i!r})")
    return f"{int(i):02d}"


def select_ids(transcripts, only=None, start=None):
    ids = sorted(transcripts)
    if only:
        only = [_sid(i) for i in only]
        bad = [i for i in only if i not in transcripts]
        if bad:
            raise ValueError(f"unknown sentence id(s): {', '.join(bad)}")
        ids = [i for i in ids if i in only]
    if start:
        ids = [i for i in ids if i >= _sid(start)]
        if not ids:
            raise ValueError(f"no sentence from {_sid(start)} on (last id is {sorted(transcripts)[-1]})")
    return ids


def run_session(speaker, data_dir, transcripts, ids, styles=STYLES, redo=False, seconds=DEFAULT_SECONDS,
                rec=default_rec, play=default_play, ask=input, sleep=time.sleep, out=print):
    """Go through the selected sentences and styles. Returns True if the user quit early."""
    os.makedirs(data_dir, exist_ok=True)
    total = len(transcripts)
    for sid in ids:
        for style in styles:
            path = clip_path(data_dir, speaker, sid, style)
            if os.path.isfile(path) and not redo:
                out(f"Sentence {sid}/{total:02d}, {style.upper()}: already recorded, skipping (use --redo to record again)")
                continue
            title = f"Sentence {sid}/{total:02d}, {style.upper()}"
            try:
                r = record_one(path, title, transcripts[sid], WHISPER_HELP if style == "whisper" else NORMAL_HELP,
                               seconds, rec, play, ask, sleep, out)
            except EOFError:
                return True
            if r == "quit":
                return True
    return False


def run_shared(data_dir, transcripts, redo=False, seconds=SHARED_SECONDS, rec=default_rec, play=default_play,
               ask=input, sleep=time.sleep, out=print):
    """Record data/raw/shared_test.wav: whispered sentences 01 and 02 back to back."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "shared_test.wav")
    if os.path.isfile(path) and not redo:
        out(f"{path} already exists (use --redo to record it again)")
        return "skipped"
    text = "  ".join(f"{i}: {transcripts[i]}" for i in SHARED_IDS if i in transcripts)
    title = "Shared test clip: read both sentences, one after the other, with a short pause between them"
    try:
        r = record_one(path, title, text, WHISPER_HELP, seconds, rec, play, ask, sleep, out)
    except EOFError:
        return "quit"
    if r == "kept":
        out("")
        out("Commit it to main (see INSTRUCTIONS.md section 7):")
        out("    git switch main")
        out("    git pull origin main")
        out("    git add data/raw/shared_test.wav")
        out('    git commit -m "Add shared test clip"')
        out("    git push origin main")
    return r


def _compact(names):
    """['spk02_s01_whisper.wav', 'spk02_s01_normal.wav', ...] -> ['01 both', '07 normal', ...]"""
    by_id = {}
    for n in names:
        sid, style = n.split("_")[1][1:], n.split("_")[2][:-4]
        by_id.setdefault(sid, []).append(style)
    return [f"{sid} {'both' if len(st) == 2 else st[0]}" for sid, st in by_id.items()]


def summary(speaker, data_dir, transcripts, out=print):
    """Print how many of the expected files exist. Returns (n_present, missing_names)."""
    expected = [os.path.basename(clip_path(data_dir, speaker, sid, st)) for sid in sorted(transcripts) for st in STYLES]
    missing = [n for n in expected if not os.path.isfile(os.path.join(data_dir, n))]
    out("")
    out(f"Speaker {speaker}: {len(expected) - len(missing)} of {len(expected)} files recorded.")
    if missing:
        out("Missing (sentence id and style): " + ", ".join(_compact(missing)))
    return len(expected) - len(missing), missing


def make_zip(speaker, data_dir, zip_dir):
    """Zip this speaker's WAVs (only the spkNN_sMM_{whisper,normal}.wav files). Returns the zip path."""
    prefix = f"spk{speaker}_s"
    names = sorted(n for n in os.listdir(data_dir)
                   if n.startswith(prefix) and n.endswith(("_whisper.wav", "_normal.wav")))
    zpath = os.path.join(zip_dir, f"spk{speaker}_recordings.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            z.write(os.path.join(data_dir, n), arcname=n)
    return zpath, len(names)


def list_devices():
    import sounddevice
    print(sounddevice.query_devices())
    d = sounddevice.default.device
    print(f"\nDefault (input, output): {d[0]}, {d[1]}")
    print("Pick your headset from the first group (on Windows the devices marked MME; "
          "WASAPI and WDM-KS devices often refuse 16 kHz).")


def set_devices(device, out_device):
    import sounddevice
    if device is not None or out_device is not None:
        cur = sounddevice.default.device
        sounddevice.default.device = (cur[0] if device is None else device, cur[1] if out_device is None else out_device)
    ins, outs = sounddevice.default.device
    try:
        sounddevice.check_input_settings(device=ins, samplerate=FS, channels=1, dtype="float32")
        sounddevice.check_output_settings(device=outs, samplerate=FS, channels=1, dtype="float32")
    except Exception as e:
        raise SystemExit(f"The sound device cannot record or play 16 kHz mono ({e}).\n"
                         "Run with --list-devices and choose another --device / --output-device "
                         "(on Windows one marked MME), or set your headset as the default device.")
    print(f"Input device: {sounddevice.query_devices(ins)['name']}")
    print(f"Output device: {sounddevice.query_devices(outs)['name']}")


def build_parser():
    ap = argparse.ArgumentParser(description="Guided recording session (whispered and normal sentences).")
    ap.add_argument("--speaker", help="speaker id, e.g. 02 (01 Hafsa, 02 Akhila, 03 Raks)")
    ap.add_argument("--seconds", type=float, default=None, help=f"seconds per take (default {DEFAULT_SECONDS:g}, shared {SHARED_SECONDS:g})")
    ap.add_argument("--styles", choices=["whisper", "normal", "both"], default="both")
    ap.add_argument("--only", nargs="+", metavar="ID", help="record only these sentence ids, e.g. --only 07 12")
    ap.add_argument("--from", dest="start", metavar="ID", help="start at this sentence id")
    ap.add_argument("--redo", action="store_true", help="record again even if the file exists")
    ap.add_argument("--shared", action="store_true", help="record data/raw/shared_test.wav (Hafsa only)")
    ap.add_argument("--zip", action="store_true", help="create spkNN_recordings.zip in the repo root at the end")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", type=int, help="input device number (see --list-devices)")
    ap.add_argument("--output-device", type=int, help="playback device number")
    ap.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data", "raw"))
    ap.add_argument("--zip-dir", default=REPO_ROOT, help=argparse.SUPPRESS)
    return ap


def main(argv=None, rec=default_rec, play=default_play, ask=input, sleep=time.sleep, out=print):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.list_devices:
        list_devices()
        return 0
    if not args.shared and not args.speaker:
        ap.error("--speaker is required (for example --speaker 02)")
    transcripts = load_transcripts(os.path.join(args.data_dir, "transcripts.txt"))
    if not transcripts:
        ap.error(f"no sentences found in {os.path.join(args.data_dir, 'transcripts.txt')}")
    try:
        speaker = speaker_id(args.speaker) if args.speaker and not args.shared else None
    except ValueError as e:
        ap.error(str(e))
    if rec is default_rec:
        set_devices(args.device, args.output_device)
    try:
        if args.shared:
            run_shared(args.data_dir, transcripts, args.redo, args.seconds or SHARED_SECONDS, rec, play, ask, sleep, out)
            return 0
        try:
            ids = select_ids(transcripts, args.only, args.start)
        except ValueError as e:
            ap.error(str(e))
        styles = STYLES if args.styles == "both" else (args.styles,)
        quit_early = run_session(speaker, args.data_dir, transcripts, ids, styles, args.redo,
                                 args.seconds or DEFAULT_SECONDS, rec, play, ask, sleep, out)
    except KeyboardInterrupt:
        out("\nInterrupted.")
        if args.shared:
            return 0
        quit_early = True
    n, _ = summary(speaker, args.data_dir, transcripts, out)
    want_zip = args.zip
    if not want_zip and not quit_early and n:
        try:
            want_zip = ask(f"Create spk{speaker}_recordings.zip to send to Raks? [y/N]: ").strip().lower().startswith("y")
        except (EOFError, KeyboardInterrupt):
            want_zip = False
    if want_zip and n:
        zpath, count = make_zip(speaker, args.data_dir, args.zip_dir)
        out(f"Wrote {zpath} ({count} files). Send it as a document, not as a compressed photo/video.")
    return 0


if __name__ == "__main__":
    try:  # prompts and countdown appear at once even when stdout is not a console (Git Bash); odd device names never crash
        sys.stdout.reconfigure(line_buffering=True, errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
