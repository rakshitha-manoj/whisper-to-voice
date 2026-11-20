import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import scipy.signal
import soundfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.framing import load_audio
from src.lpc import lpc_frame
from src.evaluate import load_transcripts

FS = 16000
FRAME = 320
HOP = 160
LEAD = 0.5
NOISE_FLOOR = 5e-4


def sapi_speak(text, wav_path):
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.SetOutputToWaveFile($env:W2V_WAV); "
        "$s.Speak($env:W2V_TEXT); $s.Dispose()"
    )
    env = dict(os.environ, W2V_WAV=wav_path, W2V_TEXT=text)
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", script], env=env,
                           capture_output=True, text=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        sys.exit(f"Windows SAPI TTS unavailable (PowerShell failed: {exc}).")
    if r.returncode != 0 or not os.path.isfile(wav_path):
        sys.exit(f"Windows SAPI TTS unavailable or failed: {r.stderr.strip()}")


def whisperize(x, seed=0):
    rng = np.random.default_rng(seed)
    n = (len(x) - FRAME) // HOP + 1
    y = np.zeros(len(x))
    zi = np.zeros(12)
    for i in range(n):
        a, g = lpc_frame(x[i * HOP:i * HOP + FRAME], 12)
        seg, zi = scipy.signal.lfilter([g], a, rng.standard_normal(HOP), zi=zi)
        y[i * HOP:(i + 1) * HOP] = seg
    y = scipy.signal.lfilter([1.0], [1.0, -0.97], y)
    for i in range(n):
        t = np.sqrt(np.mean(x[i * HOP:(i + 1) * HOP] ** 2))
        r = np.sqrt(np.mean(y[i * HOP:(i + 1) * HOP] ** 2))
        if r > 1e-12:
            y[i * HOP:(i + 1) * HOP] *= t / r
    return y


def main():
    ap = argparse.ArgumentParser(description="Make SYNTHETIC dev clips (SAPI TTS + LPC whisperisation).")
    ap.add_argument("--ids", nargs="+", default=["01", "02", "03"])
    ap.add_argument("--out", default="results/dev")
    ap.add_argument("--transcripts", default="data/raw/transcripts.txt")
    args = ap.parse_args()
    tr = load_transcripts(args.transcripts)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(1)
    for sid in args.ids:
        if sid not in tr:
            print(f"skip {sid}: not in transcripts")
            continue
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "tts.wav")
            sapi_speak(tr[sid], raw)
            x, _ = load_audio(raw)
        x = x.astype(np.float64)
        x = 0.3 * x / (np.max(np.abs(x)) + 1e-9)
        pad = np.zeros(int(LEAD * FS))
        x = np.concatenate([pad, x, pad])
        w = whisperize(x, seed=int(sid))
        floor = NOISE_FLOOR * rng.standard_normal(len(x))
        soundfile.write(os.path.join(args.out, f"dev_s{sid}_normal.wav"), (x + floor).astype(np.float32), FS)
        soundfile.write(os.path.join(args.out, f"dev_s{sid}_whisper.wav"), (w + floor).astype(np.float32), FS)
        print(f"dev_s{sid}: {len(x) / FS:.1f} s (SYNTHETIC)")


if __name__ == "__main__":
    main()
