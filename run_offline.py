import sys
import soundfile
from src.framing import load_audio, frame_signal, frame_energy
from src.detect import classify_frames
from src.lpc import lpc_all
from src.pitch import build_f0, make_excitation
from src.synth import synthesize, output_stage, run_ablation, evaluate_wer

if len(sys.argv) < 3:
    print("Usage: python run_offline.py <input.wav> <output.wav> [--ablation] [--wer]")
    sys.exit(1)

path_in = sys.argv[1]
path_out = sys.argv[2]
do_ablation = "--ablation" in sys.argv
do_wer = "--wer" in sys.argv

x, fs = load_audio(path_in)
frames = frame_signal(x)
energy = frame_energy(frames)
labels = classify_frames(frames, energy)
A, G = lpc_all(frames, order=12)
f0 = build_f0(labels, energy, base_f0=120)
e = make_excitation(labels, f0)
y_raw = synthesize(A, G, e, labels, energy)
y_out = output_stage(y_raw, energy)
soundfile.write(path_out, y_out, fs)
print(f"Written: {path_out}")

if do_ablation:
    run_ablation(path_in, results_dir="results")

if do_wer:
    evaluate_wer(path_in, results_dir="results")
