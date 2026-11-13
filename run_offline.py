import sys
import soundfile
from src.framing import load_audio, frame_signal, frame_energy
from src.detect import classify_frames
from src.lpc import lpc_all
from src.pitch import build_f0, make_excitation
from src.synth import synthesize

x, fs = load_audio(sys.argv[1])
frames = frame_signal(x)
energy = frame_energy(frames)
labels = classify_frames(frames, energy)
A, G = lpc_all(frames, order=12)
f0 = build_f0(labels, energy, base_f0=120)
e = make_excitation(labels, f0)
y = synthesize(A, G, e, labels, energy)
soundfile.write(sys.argv[2], y, fs)
