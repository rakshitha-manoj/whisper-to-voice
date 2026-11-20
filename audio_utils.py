"""Top‑level convenience module.
Re‑exports the core utilities defined in the new ``src`` package so existing imports
(e.g. the legacy test script) continue to work.
"""

# Import symbols from the new layout
from src.framing import load_audio, frame_signal, frame_energy, frame_zcr
from src.detect import voicing_strength, classify_frames, plot_spectrogram

__all__ = [
    "load_audio",
    "frame_signal",
    "frame_energy",
    "frame_zcr",
    "voicing_strength",
    "classify_frames",
    "plot_spectrogram",
]
