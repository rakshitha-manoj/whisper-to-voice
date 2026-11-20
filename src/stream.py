# src/stream.py
"""Real‑time duplex streaming utilities for Module A.
The core processing is a simple passthrough – teammates will later replace
``process_hop`` with a voiced‑speech synthesis pipeline.
"""

import numpy as np
import sounddevice as sd
import time
import os
from .detect import classify_frames
from .framing import frame_energy, frame_zcr
from .framing import load_audio, frame_signal

# Global mutable state for the live callback
_state = {
    "prev_hop": np.zeros(160, dtype=np.float32),
    "label_thr": None,  # (zcr_thr, silence_thr)
    "latencies": [],
    "frame_buf": np.zeros(320, dtype=np.float32),
    "calibrated": False,
}


def _calibrate_noise_floor(frames, hop=160, fs=16000):
    """Estimate noise floor from first 0.3 s of audio (early frames)."""
    rms = frame_energy(frames)
    samples_03s = int(0.3 * fs)
    frames_early = max(0, (samples_03s - frames.shape[1]) // hop) + 1
    frames_early = min(frames_early, len(rms))
    noise_floor = np.median(rms[:frames_early]) if frames_early > 0 else 0.0
    silence_thr = max(3 * noise_floor, 1e-4)
    # ZCR threshold from non‑silent frames (percentiles as in classify_frames)
    zcr = frame_zcr(frames)
    non_silent = rms >= silence_thr
    if np.any(non_silent):
        zcr_thr = np.percentile(zcr[non_silent], 70)
    else:
        zcr_thr = 0.5
    return zcr_thr, silence_thr


def process_hop(block: np.ndarray) -> np.ndarray:
    """Process a single 160‑sample hop.
    Currently a pass‑through that also updates latency statistics and prints
    a throttled label for demonstration.
    """
    start_time = time.time()
    # Assemble 320‑sample frame (previous hop + current)
    _state["frame_buf"][:160] = _state["prev_hop"]
    _state["frame_buf"][160:] = block
    frame = _state["frame_buf"]

    # Compute features
    rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
    signs = np.sign(frame)
    signs[signs == 0] = 1
    zcr = np.mean(np.abs(np.diff(signs)) > 0)

    # On first callback, calibrate thresholds using the initial frame(s)
    if not _state["calibrated"]:
        # Use a short buffer of early frames to estimate thresholds – here we simply
        # take the first frame as a proxy (the user will start with silence).
        _state["label_thr"] = _calibrate_noise_floor(np.expand_dims(frame, 0))
        _state["calibrated"] = True
        print(f"Calibrated thresholds – ZCR { _state['label_thr'][0]:.3f}, silence { _state['label_thr'][1]:.6f }")

    zcr_thr, silence_thr = _state["label_thr"]
    if rms < silence_thr:
        label = 0
    elif zcr > zcr_thr:
        label = 1
    else:
        label = 2

    # Throttled printing (max 5 prints per second)
    if int(time.time() * 5) != int(_state.get("last_print", -1) * 5):
        print(f"Live label: {label}")
        _state["last_print"] = time.time()

    # Record latency (ms)
    _state["latencies"].append((time.time() - start_time) * 1000.0)

    # Update previous hop for next call and return untouched block
    _state["prev_hop"] = block.copy()
    return block


def run_stream(sample_rate: int = 16000, blocksize: int = 160, device: tuple = None):
    """Start a duplex sounddevice stream.

    Parameters
    ----------
    sample_rate : int, default 16000
    blocksize : int, default 160 (hop size)
    device : tuple(int, int) or None
        (input_index, output_index). If None, the default devices are used.
    """
    if device is None:
        dev_desc = sd.query_devices()
        print("Available audio devices (index, name):")
        for i, d in enumerate(dev_desc):
            print(f"{i}: {d['name']}")
        print("Use --in and --out indices to select devices.")
    else:
        print(f"Using input device {device[0]}, output device {device[1]}")

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print('Stream status:', status)
        block = indata[:, 0].astype(np.float32)
        outdata[:, 0] = process_hop(block)

    stream = sd.Stream(samplerate=sample_rate,
                       blocksize=blocksize,
                       dtype='float32',
                       channels=1,
                       device=device,
                       callback=callback)
    stream.start()
    print(f"Duplex stream started – {sample_rate} Hz, blocksize {blocksize}")
    print("Press Ctrl‑C to stop.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Stopping stream…")
        stream.stop()
        stream.close()
        # After stopping, plot latency statistics
        if _state["latencies"]:
            import matplotlib.pyplot as plt
            lat = np.array(_state["latencies"])
            over_budget = lat > 10.0
            print(f"Mean hop time: {lat.mean():.2f} ms, max: {lat.max():.2f} ms, over 10 ms: {over_budget.sum()} of {len(lat)} hops")
            # Plot
            os.makedirs('results', exist_ok=True)
            plt.figure(figsize=(6, 3))
            plt.plot(lat, label='hop time')
            plt.axhline(10, color='r', linestyle='--', label='10 ms budget')
            plt.xlabel('Hop index')
            plt.ylabel('Time (ms)')
            plt.title('Realtime processing latency')
            plt.legend()
            plt.tight_layout()
            plt.savefig('results/latency.png')
            plt.close()
            print('Latency plot saved to results/latency.png')
        else:
            print('No latency data recorded.')

    return stream
