"""Real-time duplex stream. The callback only calls process_hop_live; plotting happens after stop."""

import numpy as np
import sounddevice as sd

from .live import init_live_state, latency_report, plot_latency, process_hop_live

FS = 16000
HOP = 160


def list_devices():
    print("Available audio devices (index, name, host API, in/out channels):")
    apis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        print(f"{i}: {d['name']} [{apis[d['hostapi']]['name']}] in={d['max_input_channels']} out={d['max_output_channels']}")


def find_wasapi_devices():
    """Return (input_index, output_index) of the WASAPI default devices, or None."""
    try:
        for api in sd.query_hostapis():
            if "WASAPI" in api["name"]:
                i, o = api.get("default_input_device", -1), api.get("default_output_device", -1)
                if i >= 0 and o >= 0:
                    return int(i), int(o)
    except Exception as exc:
        print(f"WASAPI lookup failed: {exc}")
    return None


def _wasapi_settings():
    try:
        return sd.WasapiSettings(exclusive=False, auto_convert=True)
    except Exception as exc:
        print(f"WasapiSettings unavailable: {exc}")
        return None


def make_callback(state, passthrough=False):
    def callback(indata, outdata, frames, time_info, status):
        if status:
            state.setdefault("status_log", []).append(str(status))
        block = indata[:, 0]
        if passthrough:
            outdata[:, 0] = block
            return
        try:
            out, _ = process_hop_live(block, state)
        except Exception as exc:
            state["error_count"] = state.get("error_count", 0) + 1
            state.setdefault("first_error", repr(exc))
            out = np.zeros(len(block), dtype=np.float32)
        outdata[:, 0] = out
    return callback


def report_and_plot(state, stream_latency, path="results/latency.png"):
    rep = latency_report(state, stream_latency)
    print(f"Hops: {rep['hops']}  mean {rep['mean_ms']:.3f} ms  p95 {rep['p95_ms']:.3f} ms  max {rep['max_ms']:.3f} ms  "
          f"over 10 ms: {100 * rep['frac_over_budget']:.2f}%")
    dev = "n/a" if rep["device_ms"] is None else f"{rep['device_ms']:.1f} ms"
    print(f"Algorithmic delay {rep['algorithmic_ms']:.1f} ms, device latency {dev}, total {rep['total_ms']:.1f} ms")
    if state["timings"]:
        print(f"Latency plot saved to {plot_latency(state, path)}")
    if state.get("error_count"):
        print(f"Processing errors: {state['error_count']} hops output as silence (first: {state['first_error']})")
    if state.get("status_log"):
        print(f"Stream status events: {len(state['status_log'])} (e.g. {state['status_log'][0]})")
    return rep


def run_stream(sample_rate=FS, blocksize=HOP, device=None, params=None, passthrough=False, wasapi=False, state=None):
    """Start a duplex stream until Ctrl-C. device is (input, output) indices or None."""
    if state is None:
        state = init_live_state(params, sample_rate, blocksize)
    extra = None
    if wasapi:
        found = find_wasapi_devices()
        if found is None:
            print("No WASAPI devices found, using default host API.")
        else:
            if device is None:
                device = found
            else:
                device = tuple(f if d is None else d for d, f in zip(device, found))
            extra = _wasapi_settings()
    if device is None:
        print("Using default input/output devices. Use --list-devices, --in and --out to choose.")
    else:
        names = ["default" if d is None else d for d in device]
        print(f"Using input device {names[0]}, output device {names[1]}")
    kwargs = dict(samplerate=sample_rate, blocksize=blocksize, dtype="float32", channels=1,
                  device=device, callback=make_callback(state, passthrough))
    if extra is not None:
        kwargs["extra_settings"] = extra
    try:
        stream = sd.Stream(**kwargs)
    except Exception as exc:
        if extra is None:
            raise
        print(f"WASAPI settings rejected ({exc}); retrying without them.")
        kwargs.pop("extra_settings")
        stream = sd.Stream(**kwargs)
    stream.start()
    mode = "passthrough" if passthrough else "whisper to voice"
    print(f"Stream started ({mode}) at {sample_rate} Hz, blocksize {blocksize}. Stay quiet for the first "
          f"{state['params']['calib_s']} s (calibration). Press Ctrl-C to stop.")
    try:
        while True:
            sd.sleep(500)
    except KeyboardInterrupt:
        print("Stopping stream...")
    finally:
        latency = stream.latency
        stream.stop()
        stream.close()
    if not passthrough:
        report_and_plot(state, latency)
    return state
