import argparse
import sys

import numpy as np


def run_file(args):
    import soundfile
    from src.framing import load_audio
    from src.live import init_live_state, latency_report, plot_latency, process_file_live

    x, fs = load_audio(args.file[0])
    state = init_live_state(params_from_args(args), fs)
    y = process_file_live(x, fs=fs, state=state)
    soundfile.write(args.file[1], y, fs)
    rep = latency_report(state)
    print(f"Wrote {args.file[1]} ({len(y) / fs:.2f} s). Per-hop mean {rep['mean_ms']:.3f} ms, "
          f"p95 {rep['p95_ms']:.3f} ms, max {rep['max_ms']:.3f} ms, algorithmic delay {rep['algorithmic_ms']:.0f} ms")
    if args.play:
        import sounddevice as sd
        sd.play(y, fs)
        sd.wait()
    return 0


def params_from_args(args):
    p = dict(base_f0=args.base_f0, aspiration=args.aspiration, volume=args.volume)
    if args.warp is not None and args.warp < 1.0:
        p["warp"] = args.warp
    return p


def device_from_args(args):
    """(input, output) device pair for sounddevice; None on one side means that side's default."""
    if args.in_idx is None and args.out_idx is None:
        return None
    return (args.in_idx, args.out_idx)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live whisper to voice conversion")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--in", dest="in_idx", type=int, help="Input device index")
    parser.add_argument("--out", dest="out_idx", type=int, help="Output device index")
    parser.add_argument("--passthrough", action="store_true", help="Mic straight to output (A7 check)")
    parser.add_argument("--wasapi", action="store_true", help="Use WASAPI devices/settings (Windows)")
    parser.add_argument("--base-f0", type=float, default=120.0)
    parser.add_argument("--aspiration", type=float, default=0.15)
    parser.add_argument("--warp", type=float, default=None, help="Pole warp factor, e.g. 0.92 (1.0 = off)")
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--file", nargs=2, metavar=("IN.wav", "OUT.wav"), help="Run a file through the live path")
    parser.add_argument("--play", action="store_true", help="Play the --file result")
    args = parser.parse_args(argv)

    if args.list_devices:
        from src.stream import list_devices
        list_devices()
        return 0
    if args.file:
        return run_file(args)

    from src.stream import run_stream
    run_stream(device=device_from_args(args), params=params_from_args(args), passthrough=args.passthrough, wasapi=args.wasapi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
