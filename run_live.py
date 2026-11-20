import argparse
import sounddevice as sd
from src.stream import run_stream

def main():
    parser = argparse.ArgumentParser(description='Live duplex stream for Module A')
    parser.add_argument('--list-devices', action='store_true', help='List audio devices and exit')
    parser.add_argument('--in', dest='in_idx', type=int, help='Input device index')
    parser.add_argument('--out', dest='out_idx', type=int, help='Output device index')
    args = parser.parse_args()

    if args.list_devices:
        print('Available audio devices (index, name):')
        for i, d in enumerate(sd.query_devices()):
            print(f"{i}: {d['name']}")
        return

    device = None
    if args.in_idx is not None and args.out_idx is not None:
        device = (args.in_idx, args.out_idx)
        print(f"Using devices – input: {args.in_idx}, output: {args.out_idx}")
    else:
        print('Using default input/output devices. Use --list-devices and specify --in and --out for explicit selection.')

    run_stream(sample_rate=16000, blocksize=160, device=device)

if __name__ == '__main__':
    main()
