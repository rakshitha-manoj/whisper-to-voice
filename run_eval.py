import argparse
import os
import sys

from src import evaluate as ev


def main():
    ap = argparse.ArgumentParser(description="Ablation evaluation: WER and LSD per system variant.")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--results", default="results")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--model", default="base")
    ap.add_argument("--speakers", nargs="+", metavar="ID",
                    help='only these speakers, e.g. --speakers 02 or --speakers 01 03; add "shared" to include shared_test.wav')
    ap.add_argument("--dev", action="store_true", help="use synthetic clips from results/dev (tools/make_dev_clip.py)")
    ap.add_argument("--listening", action="store_true", help="create the blinded listening sheet")
    ap.add_argument("--aggregate-listening", metavar="SHEET")
    args = ap.parse_args()

    results = os.path.join(args.results, "dev_eval") if args.dev else args.results

    if args.aggregate_listening:
        key = os.path.join(os.path.dirname(args.aggregate_listening), "key.csv")
        out = ev.aggregate_listening(args.aggregate_listening, key, os.path.join(results, "listening_scores.csv"))
        for s in out.values():
            print(f"{s['system']}: clarity {s['clarity']:.2f} naturalness {s['naturalness']:.2f} (n={s['n']})")
        return 0

    raw_dir = os.path.join(args.results, "dev") if args.dev else args.raw_dir
    clips = ev.filter_speakers(ev.discover_clips(raw_dir), args.speakers)
    if not clips:
        if args.speakers:
            print(f"No clips for speakers {' '.join(args.speakers)} in {raw_dir}.")
        elif args.dev:
            print("No dev clips found. Run: python tools/make_dev_clip.py")
        else:
            print(f"No recordings found in {raw_dir}. Record them (16 kHz mono) with:")
            print("  python tools/record_session.py --speaker 01     # all 20 sentences, whispered and normal")
            print("  python tools/record_session.py --shared         # shared_test.wav (whispered sentences 01 and 02)")
            print("Then run: python run_eval.py")
        return 0

    suffix = " [SYNTHETIC DEV DATA]" if args.dev else ""
    if args.listening:
        out = ev.make_listening_sheet(clips, results)
        print(f"Listening sheet written to {out}")
        return 0

    rows, agg = ev.run_ablation_eval(clips, results, asr=not args.no_asr, model_name=args.model, dev=args.dev,
                                     transcripts_path=os.path.join(args.raw_dir, "transcripts.txt"))
    figs = ev.make_figures(clips, agg, results, suffix=suffix)
    print(f"{len(clips)} clips. Figures: " + ", ".join(figs))
    print()
    with open(os.path.join(results, "ablation.md"), encoding="utf-8") as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
