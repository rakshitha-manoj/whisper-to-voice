import argparse

from src.synth import convert, run_ablation


def main():
    p = argparse.ArgumentParser(description="Convert a whispered wav to voiced speech.")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--base-f0", type=float, default=120.0)
    p.add_argument("--decl", type=float, default=0.12)
    p.add_argument("--stress", type=float, default=0.25)
    p.add_argument("--aspiration", type=float, default=0.15)
    p.add_argument("--warp", type=float, default=None)
    p.add_argument("--volume", type=float, default=1.0)
    p.add_argument("--variants", action="store_true", help="also write all ablation variants into results/")
    args = p.parse_args()

    convert(args.input, args.output, volume=args.volume, base_f0=args.base_f0, decl=args.decl,
            stress=args.stress, aspiration=args.aspiration, warp=args.warp)
    print(f"Written: {args.output}")
    if args.variants:
        run_ablation(args.input, results_dir="results")


if __name__ == "__main__":
    main()
