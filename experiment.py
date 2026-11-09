"""Run CoGa on the datasets of the paper and print the results; --save writes them to results/.

    python experiment.py                              # five datasets, settings of the paper
    python experiment.py --datasets MBA805 IOPS       # a subset
    python experiment.py --no-split                   # a variant
    python experiment.py --sweep --datasets MBA805    # target-SNR range and alpha sweeps of the sensitivity figure
    python experiment.py --save detection             # also write results/detection.json
"""
import argparse
import json
import time
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from data import DATASETS, load, raw_size
from src import detect
ROOT = Path(__file__).resolve().parent

# Terms of the path score, and the target-SNR ranges and alphas swept for the sensitivity figure.
TERMS = {"full": ("node", "edge"), "node": ("node",), "edge": ("edge",)}
SWEEP_GAMMA_MAX = (20, 25, 30, 35, 40, 45, 50)
SWEEP_ALPHAS = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


def run(name, gamma_min=25, gamma_max=35, split=True, alpha=1.0, terms="full"):
    """CoGa on one dataset; returns the numbers of one row of the result file."""
    values, labels = load(name)
    start = time.perf_counter()
    result = detect(values, gamma_min=gamma_min, gamma_max=gamma_max, residual_split=split, alpha=alpha, terms=TERMS[terms])
    runtime = time.perf_counter() - start

    n = min(len(result.scores), len(labels))
    base, graph = result.base, result.graph
    return {
        "n": len(values),
        "gamma": int(base.gamma),
        "epsilon": float(base.epsilon),
        "residual_split": split,
        "segments": len(base),
        "n_over_M": len(values) / len(base),
        "nodes": graph.graph_.number_of_nodes(),
        "edges": graph.graph_.number_of_edges(),
        "query_length": int(result.query_length),
        "roc": 100 * float(roc_auc_score(labels[:n], result.scores[:n])),
        "pr": 100 * float(average_precision_score(labels[:n], result.scores[:n])),
        "footprint_kb": base.footprint() / 1024,
        "raw_kb": raw_size(values) / 1024,
        "runtime_s": runtime,
    }


def sweep(name, split=True, terms="full"):
    """The sensitivity figure: shift the target-SNR range, then vary alpha; prints one line per setting."""
    rows = {"gamma_range": [], "alpha": []}
    print(f"{name}: target-SNR range")
    for gamma_max in SWEEP_GAMMA_MAX:
        r = run(name, gamma_max - 10, gamma_max, split=split, terms=terms)
        rows["gamma_range"].append({"gamma_min": gamma_max - 10, "gamma_max": gamma_max, **r})
        print(f"  [{gamma_max - 10:2d}, {gamma_max:2d}] gamma={r['gamma']:2d} segments={r['segments']:7d} "
              f"ROC={r['roc']:5.1f} PR={r['pr']:5.1f} footprint={r['footprint_kb']:7.2f} KB")
    print(f"{name}: alpha")
    for alpha in SWEEP_ALPHAS:
        r = run(name, split=split, alpha=alpha, terms=terms)
        rows["alpha"].append({"alpha": alpha, **r})
        print(f"  alpha={alpha:5.2f} ROC={r['roc']:5.1f} PR={r['pr']:5.1f}")
    return rows


def show(results):
    """Print one line per dataset."""
    print(f"{'dataset':9s} {'gamma':>6s} {'segments':>9s} {'n/M':>6s} {'ROC':>6s} {'PR':>6s} "
          f"{'footprint (KB)':>15s} {'raw (KB)':>9s} {'time (s)':>9s}")
    for name, row in results.items():
        print(f"{name:9s} {row['gamma']:6d} {row['segments']:9d} {row['n_over_M']:6.1f} {row['roc']:6.1f} {row['pr']:6.1f} "
              f"{row['footprint_kb']:15.2f} {row['raw_kb']:9.2f} {row['runtime_s']:9.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--gamma-min", type=int, default=25, help="lower bound of the entropy-guided target SNR in dB")
    parser.add_argument("--gamma-max", type=int, default=35, help="upper bound; equal bounds give a fixed target")
    parser.add_argument("--no-split", action="store_true", help="disable the residual split of the encoder")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing constant")
    parser.add_argument("--terms", choices=list(TERMS), default="full", help="terms of the path score")
    parser.add_argument("--sweep", action="store_true",
                        help="run the target-SNR range and alpha sweeps of the sensitivity figure instead of one setting")
    parser.add_argument("--save", metavar="NAME", help="also write the results to results/NAME.json")
    args = parser.parse_args()

    if args.sweep:
        results = {d: sweep(d, split=not args.no_split, terms=args.terms) for d in args.datasets}
        settings = {"no_split": args.no_split, "terms": args.terms}
    else:
        results = {d: run(d, args.gamma_min, args.gamma_max, not args.no_split, args.alpha, args.terms) for d in args.datasets}
        settings = {"gamma_min": args.gamma_min, "gamma_max": args.gamma_max, "no_split": args.no_split,
                    "alpha": args.alpha, "terms": args.terms}
        show(results)

    if args.save:   # datasets that were not run keep their stored values
        out = ROOT / "results" / f"{args.save}.json"
        out.parent.mkdir(exist_ok=True)
        kept = json.loads(out.read_text())["datasets"] if out.exists() else {}
        out.write_text(json.dumps({"settings": settings, "datasets": {**kept, **results}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
