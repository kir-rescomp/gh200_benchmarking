"""Aggregate per-run JSON results from both languages into a single
comparison table.

Each harness invocation writes one JSON file (one arm, one language, one
repeat). This script globs a results directory, groups by
(language, arm, model, precision, batch_size), computes the median across
repeats of the per-run medians, and prints Python-vs-C++ side by side with
the relative difference.

Reporting the cross-run median of per-run medians (rather than pooling all
raw samples) keeps each run weighted equally and is robust to a single
straggling run. Variability across runs is reported as the spread of the
per-run medians so the reader can see whether a difference exceeds run noise.
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def run_metric(r):
    """Extract the single headline metric for an arm, in images/sec where
    meaningful and ms for latency."""
    arm = r["arm"]
    if arm == "train_throughput":
        return ("images_per_sec", r["step_ms"]["images_per_sec_median"])
    if arm == "inference_throughput":
        return ("images_per_sec", r["batch_ms"]["images_per_sec_median"])
    if arm == "inference_latency":
        return ("latency_ms_p50", r["latency_ms_percentiles"]["50"])
    raise ValueError(arm)


def key(r):
    return (r["arm"], r["model"], r["precision"], r.get("batch_size", "na"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.results_dir, "*.json"))
    if not files:
        print(f"no result files in {args.results_dir}")
        return

    # group[(arm,model,precision,bs)][language] -> list of per-run values
    group = defaultdict(lambda: defaultdict(list))
    metric_name = {}
    for path in files:
        with open(path) as f:
            r = json.load(f)
        name, val = run_metric(r)
        group[key(r)][r["language"]].append(val)
        metric_name[key(r)] = name

    rows = []
    for k in sorted(group):
        arm, model, prec, bs = k
        langs = group[k]
        py = langs.get("python", [])
        cpp = langs.get("cpp", [])
        row = {
            "arm": arm, "model": model, "precision": prec, "batch_size": bs,
            "metric": metric_name[k],
            "python_median": statistics.median(py) if py else None,
            "python_runs": len(py),
            "python_spread": (max(py) - min(py)) if len(py) > 1 else 0.0,
            "cpp_median": statistics.median(cpp) if cpp else None,
            "cpp_runs": len(cpp),
            "cpp_spread": (max(cpp) - min(cpp)) if len(cpp) > 1 else 0.0,
        }
        if row["python_median"] and row["cpp_median"]:
            higher_is_better = row["metric"].startswith("images_per_sec")
            pm, cm = row["python_median"], row["cpp_median"]
            if higher_is_better:
                row["cpp_vs_python_pct"] = (cm - pm) / pm * 100.0
            else:
                # latency: lower is better; positive % = C++ faster
                row["cpp_vs_python_pct"] = (pm - cm) / pm * 100.0
        rows.append(row)

    # pretty print
    print(f"{'arm':22s} {'model':9s} {'prec':9s} {'bs':>5s} "
          f"{'metric':18s} {'python':>12s} {'cpp':>12s} {'cpp_vs_py%':>10s}")
    print("-" * 110)
    for r in rows:
        pm = f"{r['python_median']:.3f}" if r['python_median'] else "-"
        cm = f"{r['cpp_median']:.3f}" if r['cpp_median'] else "-"
        pct = (f"{r['cpp_vs_python_pct']:+.1f}"
               if r.get('cpp_vs_python_pct') is not None else "-")
        print(f"{r['arm']:22s} {r['model']:9s} {r['precision']:9s} "
              f"{str(r['batch_size']):>5s} {r['metric']:18s} "
              f"{pm:>12s} {cm:>12s} {pct:>10s}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
