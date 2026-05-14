#!/usr/bin/env python3
"""
Merge per-job lm-eval result files into a single JSON + a readable TXT table.

Reads:   logs/eval_<run_name>_<jobid>.json   (written by slurm_eval_lm_harness.sh)
Writes:  logs/eval_all_results.json          (run -> task -> metrics)
         logs/eval_all_results.txt           (one row per run, primary metric per task)

Usage:
    python scripts/aggregate_eval_results.py
    python scripts/aggregate_eval_results.py --logs-dir logs --out-prefix logs/eval_all_results
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

# eval_<run_name>_<jobid>.json — jobid is purely numeric, run name may contain underscores
FNAME_RE = re.compile(r"^eval_(?P<run>.+)_(?P<jobid>\d+)\.json$")

# Preferred headline metric per task (falls back to first numeric metric if missing).
PRIMARY_METRIC = {
    "hellaswag": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
    "openbookqa": "acc_norm,none",
}


def pick_metric(task: str, metrics: dict):
    preferred = PRIMARY_METRIC.get(task)
    if preferred and preferred in metrics and isinstance(metrics[preferred], (int, float)):
        return preferred, metrics[preferred]
    for k, v in metrics.items():
        if k.endswith("_stderr,none"):
            continue
        if isinstance(v, (int, float)):
            return k, v
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--out-prefix", default="logs/eval_all_results")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.logs_dir, "eval_*.json")))
    # Keep latest jobid per run name.
    per_run_file = {}
    for f in files:
        m = FNAME_RE.match(os.path.basename(f))
        if not m:
            continue
        run = m.group("run")
        jobid = int(m.group("jobid"))
        if run not in per_run_file or jobid > per_run_file[run][0]:
            per_run_file[run] = (jobid, f)

    if not per_run_file:
        print(f"No eval_*_<jobid>.json files matched under {args.logs_dir}/")
        return

    combined = {}
    tasks_seen = set()
    for run_from_fname, (_jobid, path) in sorted(per_run_file.items()):
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        # Prefer the run_name embedded in the file (added by lm_eval_megatron.py);
        # fall back to the filename-derived one for older result files.
        run = payload.get("run_name") or run_from_fname
        combined[run] = {
            "checkpoint": payload.get("checkpoint"),
            "tasks": payload.get("tasks"),
            "num_fewshot": payload.get("num_fewshot"),
            "batch_size": payload.get("batch_size"),
            "limit": payload.get("limit"),
            "results": payload.get("results", {}),
        }
        tasks_seen.update(payload.get("results", {}).keys())

    json_path = args.out_prefix + ".json"
    with open(json_path, "w") as fh:
        json.dump(combined, fh, indent=2, default=str)

    tasks = sorted(tasks_seen)
    run_width = max(len("run"), max((len(r) for r in combined), default=0))
    col_width = max(8, max((len(t) for t in tasks), default=8))

    lines = []
    header = "run".ljust(run_width) + "  " + "  ".join(t.ljust(col_width) for t in tasks)
    lines.append(header)
    lines.append("-" * len(header))
    for run, entry in sorted(combined.items()):
        results = entry.get("results", {})
        row = [run.ljust(run_width)]
        for t in tasks:
            metric_key, val = pick_metric(t, results.get(t, {}))
            cell = f"{val:.4f}" if isinstance(val, (int, float)) else "-"
            row.append(cell.ljust(col_width))
        lines.append("  ".join(row))

    # Legend: which metric we picked per task.
    lines.append("")
    lines.append("metric per task:")
    for t in tasks:
        # use first run that has this task to report which key we picked
        for entry in combined.values():
            results = entry.get("results", {})
            if t in results:
                k, _ = pick_metric(t, results[t])
                lines.append(f"  {t}: {k}")
                break

    txt_path = args.out_prefix + ".txt"
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Aggregated {len(combined)} run(s), {len(tasks)} task(s).")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")


if __name__ == "__main__":
    main()
