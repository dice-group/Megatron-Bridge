#!/bin/bash
#
# Concatenate per-run sweep CSVs into one file for analysis.
# Each sweep job writes logs/sweep_diag_<RUN_NAME>.csv; this glues them
# into a single CSV. The run_name column distinguishes rows.
#
# Usage:
#   bash scripts/consolidate_sweep_csv.sh                  # → logs/sweep_combined.csv
#   bash scripts/consolidate_sweep_csv.sh path/to/out.csv  # custom output
#
set -euo pipefail

OUT="${1:-logs/sweep_combined.csv}"
GLOB="logs/sweep_diag_*.csv"

files=( $GLOB )
if [ ! -e "${files[0]}" ]; then
    echo "No files matching $GLOB"
    exit 1
fi

# Header from first file, body (no header) from all of them.
head -1 "${files[0]}" > "$OUT"
for f in "${files[@]}"; do
    tail -n +2 "$f" >> "$OUT"
done

n_files=${#files[@]}
n_rows=$(($(wc -l < "$OUT") - 1))
size=$(du -h "$OUT" | cut -f1)
echo "Combined $n_files file(s) into $OUT  ($n_rows rows, $size)"
echo "Per-run row counts:"
for f in "${files[@]}"; do
    rn=$(basename "$f" .csv | sed 's/^sweep_diag_//')
    rows=$(($(wc -l < "$f") - 1))
    printf "  %-40s %5d\n" "$rn" "$rows"
done
