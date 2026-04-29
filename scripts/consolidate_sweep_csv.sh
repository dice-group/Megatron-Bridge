#!/bin/bash
#
# Combine per-run sweep CSVs (routing diagnostics + val loss) into one file
# for analysis.
#
# Inputs (per run, written during training):
#   logs/sweep_diag_<RUN>.csv  — routing diagnostics, one row per N forward calls
#   logs/sweep_val_<RUN>.csv   — val loss, one row per eval (step, key, value)
#
# Output (default logs/sweep_combined.csv):
#   - All routing diagnostic rows from every run (stacked, run_name distinguishes)
#   - All val loss rows from every run, with same run_name column for joining
#
# We don't try to row-merge by step (the diag and val schemas differ); instead
# we stack all rows in two sections separated by a comment marker. The user
# (or me, downstream) reads them in pandas and joins on (run_name, step).
#
# Usage:
#   bash scripts/consolidate_sweep_csv.sh                  # → logs/sweep_combined.csv
#   bash scripts/consolidate_sweep_csv.sh path/to/out.csv  # custom output

set -euo pipefail

OUT="${1:-logs/sweep_combined.csv}"
DIAG_GLOB="logs/sweep_diag_*.csv"
VAL_GLOB="logs/sweep_val_*.csv"

# --- Routing diagnostics ---
diag_files=( $DIAG_GLOB )
if [ ! -e "${diag_files[0]}" ]; then
    echo "No diag files matching $DIAG_GLOB"
    exit 1
fi

head -1 "${diag_files[0]}" > "$OUT"
for f in "${diag_files[@]}"; do
    tail -n +2 "$f" >> "$OUT"
done

n_diag=${#diag_files[@]}
n_diag_rows=$(($(wc -l < "$OUT") - 1))

# --- Val loss (separator + new section) ---
val_files=( $VAL_GLOB )
n_val=0
n_val_rows=0
if [ -e "${val_files[0]}" ]; then
    echo "" >> "$OUT"
    echo "# === VAL LOSS SECTION ===" >> "$OUT"
    head -1 "${val_files[0]}" >> "$OUT"
    for f in "${val_files[@]}"; do
        tail -n +2 "$f" >> "$OUT"
    done
    n_val=${#val_files[@]}
    # Subtract 3 for blank line, comment, header
    n_val_rows=$(($(wc -l < "$OUT") - 1 - n_diag_rows - 3))
fi

size=$(du -h "$OUT" | cut -f1)
echo "Combined into $OUT  ($size)"
echo "  Routing diag: $n_diag file(s), $n_diag_rows rows"
echo "  Val loss   : $n_val file(s), $n_val_rows rows"
echo
echo "Per-run row counts (diag):"
for f in "${diag_files[@]}"; do
    rn=$(basename "$f" .csv | sed 's/^sweep_diag_//')
    rows=$(($(wc -l < "$f") - 1))
    printf "  %-40s %5d\n" "$rn" "$rows"
done
