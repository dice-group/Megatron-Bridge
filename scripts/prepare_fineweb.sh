#!/usr/bin/env bash
# Prepare FineWeb data for Nemotron 3 Super pretraining.
# Downloads a sample of FineWeb, converts to JSONL, tokenizes with
# Megatron's preprocess_data.py, and writes a blend JSON ready to pass
# to --per-split-data-args-path.
#
# Usage:
#   bash scripts/prepare_fineweb.sh \
#       --hf-model  /path/to/hf/model \
#       --output    /path/to/data/dir \
#       [--num-samples 5000000] \
#       [--workers 16]
#
# Outputs (in $OUTPUT_DIR):
#   fineweb_train.bin / .idx
#   fineweb_valid.bin / .idx
#   fineweb_test.bin  / .idx
#   blend.json

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
HF_MODEL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/hf-tokenizer"
OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/fineweb"
NUM_SAMPLES=5000000   # ~10 GB of text; adjust up for longer runs
WORKERS=16
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --hf-model)   HF_MODEL="$2";    shift 2 ;;
    --output)     OUTPUT_DIR="$2";  shift 2 ;;
    --num-samples)NUM_SAMPLES="$2"; shift 2 ;;
    --workers)    WORKERS="$2";     shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "Using hf-model: $HF_MODEL"
echo "Using output:   $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# ── step 1: download FineWeb and write per-split JSONL ───────────────────────

# Train / valid / test split: 98% / 1% / 1%
TRAIN_SAMPLES=$(( NUM_SAMPLES * 98 / 100 ))
VALID_SAMPLES=$(( NUM_SAMPLES *  1 / 100 ))
TEST_SAMPLES=$(( NUM_SAMPLES - TRAIN_SAMPLES - VALID_SAMPLES ))

if [[ -f "$OUTPUT_DIR/fineweb_train.jsonl" && -f "$OUTPUT_DIR/fineweb_valid.jsonl" && -f "$OUTPUT_DIR/fineweb_test.jsonl" ]]; then
  echo "[1/3] JSONL files already exist, skipping download."
else
  echo "[1/3] Downloading FineWeb (${NUM_SAMPLES} samples) and writing JSONL..."
  python3 - <<EOF
import json
from datasets import load_dataset

num_samples    = ${NUM_SAMPLES}
train_samples  = ${TRAIN_SAMPLES}
valid_samples  = ${VALID_SAMPLES}
test_samples   = ${TEST_SAMPLES}
output_dir     = "${OUTPUT_DIR}"

print(f"  Streaming FineWeb sample-350BT (first {num_samples:,} docs)...")
ds = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-350BT",
    split="train",
    streaming=True,
)

splits = {
    "train": (output_dir + "/fineweb_train.jsonl", train_samples),
    "valid": (output_dir + "/fineweb_valid.jsonl", valid_samples),
    "test":  (output_dir + "/fineweb_test.jsonl",  test_samples),
}

handles = {k: open(path, "w") for k, (path, _) in splits.items()}
counts  = {k: 0 for k in splits}

for doc in ds:
    text = doc.get("text", "").strip()
    if not text:
        continue
    for split, (_, limit) in splits.items():
        if counts[split] < limit:
            handles[split].write(json.dumps({"text": text}) + "\n")
            counts[split] += 1
            break
    if all(counts[k] >= lim for k, (_, lim) in splits.items()):
        break

for h in handles.values():
    h.close()

for split, count in counts.items():
    print(f"  {split}: {count:,} docs written")

import os as _os; _os._exit(0)
EOF
fi

# ── step 2: tokenize each split ──────────────────────────────────────────────
echo "[2/3] Tokenizing with preprocess_data.py..."

PREPROCESS="$REPO_ROOT/3rdparty/Megatron-LM/tools/preprocess_data.py"

for SPLIT in train valid test; do
  echo "  Tokenizing $SPLIT..."
  python3 "$PREPROCESS" \
    --input        "$OUTPUT_DIR/fineweb_${SPLIT}.jsonl" \
    --output-prefix "$OUTPUT_DIR/fineweb_${SPLIT}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$HF_MODEL" \
    --json-keys    text \
    --workers      "$WORKERS" \
    --append-eod
done

# ── step 3: write blend JSON ─────────────────────────────────────────────────
echo "[3/3] Writing blend.json..."

python3 - <<EOF
import json, os

output_dir = "${OUTPUT_DIR}"
blend = {
    "train": ["1.0", os.path.join(output_dir, "fineweb_train_text_document")],
    "valid": ["1.0", os.path.join(output_dir, "fineweb_valid_text_document")],
    "test":  ["1.0", os.path.join(output_dir, "fineweb_test_text_document")],
}
path = os.path.join(output_dir, "blend.json")
with open(path, "w") as f:
    json.dump(blend, f, indent=2)
print(f"  Written: {path}")
print(json.dumps(blend, indent=2))
EOF

echo ""
echo "Done! Run pretraining with:"
echo "  --per-split-data-args-path=${OUTPUT_DIR}/blend.json"
