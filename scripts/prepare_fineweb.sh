#!/bin/bash

#SBATCH --job-name=prepare-fineweb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --mem=128GB
#SBATCH --account=hpc-prf-merlin
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# HuggingFace tokenizer model directory
HF_MODEL=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/hf-tokenizer

# Where to write JSONL, tokenized .bin/.idx, and blend.json
OUTPUT_DIR=/scratch/hpc-prf-merlin/luke/Megatron-Bridge/data/fineweb

# FineWeb dataset config name. Options:
#   sample-10BT   ~20 GB tokenized   (quick testing)
#   sample-100BT  ~200 GB tokenized
#   sample-350BT  ~700 GB tokenized
#   default       full ~15T-token dataset (~30 TB tokenized)
DATASET_NAME="sample-10BT"

# Number of documents to download: 0 = download ALL documents in DATASET_NAME.
# 3M docs ≈ 1.5B tokens ≈ ~10h training on 4×H100 (GBS=16, seq=2048)
NUM_SAMPLES=3000000

# Parallel workers for downloading and tokenization
WORKERS=32

# ==============================================================================
# Derived paths — do not edit
# ==============================================================================

set -euo pipefail

CONTAINER="$PWD/../nemo-container"
REPO_MOUNT=/opt/Megatron-Bridge
PREPROCESS="$REPO_MOUNT/3rdparty/Megatron-LM/tools/preprocess_data.py"
PARTITIONS=2

# Train / valid / test split: 98% / 1% / 1%
# When NUM_SAMPLES=0 we stream the entire dataset; split limits are set to
# effectively unlimited values and the loop ends when the dataset is exhausted.
if [[ "$NUM_SAMPLES" -eq 0 ]]; then
  TRAIN_SAMPLES=0
  VALID_SAMPLES=0
  TEST_SAMPLES=0
else
  TRAIN_SAMPLES=$(( NUM_SAMPLES * 98 / 100 ))
  VALID_SAMPLES=$(( NUM_SAMPLES *  1 / 100 ))
  TEST_SAMPLES=$(( NUM_SAMPLES - TRAIN_SAMPLES - VALID_SAMPLES ))
fi

# ==============================================================================

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

module load tools/Apptainer/1.3.5-GCCcore-13.3.0

echo "=============================="
echo "Job ID    : ${SLURM_JOB_ID:-local}"
echo "Node      : ${SLURMD_NODENAME:-$(hostname)}"
echo "Container : $CONTAINER"
echo "HF Model  : $HF_MODEL"
echo "Output    : $OUTPUT_DIR"
echo "Dataset   : $DATASET_NAME"
echo "Samples   : $NUM_SAMPLES (0=all)"
echo "Workers   : $WORKERS"
echo "=============================="

# Helper: run python3 inside the container with all necessary bind mounts
run_python() {
  apptainer exec \
    --no-home \
    --bind "$PWD":"$REPO_MOUNT" \
    --bind "$OUTPUT_DIR":"$OUTPUT_DIR" \
    --bind "$HF_MODEL":"$HF_MODEL" \
    --pwd "$REPO_MOUNT" \
    "$CONTAINER" \
    env HOME=/tmp TOKENIZERS_PARALLELISM=false \
    python3 "$@"
}

# ── step 1: download FineWeb and write per-split JSONL ───────────────────────

if [[ -f "$OUTPUT_DIR/fineweb_train_text_document.bin" && \
      -f "$OUTPUT_DIR/fineweb_valid_text_document.bin" && \
      -f "$OUTPUT_DIR/fineweb_test_text_document.bin" ]]; then
  echo "[1/3] Tokenized files already exist, skipping download and tokenization."
  # Jump straight to blend.json (step 3)
elif [[ -f "$OUTPUT_DIR/fineweb_train.jsonl" && -f "$OUTPUT_DIR/fineweb_valid.jsonl" && -f "$OUTPUT_DIR/fineweb_test.jsonl" ]]; then
  echo "[1/3] JSONL files already exist, skipping download."
else
  if [[ "$NUM_SAMPLES" -eq 0 ]]; then
    echo "[1/3] Downloading FULL FineWeb dataset and writing JSONL..."
  else
    echo "[1/3] Downloading FineWeb (${NUM_SAMPLES} samples) and writing JSONL..."
  fi
  run_python - <<EOF
import json, os, sys

# Redirect HF cache to writable location before importing datasets
_hf_cache = "${OUTPUT_DIR}/.hf_cache"
os.makedirs(_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _hf_cache)

from multiprocessing import Process, Queue, Value
from ctypes import c_bool
from datasets import load_dataset
from tqdm import tqdm

# Use orjson for faster serialization if available
try:
    import orjson
    def dumps(obj): return orjson.dumps(obj).decode()
except ImportError:
    def dumps(obj): return json.dumps(obj)

num_samples    = ${NUM_SAMPLES}
train_samples  = ${TRAIN_SAMPLES}
valid_samples  = ${VALID_SAMPLES}
test_samples   = ${TEST_SAMPLES}
output_dir     = "${OUTPUT_DIR}"
dataset_name   = "${DATASET_NAME}"
num_dl_workers = min(${WORKERS}, 32)

# num_samples == 0 means unlimited (stream entire named config)
unlimited = num_samples == 0

if unlimited:
    print(f"  Downloading ALL of FineWeb/{dataset_name} with {num_dl_workers} parallel workers...")
else:
    print(f"  Downloading FineWeb/{dataset_name} ({num_samples:,} docs) with {num_dl_workers} parallel workers...")

def worker_fn(worker_id, num_workers, queue, stop_flag):
    os.environ.setdefault("HF_HOME", _hf_cache)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name=dataset_name,
        split="train",
        streaming=True,
    ).shard(num_shards=num_workers, index=worker_id)
    for doc in ds:
        if stop_flag.value:
            break
        text = doc.get("text", "").strip()
        if text:
            while not stop_flag.value:
                try:
                    queue.put(text, timeout=0.1)
                    break
                except Exception:
                    pass
    queue.put(None)  # sentinel

queue      = Queue(maxsize=50_000)
stop_flag  = Value(c_bool, False)
workers    = [Process(target=worker_fn, args=(i, num_dl_workers, queue, stop_flag), daemon=True)
              for i in range(num_dl_workers)]
for p in workers:
    p.start()

splits = {
    "train": (output_dir + "/fineweb_train.jsonl", train_samples),
    "valid": (output_dir + "/fineweb_valid.jsonl", valid_samples),
    "test":  (output_dir + "/fineweb_test.jsonl",  test_samples),
}

handles      = {k: open(path, "w", buffering=1 << 20) for k, (path, _) in splits.items()}
counts       = {k: 0 for k in splits}
total        = 0
done_workers = 0

# In unlimited mode we use a ratio-based split: write 98/100 docs to train,
# then 1/100 to valid, then 1/100 to test, cycling through.
# In limited mode we fill each split up to its limit as before.
pbar_total = None if unlimited else num_samples
with tqdm(total=pbar_total, unit="doc", dynamic_ncols=True) as pbar:
    while done_workers < num_dl_workers:
        if not unlimited and total >= num_samples:
            stop_flag.value = True
            break
        try:
            item = queue.get(timeout=1.0)
        except Exception:
            continue
        if item is None:
            done_workers += 1
            continue

        if unlimited:
            # Ratio-based split: cycle 98 train, 1 valid, 1 test
            cycle_pos = total % 100
            if cycle_pos < 98:
                split = "train"
            elif cycle_pos < 99:
                split = "valid"
            else:
                split = "test"
            handles[split].write(dumps({"text": item}) + "\n")
            counts[split] += 1
            total += 1
            pbar.update(1)
        else:
            for split, (_, limit) in splits.items():
                if counts[split] < limit:
                    handles[split].write(dumps({"text": item}) + "\n")
                    counts[split] += 1
                    total += 1
                    pbar.update(1)
                    break
            if total >= num_samples:
                stop_flag.value = True
                break

stop_flag.value = True
for h in handles.values():
    h.flush()
    h.close()
for p in workers:
    p.terminate()
    p.join(timeout=5)

for split, count in counts.items():
    print(f"  {split}: {count:,} docs written")
print(f"  Total: {total:,} docs written")

import os as _os; _os._exit(0)
EOF
fi

# ── step 2: tokenize each split ──────────────────────────────────────────────
if [[ -f "$OUTPUT_DIR/fineweb_train_text_document.bin" && \
      -f "$OUTPUT_DIR/fineweb_valid_text_document.bin" && \
      -f "$OUTPUT_DIR/fineweb_test_text_document.bin" ]]; then
  echo "[2/3] Tokenized files already exist, skipping tokenization."
else
  echo "[2/3] Tokenizing with preprocess_data.py..."

  for SPLIT in train valid test; do
    echo "  Tokenizing $SPLIT..."
    run_python "$PREPROCESS" \
      --input        "$OUTPUT_DIR/fineweb_${SPLIT}.jsonl" \
      --output-prefix "$OUTPUT_DIR/fineweb_${SPLIT}" \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model "$HF_MODEL" \
      --json-keys    text \
      --workers      "$WORKERS" \
      --partitions   "$PARTITIONS" \
      --append-eod

    # Clean up partition files (preprocess_data.py merges internally)
    for i in $(seq 0 $((PARTITIONS-1))); do
      rm -f "$OUTPUT_DIR/fineweb_${SPLIT}_${i}"*.bin \
            "$OUTPUT_DIR/fineweb_${SPLIT}_${i}"*.idx \
            "$OUTPUT_DIR/fineweb_${SPLIT}_${i}".jsonl
    done
  done

  # Remove intermediate JSONL files now that .bin/.idx are written
  echo "  Cleaning up intermediate JSONL files..."
  rm -f "$OUTPUT_DIR"/fineweb_*.jsonl
fi

# ── step 3: write blend JSON ─────────────────────────────────────────────────
echo "[3/3] Writing blend.json..."

run_python - <<EOF
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
echo "=============================="
echo "Done! Run pretraining with:"
echo "  --per-split-data-args-path=${OUTPUT_DIR}/blend.json"
echo "=============================="
