#!/usr/bin/env python3
"""
Calculate the exact number of training iterations required for exactly 1 epoch
over a Megatron-LM dataset (MMapIndexedDataset).

Usage:
    python scripts/calc_iters.py \
        --data-prefix data/fineweb/fineweb_train_text_document \
        --global-batch-size 4 \
        --seq-length 2048
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Calculate exactly 1 epoch of training iterations")
    parser.add_argument("--data-prefix", required=True, help="Prefix of the Megatron dataset (.bin / .idx)")
    parser.add_argument("--global-batch-size", "-b", type=int, required=True, help="Train Global Batch Size")
    parser.add_argument("--seq-length", "-s", type=int, required=True, help="Sequence Length")
    
    args = parser.parse_args()

    # Add Megatron-LM to path to import MMapIndexedDataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    megatron_path = os.path.join(repo_root, "3rdparty", "Megatron-LM")
    
    if megatron_path not in sys.path:
        sys.path.insert(0, megatron_path)

    try:
        from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
    except ImportError as e:
        print(f"Error importing Megatron's MMapIndexedDataset: {e}")
        print(f"Looked in: {megatron_path}")
        sys.exit(1)

    print(f"Loading dataset from: {args.data_prefix}")
    
    # Check if files exist to provide a better error message
    idx_path = args.data_prefix + ".idx"
    bin_path = args.data_prefix + ".bin"
    if not os.path.exists(idx_path) or not os.path.exists(bin_path):
        print(f"\n[Error] Dataset files not found!")
        print(f"  Looking for: {idx_path}")
        print(f"  Looking for: {bin_path}")
        print("  Check your --data-prefix argument.")
        sys.exit(1)

    try:
        dataset = MMapIndexedDataset(args.data_prefix)
    except Exception as e:
        print(f"\nFailed to load dataset: {e}")
        sys.exit(1)

    # .sizes contains the lengths (in tokens) of every document in the index
    import numpy as np
    total_tokens = np.sum(dataset.sizes)
    total_docs = len(dataset.sizes)
    
    print("\n" + "="*50)
    print(f"Dataset Stats:")
    print(f"  Total Documents : {total_docs:,}")
    print(f"  Total Tokens    : {total_tokens:,}")
    print(f"  Avg Tokens/Doc  : {total_tokens / total_docs:.1f}")
    
    tokens_per_step = args.global_batch_size * args.seq_length
    print(f"\nTraining Settings:")
    print(f"  Global Batch Size : {args.global_batch_size}")
    print(f"  Sequence Length   : {args.seq_length}")
    print(f"  Tokens / Step     : {tokens_per_step:,}")

    total_iters = total_tokens / tokens_per_step
    
    print("\n" + "="*50)
    print(f"EXACT 1-EPOCH ITERATIONS : {int(total_iters):,}")
    print("="*50 + "\n")
    print(f"Suggested SLURM Script Setup:")
    print(f"  TRAIN_ITERS={int(total_iters)}")
    print(f"  LR_WARMUP_ITERS={max(10, int(total_iters * 0.1))}  # ~10% warmup")

if __name__ == "__main__":
    main()
