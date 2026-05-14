#!/bin/bash
#
# Submit lm-evaluation-harness jobs for every training run under
# sweep_ckpts_combined_24h. One eval job per run, targeting a *common*
# checkpoint iteration so val/eval comparisons across runs are apples-to-apples
# (since e.g. small_topk advanced further than slower routers within the 24h
# wall clock). The default chosen iter is the largest iter_* present in every
# selected run (intersection of iter sets). Override with EVAL_ITER=<n> or
# EVAL_ITER=latest (old per-run-latest behavior).
#
# Per-task metrics push to W&B under the original training run name.
#
# Usage:
#   bash scripts/submit_eval_sweep.sh                       # A100, common iter
#   GPU_TYPE=h100 bash scripts/submit_eval_sweep.sh         # H100
#   EVAL_ITER=10000 bash scripts/submit_eval_sweep.sh       # force iter 10000
#   EVAL_ITER=latest bash scripts/submit_eval_sweep.sh      # per-run latest
#   DRY_RUN=1 bash scripts/submit_eval_sweep.sh             # print plan only
#
#   # smoke test on one run, one task
#   ONLY="small_topk_24h" LIMIT=50 TASKS=hellaswag bash scripts/submit_eval_sweep.sh

set -euo pipefail

CKPT_BASE="${CKPT_BASE:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/sweep_ckpts_combined_24h}"
GPU_TYPE="${GPU_TYPE:-a100}"
WANDB_PROJECT="${WANDB_PROJECT:-variable-moe-routing}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq,openbookqa}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LIMIT="${LIMIT:-}"
# Optional whitelist: ONLY="name1 name2" filters to those run names.
ONLY="${ONLY:-}"
# Common eval iter: unset = auto (largest iter present in every run);
# integer = force that iter; "latest" = per-run latest (legacy behavior).
EVAL_ITER="${EVAL_ITER:-}"
# DRY_RUN=1 resolves the iter and prints the per-run plan without submitting
# sbatch jobs or creating staging dirs/symlinks.
DRY_RUN="${DRY_RUN:-0}"

if [ ! -d "$CKPT_BASE" ]; then
    echo "ERROR: CKPT_BASE not found: $CKPT_BASE" >&2
    exit 1
fi

mkdir -p logs

# ── Resolve which iter to evaluate ────────────────────────────────────────
# When EVAL_ITER is empty or numeric, we stage per-run dirs that symlink
# a specific iter_XXXXXXX subdir + write a legacy tracker file so the
# Megatron-Bridge loader picks exactly that iter (it falls back to
# latest_checkpointed_iteration.txt when no new-format train_state is present).
COMMON_ITER=""
if [ "$EVAL_ITER" != "latest" ]; then
    COMMON_ITER=$(python3 - "$CKPT_BASE" "$ONLY" "$EVAL_ITER" <<'PY'
import os, re, sys
base, only_str, forced = sys.argv[1], sys.argv[2], sys.argv[3]
only = set(only_str.split()) if only_str else None
sets = []
runs = []
for r in sorted(os.listdir(base)):
    p = os.path.join(base, r)
    if not os.path.isdir(p) or r.startswith("."):
        continue
    if only is not None and r not in only:
        continue
    iters = set()
    for d in os.listdir(p):
        m = re.match(r"iter_(\d+)$", d)
        if m and os.path.isfile(os.path.join(p, d, "run_config.yaml")):
            iters.add(int(m.group(1)))
    if iters:
        sets.append(iters)
        runs.append(r)
if not sets:
    sys.stderr.write("no evaluable runs found\n"); sys.exit(1)
if forced:
    target = int(forced)
    missing = [r for r, s in zip(runs, sets) if target not in s]
    if missing:
        sys.stderr.write(
            f"EVAL_ITER={target} not present in: {' '.join(missing)}\n"
        )
        sys.exit(1)
else:
    common = set.intersection(*sets)
    if not common:
        sys.stderr.write(
            "no common iter across runs — pass EVAL_ITER=<n> or EVAL_ITER=latest\n"
        )
        sys.exit(1)
    target = max(common)
print(target)
PY
)
    echo "Evaluation iter: $COMMON_ITER (common across selected runs)"
fi

STAGE_BASE="$CKPT_BASE/.eval_staging"
if [ -n "$COMMON_ITER" ]; then
    STAGE_DIR="$STAGE_BASE/iter_${COMMON_ITER}"
    if [ "$DRY_RUN" = "0" ]; then
        mkdir -p "$STAGE_DIR"
    fi
fi

if [ "$DRY_RUN" != "0" ]; then
    echo "=== DRY RUN (no sbatch / no staging-dir writes) ==="
fi

n_submitted=0
n_skipped=0
eval_job_ids=()

for run_dir in "$CKPT_BASE"/*/; do
    run_dir="${run_dir%/}"
    name=$(basename "$run_dir")

    # Skip the staging dir itself and any other dotfile dirs.
    case "$name" in .*) continue ;; esac

    if [ -n "$ONLY" ] && ! [[ " $ONLY " == *" $name "* ]]; then
        continue
    fi

    if [ -n "$COMMON_ITER" ]; then
        iter_pad=$(printf 'iter_%07d' "$COMMON_ITER")
        src_iter="$run_dir/$iter_pad"
        if [ ! -f "$src_iter/run_config.yaml" ]; then
            echo "[skip] $name — $iter_pad/run_config.yaml missing"
            n_skipped=$((n_skipped + 1))
            continue
        fi
        # Per-run staging dir: symlink the chosen iter + legacy tracker file.
        stage_run="$STAGE_DIR/$name"
        if [ "$DRY_RUN" = "0" ]; then
            mkdir -p "$stage_run"
            ln -sfn "$src_iter" "$stage_run/$iter_pad"
            echo "$COMMON_ITER" > "$stage_run/latest_checkpointed_iteration.txt"
            # Mirror top-level run_config.yaml if the trainer dropped one there.
            if [ -f "$run_dir/run_config.yaml" ]; then
                ln -sfn "$run_dir/run_config.yaml" "$stage_run/run_config.yaml"
            fi
        fi
        ckpt_path="$stage_run"
        iter_num="$COMMON_ITER"
    else
        # EVAL_ITER=latest — use the run dir as-is and let the loader pick.
        latest_iter=$(ls -d "$run_dir"/iter_* 2>/dev/null | sort -V | tail -1 || true)
        if [ -z "$latest_iter" ] || [ ! -f "$latest_iter/run_config.yaml" ]; then
            echo "[skip] $name — no iter_*/run_config.yaml under $run_dir"
            n_skipped=$((n_skipped + 1))
            continue
        fi
        ckpt_path="$run_dir"
        iter_num=$(basename "$latest_iter" | sed 's/iter_0*//')
    fi

    if [ "$DRY_RUN" != "0" ]; then
        echo "[dry-run] $name  iter=$iter_num  ckpt=$ckpt_path"
        n_submitted=$((n_submitted + 1))
        continue
    fi

    echo "Submitting eval: $name  (iter=$iter_num)"

    # sbatch --export parses commas as variable separators, which mangles
    # TASKS=hellaswag,arc_easy,... — forward via the process env + --export=ALL.
    # --parsable returns just the job id so we can chain an aggregation job.
    job_id=$(CHECKPOINT="$ckpt_path" \
    TASKS="$TASKS" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_FEWSHOT="$NUM_FEWSHOT" \
    LIMIT="$LIMIT" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_RUN_NAME="$name" \
    EVAL_ITER="$iter_num" \
    sbatch --parsable \
        --gres=gpu:${GPU_TYPE}:1 \
        --job-name="eval-${name}" \
        --export=ALL \
        scripts/slurm_eval_lm_harness.sh)
    eval_job_ids+=("$job_id")
    n_submitted=$((n_submitted + 1))
done

# Chain an aggregation job that runs once every eval finishes (afterany so
# partial results still aggregate if some jobs fail). Pure-python, no GPU.
agg_job_id=""
if [ "$n_submitted" -gt 0 ] && [ "$DRY_RUN" = "0" ]; then
    dep_list=$(IFS=:; echo "${eval_job_ids[*]}")
    agg_job_id=$(sbatch --parsable \
        --job-name="eval-aggregate" \
        --dependency=afterany:"$dep_list" \
        --kill-on-invalid-dep=yes \
        --time=00:10:00 \
        --cpus-per-task=1 \
        --mem=4GB \
        --output=logs/eval_aggregate_%j.out \
        --error=logs/eval_aggregate_%j.err \
        --wrap "python scripts/aggregate_eval_results.py --logs-dir logs --out-prefix logs/eval_all_results")
fi

echo
if [ "$DRY_RUN" != "0" ]; then
    echo "Dry run: would submit $n_submitted eval job(s); skipped $n_skipped."
else
    echo "Submitted $n_submitted eval job(s); skipped $n_skipped."
fi
if [ -n "$COMMON_ITER" ]; then
    echo "Iter:       $COMMON_ITER  (staged at $STAGE_DIR/<run_name>)"
else
    echo "Iter:       per-run latest (EVAL_ITER=latest)"
fi
if [ -n "$agg_job_id" ]; then
    echo "Aggregate:  job $agg_job_id (runs after all evals via afterany)"
fi
echo "Track:    squeue -u \$USER"
echo "W&B:      $WANDB_PROJECT"
echo "Results:  logs/eval_<run_name>_<jobid>.json (per-run)"
echo "          logs/eval_all_results.json + .txt (aggregated)"
