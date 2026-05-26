# Variable-K MoE Routing — Scripts & Gating

This directory contains the training launchers, sweep drivers, evaluation
pipeline, and data/model utilities used to develop and ablate the custom
variable-K MoE routers defined in
`3rdparty/Megatron-LM/megatron/core/transformer/moe/gate.py`.

---

## 1. Custom Gating Mechanism

A library of **variable-K MoE routers** that drop into Megatron-Core in place
of `TopKRouter`. All produce the standard `(probs, routing_map)` output and
share a unified diagnostic CSV (`logs/sweep_diag_<RUN>.csv`) gated by env
vars `SWEEP_DIAG_DENSE` / `SWEEP_DIAG_STRIDE`. Most routers honor
`TOPANY_FORCE_TOP1=1` to fall back to top-1 when a token would otherwise
route to zero experts.

**Shared primitive**

- `GAMoEGateSTEBackward` — Straight-Through Estimator: forward =
  `(scores > 0).float()`, backward = identity. Used by every binary-decision
  router.

**Routers**

| Class | Pre-activation | Decision | Load balancing |
|---|---|---|---|
| `TopAnyRouter` | Cosine-sim (orthogonal `sim_matrix`, scaled by `sigmoid_target·√d`) → σ | `σ(score) − σ(learned per-expert threshold)` via STE | Standard MoE aux loss + env `TOPANY_K_TARGET_COEFF` K-target loss |
| `LossFreeTopAnyRouter` | Same cosine + scale | STE with **buffer** thresholds; init at normal-quantile for `target_K/E` | **Loss-free**: thresholds updated outside autograd from per-expert error `e_i = actual_c − target_c`, mode `sign` or `magnitude`, all-reduced across TP×DP×CP, clamped to ±3·`sigmoid_target` (anti-windup) |
| `SigmoidGateRouter` | `Linear(d, E)` → σ | STE with fixed cutoff 0.5 | Standard aux loss + optional K-target |
| `LossFreeSigmoidRouter` | `Linear(d, E)` + per-expert bias `b_e` → σ | STE with fixed 0.5 cutoff | Loss-free per-expert bias updates (cosine analog, without sim-matrix drift) |
| `LossFreeSigmoidAnnealRouter` | As above | As above | Adds cosine anneal of `target_K` between `[START, END]` env-controlled steps |
| `ETRouter` | `Linear(d, E)` → σ | STE with per-expert threshold `c_e` | `c_e ← β·c_e + (1−β)·kth-largest(score_e)`, with `k` chosen so uniform routing → `target_K` |
| `TopPRouter` | `Linear(d, E)` → σ → sort + cumsum | Take experts until cumulative ≥ `top_p` (env `TOPP_THRESHOLD`) | None (env-tuned only) |
| `DynamicTopPRouter` | Inherits TopP | PI-controlled `top_p` updated each step toward `target_K` | PI controller (`DTOPP_KP`, `DTOPP_KI`) clamped to `[p_min, p_max]` |
| `ReMoERouter` | `Linear(d, E)` → ReLU | `gate > 0`, probs = raw gate (un-normalized) | L1 sparsity with per-expert weight = batch freq; λ multiplicatively updated toward `target_K` |
| `AdaMoERouter` | `Linear(d, N+m)` (N real + m null) → softmax | Top-(k+m′); null slots drop, real picks renormalize; FORCE_TOP1 fallback | Inherent via null competition |

**Diagnostics** (logged via `save_to_aux_losses_tracker`, always with an
`+_EPS = 1e-30` guard against the moe_utils "0 = layer not written" filter):
`topany_k_{mean,min,max,std}`, `topany_k_dist_i` per K-value,
`threshold_{mean,std,abs_max}`, `threshold_delta_abs_{mean,max}`,
`no_expert_fallback_frac`, `expert_load_{max,min}_over_mean`,
`load_balancing_loss`, `k_target_loss`.

### Environment variables

All router behavior is controlled by env vars read directly inside `gate.py`.
The sweep scripts (`sweep_combined_24h.sh`, `sweep_combined_1b_24h.sh`,
`sweep_routing_small.sh`) pass them through via `sbatch --export=ALL,...`;
when running ad-hoc, set them in your shell before invoking the launcher.

**Diagnostics infrastructure** (applies to all routers)

| Var | Default | Purpose |
|---|---|---|
| `RUN_NAME` | `unknown` | Tag written into every row of the per-run diagnostic CSV. Set per run. |
| `SWEEP_DIAG_FILE` | `logs/sweep_diag_<RUN>.csv` | Override the diagnostic CSV path. |
| `SWEEP_DIAG_DENSE` | `50` | Log every step for the first N steps. |
| `SWEEP_DIAG_STRIDE` | `200` | After the dense phase, log every N steps. |

**Universal routing knobs** (read by every variable-K router)

| Var | Default | Purpose |
|---|---|---|
| `TOPANY_FORCE_TOP1` | `1` | If a token routes to 0 experts, fall back to top-1. Set `0` to allow drops. |
| `TOPANY_K_TARGET_COEFF` | `0` | Coefficient for the K-target auxiliary loss. `0` disables it. |
| `TOPANY_K_TARGET` | `2.0` | Target average K for the K-target loss. |

**LossFreeSigmoidAnnealRouter** (cosine anneal of `target_K`)

| Var | Default | Purpose |
|---|---|---|
| `TOPANY_K_ANNEAL_START` | `target_K` | Starting `target_K` for the anneal. |
| `TOPANY_K_ANNEAL_END` | `target_K` | Final `target_K` after the anneal. |
| `TOPANY_K_ANNEAL_START_STEP` | `0` | Step at which the anneal begins. |
| `TOPANY_K_ANNEAL_END_STEP` | `0` | Step at which the anneal completes. |

**ETRouter**

| Var | Default | Purpose |
|---|---|---|
| `ET_EMA_BETA` | constructor arg | EMA decay for per-expert threshold tracking. |

**TopPRouter**

| Var | Default | Purpose |
|---|---|---|
| `TOPP_THRESHOLD` | constructor arg | Cumulative-probability cutoff for top-p selection. |
| `TOPP_ENTROPY_COEFF` | `0` | Coefficient on an entropy regularizer over routing probs. |

**DynamicTopPRouter** (PI-controlled `top_p`)

| Var | Default | Purpose |
|---|---|---|
| `DTOPP_P_INIT` | `p_init` ctor arg | Initial top-p value (also seeds `TOPP_THRESHOLD`). |
| `DTOPP_TARGET_K` | `moe_topany_target_k` | Target K the PI controller drives toward. |
| `DTOPP_KP` | `0.05` | Proportional gain. |
| `DTOPP_KI` | `0.005` | Integral gain. |
| `DTOPP_P_MIN` | `0.05` | Lower clamp on `top_p`. |
| `DTOPP_P_MAX` | `8.0` | Upper clamp on `top_p`. |

**ReMoERouter** (ReLU gate with L1 sparsity)

| Var | Default | Purpose |
|---|---|---|
| `REMOE_TARGET_K` | `moe_topany_target_k` | Target average K. |
| `REMOE_LAMBDA_INIT` | `1e-4` | Initial L1 coefficient. |
| `REMOE_LAMBDA_ALPHA` | `0.01` | Multiplicative step size for λ updates. |
| `REMOE_LAMBDA_MIN` | `1e-8` | Lower clamp on λ. |
| `REMOE_LAMBDA_MAX` | `1.0` | Upper clamp on λ. |

**AdaMoERouter** (null-expert softmax)

| Var | Default | Purpose |
|---|---|---|
| `ADAMOE_NUM_NULL` | constructor arg | Number of null experts (`m`). |
| `ADAMOE_TOPK` | constructor arg | Top-k including null slots. |

---

## 2. Scripts

### Training launchers (sbatch)

- `slurm_train_super.sh` — base Nemotron-3 Super sbatch driver.
- `slurm_train_super_topk_1b.sh` / `slurm_train_super_lossfree_1b.sh` — 1B
  Nemotron-3 Super: topk baseline vs cosine loss-free top-any. Parallelism
  presets (DP4, TP/EP variants) documented in the headers.
- `slurm_train_super_{small,1b,topk_1b,topany_1b}_1gpu.sh` — single-GPU
  smoke variants per routing type. `small_1gpu` is the unified
  router-switchable smoke; the others mirror the multi-GPU configs at 1-GPU
  scale.
- `slurm_train_nano_30b.sh` — 30B Nano (8-node).
- `slurm_train_qwen3_moe_{1b,30b}.sh` — Qwen3-MoE (1B 4-GPU and 30B 2-node).
- `slurm_train_gpt_oss_{1b,20b}.sh` — GPT-OSS MoE (1B 4-GPU and 20B 2-node).
- `local_train_super_small.sh` — non-sbatch sibling of
  `slurm_train_super_small_1gpu.sh`; launches `apptainer exec` + `torchrun`
  directly inside an existing allocation.

### Sweep orchestration

- `sweep_moe_routing.sh` — 36-run sweep (12 configs × 3 models, 6 h each)
  over routing × mode × rate.
- `sweep_routing_small.sh` — Sweep 3: sigmoid-linear router + fallback
  ablation, introduced after the cosine-threshold parameterization was
  exhausted in sweeps 1–2.
- `sweep_combined_24h.sh` — combined 24h supersweep replacing the three
  earlier 24h sweeps, with the `SAVE_INTERVAL` / async-save fix so
  checkpoints actually land on disk before walltime.
- `sweep_combined_1b_24h.sh` — 5-config 1B counterpart (topk baseline, two
  `kanneal` sigmoid_lossfree_anneal variants, two AdaMoE variants).
- `consolidate_sweep_csv.sh` — merges per-run `logs/sweep_diag_<RUN>.csv`
  and `logs/sweep_val_<RUN>.csv` into one analysis CSV.

### Evaluation

- `slurm_eval_lm_harness.sh` — runs lm-evaluation-harness on a
  Megatron-Bridge checkpoint; auto-loads architecture from
  `<checkpoint>/run_config.yaml` so no model flags need to be re-specified;
  pushes per-task metrics to W&B.
- `lm_eval_megatron.py` — generic lm-eval-harness adapter for any Bridge
  checkpoint (loglikelihood scoring only — sufficient for HellaSwag, ARC,
  PIQA, WinoGrande, BoolQ, OpenBookQA, MMLU, LAMBADA, GLUE subtasks).
- `submit_eval_sweep.sh` / `submit_eval_sweep_1b.sh` — submit one eval job
  per sweep run, locked to a common iteration (intersection of `iter_*`
  across runs) for apples-to-apples comparison.
- `aggregate_eval_results.py` — merges per-job eval JSONs into one results
  JSON plus a readable TXT table.

### Data prep & model utilities

- `prepare_fineweb.sh` — sbatch job that downloads and tokenizes FineWeb
  into Megatron `MMapIndexedDataset` shards.
- `generate_blend_json.sh` — writes `blend.json` (train/valid/test pointers)
  for a fineweb data directory.
- `create_container.sh` — sbatch job pulling the NeMo Apptainer container.
- `calc_iters.py` — computes the exact 1-epoch iteration count from an
  `MMapIndexedDataset` given global batch size + sequence length.
- `generate_nemotron_3_super.py` — interactive text generation from a
  trained Nemotron-3 Super checkpoint (reuses the pretrain recipe so the
  architecture matches exactly).

### Parameter counting / sizing

- `count_super_params.py` — Nemotron-3 Super LatentMoE param counter.
- `count_qwen3_moe_params.py` — Qwen3 MoE param counter + ~1B config search.
- `count_gpt_oss_params.py` — GPT-OSS MoE param counter + ~1B config search.
- `evaluate_params.py` — small brute-force filter for layer/FFN combos.
