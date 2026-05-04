#!/usr/bin/env python3
"""
lm-evaluation-harness adapter for Megatron-Bridge checkpoints.

Generic — works with any checkpoint produced by the pretrain entry points
in this repo (Nemotron-3 Super/Nano, Qwen3 MoE, GPT-OSS, etc.) because the
full training config is loaded from `<checkpoint_dir>/run_config.yaml`.
No need to re-specify model.num_layers / hybrid_override_pattern / etc.

Implements loglikelihood scoring only — sufficient for HellaSwag, ARC,
PIQA, WinoGrande, BoolQ, OpenBookQA, MMLU, LAMBADA, and the GLUE
subtasks. Generation (generate_until) is not implemented.

Usage:
  torchrun --nproc-per-node=1 scripts/lm_eval_megatron.py \
      --checkpoint /scratch/.../checkpoints_lossfree \
      --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande \
      --batch-size 8 \
      --output-path eval_results.json

The checkpoint must contain `run_config.yaml` (either at the top level or
inside the latest `iter_*/` subdir — both are searched automatically).
"""

import argparse
import copy
import json
import logging
import os
from typing import List, Tuple

import torch
import yaml
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

from megatron.bridge.training.config import ConfigContainer, runtime_config_update
from megatron.bridge.training.setup import initialize_megatron, _validate_and_set_vocab_size
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.checkpointing import load_checkpoint
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.checkpoint_utils import CONFIG_FILE
from megatron.bridge.training.utils.config_utils import InstantiationMode
from megatron.bridge.utils.common_utils import print_rank_0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint config loading
# ---------------------------------------------------------------------------

def find_run_config(checkpoint_path: str) -> str:
    """Locate run_config.yaml in a checkpoint dir.

    Looks for `<checkpoint_path>/run_config.yaml` first; if absent, picks the
    `run_config.yaml` from the latest `iter_*` subdir.
    """
    top = os.path.join(checkpoint_path, CONFIG_FILE)
    if os.path.exists(top):
        return top
    iter_dirs = [
        d for d in os.listdir(checkpoint_path)
        if d.startswith("iter_") and os.path.isdir(os.path.join(checkpoint_path, d))
    ]
    if not iter_dirs:
        raise FileNotFoundError(
            f"No run_config.yaml at {top} and no iter_* subdirs in {checkpoint_path}"
        )
    iter_dirs.sort(key=lambda d: int(d.replace("iter_", "")))
    candidate = os.path.join(checkpoint_path, iter_dirs[-1], CONFIG_FILE)
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"No run_config.yaml found in {checkpoint_path}")
    return candidate


_DROP_FIELDS_ON_LOAD = [
    # The trainer serializes a tokenizer *instance* into dataset.tokenizer; on
    # reload its constructor mutates a `config` arg that comes back as None and
    # raises TypeError. We rebuild the tokenizer ourselves from cfg.tokenizer
    # below, so dropping this is safe.
    ("dataset", "tokenizer"),
]


def _sanitize_run_config_dict(d: dict) -> dict:
    d = copy.deepcopy(d)
    for parent, child in _DROP_FIELDS_ON_LOAD:
        if isinstance(d.get(parent), dict) and child in d[parent]:
            d[parent][child] = None
    return d


def load_inference_config(checkpoint_path: str) -> ConfigContainer:
    """Load the saved training config and force inference-friendly settings."""
    run_config_path = find_run_config(checkpoint_path)
    print_rank_0(f"Loading config from: {run_config_path}")
    with open(run_config_path) as f:
        raw = yaml.safe_load(f)
    raw = _sanitize_run_config_dict(raw)
    cfg: ConfigContainer = ConfigContainer.from_dict(raw, mode=InstantiationMode.LENIENT)

    # Point the loader at this checkpoint, disable saving/wandb/etc.
    cfg.checkpoint.load = checkpoint_path
    cfg.checkpoint.save = None
    cfg.train.train_iters = 0
    cfg.train.eval_iters = 0
    cfg.train.micro_batch_size = 1
    if hasattr(cfg, "logger"):
        cfg.logger.wandb_project = None
        cfg.logger.wandb_entity = None

    # Force single-GPU parallelism. torch_dist checkpoints reshard on load,
    # so this works even if training used larger TP/EP/PP.
    saved_tp = cfg.model.tensor_model_parallel_size
    saved_ep = cfg.model.expert_model_parallel_size
    saved_pp = getattr(cfg.model, "pipeline_model_parallel_size", 1)
    saved_cp = cfg.model.context_parallel_size
    if (saved_tp, saved_ep, saved_pp, saved_cp) != (1, 1, 1, 1):
        print_rank_0(
            f"[parallelism] checkpoint trained with TP={saved_tp} EP={saved_ep} "
            f"PP={saved_pp} CP={saved_cp} — overriding to 1/1/1/1 for eval"
        )
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    if hasattr(cfg.model, "pipeline_model_parallel_size"):
        cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.mtp_num_layers = 0  # disable MTP head for eval

    return cfg


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

_TOKENIZER_CHECK_STRINGS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def add(a, b): return a + b",
    "1 + 1 = 2. Two plus two equals four.",
]


def check_tokenizer_alignment(megatron_tok, hf_tok) -> None:
    """Confirm the HF tokenizer used for lm-eval matches the one the model was trained with.

    Raises if any test string produces different IDs (after stripping a possible
    BOS/leading-special prefix from the megatron side, which some wrappers add).
    """
    hf_id = getattr(hf_tok, "name_or_path", "?")
    print_rank_0(f"[tokenizer-check] HF tokenizer: {hf_id}")
    for s in _TOKENIZER_CHECK_STRINGS:
        m_ids = list(megatron_tok.tokenize(s))
        h_ids = list(hf_tok.encode(s, add_special_tokens=False))
        # If megatron added a BOS at the front, drop it for comparison.
        if m_ids and h_ids and m_ids[0] != h_ids[0] and m_ids[1:] == h_ids:
            m_ids = m_ids[1:]
        if m_ids != h_ids:
            raise RuntimeError(
                f"Tokenizer mismatch on {s!r}\n"
                f"  megatron: {m_ids[:24]}\n"
                f"  hf     : {h_ids[:24]}\n"
                f"The HF tokenizer used by lm-eval will not produce the IDs "
                f"the model was trained on. Eval results would be invalid."
            )
    print_rank_0(f"[tokenizer-check] OK on {len(_TOKENIZER_CHECK_STRINGS)} test strings")


def check_lossfree_bias(model) -> None:
    """If the model has lossfree routers, confirm their bias buffers loaded non-zero."""
    found = []
    for m in model:
        inner = m.module if hasattr(m, "module") else m
        for name, mod in inner.named_modules():
            if hasattr(mod, "lf_bias") and isinstance(mod.lf_bias, torch.Tensor):
                b = mod.lf_bias.detach().float()
                found.append((name, b.abs().max().item(), b.mean().item(), b.std().item()))
    if not found:
        print_rank_0("[bias-check] no lossfree routers found (topk/topany/other routing)")
        return
    print_rank_0(f"[bias-check] found {len(found)} lossfree router(s):")
    for name, mx, mean, std in found:
        print_rank_0(f"  {name}: |max|={mx:.4f} mean={mean:+.4f} std={std:.4f}")
    nonzero = sum(1 for _, mx, _, _ in found if mx > 1e-6)
    if nonzero == 0:
        raise RuntimeError(
            "All lf_bias tensors are zero — checkpoint did not load the trained "
            "router bias (or model trained for 0 steps). Aborting eval."
        )
    print_rank_0(f"[bias-check] {nonzero}/{len(found)} routers have non-zero trained bias")


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class _SingleBatchIterator:
    def __init__(self, input_ids, position_ids):
        self.batch = dict(tokens=input_ids, position_ids=position_ids)
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def _forward_step(data_iterator, model, **kwargs):
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


# ---------------------------------------------------------------------------
# LM adapter
# ---------------------------------------------------------------------------

class MegatronLM(LM):
    def __init__(self, model, hf_tokenizer, max_length: int, batch_size: int = 8):
        super().__init__()
        self.model = model
        self.tokenizer = hf_tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id

    def _forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(B, T)
        with torch.no_grad():
            fwd_bwd = get_forward_backward_func()
            iterator = _SingleBatchIterator(input_ids, position_ids)
            output = fwd_bwd(
                forward_step_func=_forward_step,
                data_iterator=iterator,
                model=self.model,
                num_microbatches=1,
                forward_only=True,
                seq_length=T,
                micro_batch_size=B,
                collect_non_loss_data=True,
            )
        if isinstance(output, list) and len(output) > 0:
            output = output[0]
        tp_world = parallel_state.get_tensor_model_parallel_world_size()
        if tp_world > 1:
            gathered = [torch.zeros_like(output) for _ in range(tp_world)]
            torch.distributed.all_gather(
                gathered, output, group=parallel_state.get_tensor_model_parallel_group()
            )
            output = torch.cat(gathered, dim=2)
        return output

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        encoded = []
        for req in requests:
            context, continuation = req.args
            if context == "":
                ctx_ids = [self.eos_id]
            else:
                ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
            cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
            encoded.append((ctx_ids, cont_ids))

        order = sorted(range(len(encoded)), key=lambda i: -(len(encoded[i][0]) + len(encoded[i][1])))
        out: List = [None] * len(encoded)

        for start in range(0, len(order), self.batch_size):
            idx_chunk = order[start : start + self.batch_size]
            chunk = [encoded[i] for i in idx_chunk]

            seqs, cont_lens = [], []
            for ctx_ids, cont_ids in chunk:
                joined = (ctx_ids + cont_ids)[-(self.max_length):]
                cont_kept = min(len(cont_ids), len(joined) - 1)
                if cont_kept <= 0:
                    seqs.append(joined if len(joined) >= 2 else (joined + [self.eos_id]))
                    cont_lens.append(0)
                else:
                    seqs.append(joined)
                    cont_lens.append(cont_kept)

            T_in = max(len(s) - 1 for s in seqs)
            B = len(seqs)
            input_ids = torch.full((B, T_in), self.eos_id, dtype=torch.long)
            for j, s in enumerate(seqs):
                inp = s[:-1]
                input_ids[j, : len(inp)] = torch.tensor(inp, dtype=torch.long)
            input_ids = input_ids.cuda()

            logits = self._forward_logits(input_ids)
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            for j, (s, k) in enumerate(zip(seqs, cont_lens)):
                if k == 0:
                    out[idx_chunk[j]] = (0.0, True)
                    continue
                target = torch.tensor(s[-k:], dtype=torch.long, device=input_ids.device)
                pred_slice = log_probs[j, len(s) - 1 - k : len(s) - 1]
                tok_lp = pred_slice.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                ll = tok_lp.sum().item()
                greedy = bool((pred_slice.argmax(-1) == target).all().item())
                out[idx_chunk[j]] = (float(ll), greedy)

        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError("loglikelihood_rolling not implemented")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("generate_until not implemented")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness on a Megatron-Bridge checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint dir (containing run_config.yaml or iter_*/)")
    parser.add_argument("--tasks", type=str, required=True,
                        help="Comma-separated lm-eval task names (e.g. hellaswag,arc_easy)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples per task (for smoke tests)")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    parser.add_argument("--skip-bias-check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_cli_args()

    cfg = load_inference_config(args.checkpoint)
    runtime_config_update(cfg)

    state = GlobalState()
    state.cfg = cfg
    initialize_megatron(cfg=cfg)

    megatron_tok = build_tokenizer(cfg.tokenizer)
    cfg.model.vocab_size, cfg.model.should_pad_vocab = _validate_and_set_vocab_size(
        model_vocab_size=cfg.model.vocab_size,
        tokenizer_vocab_size=megatron_tok.vocab_size,
    )

    model = cfg.model.provide_distributed_model(wrap_with_ddp=False)
    print_rank_0(f"Loading checkpoint weights from: {cfg.checkpoint.load}")
    load_checkpoint(state, model, None, None)

    # Disable MTP after restore (training config restores mtp_num_layers > 0)
    for m in model:
        inner = m.module if hasattr(m, "module") else m
        if hasattr(inner, "config"):
            inner.config.mtp_num_layers = None
        if hasattr(inner, "mtp_process"):
            inner.mtp_process = False

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()

    hf_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.tokenizer_model, trust_remote_code=True)

    if not args.skip_tokenizer_check:
        check_tokenizer_alignment(megatron_tok, hf_tokenizer)
    if not args.skip_bias_check:
        check_lossfree_bias(model)

    lm = MegatronLM(
        model=model,
        hf_tokenizer=hf_tokenizer,
        max_length=int(cfg.model.seq_length),
        batch_size=args.batch_size,
    )

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print_rank_0(f"Running lm-eval on tasks: {task_list}")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    if torch.distributed.get_rank() == 0:
        summary = results.get("results", {})
        print("\n========= lm-eval results =========")
        print(json.dumps(summary, indent=2, default=str))
        if args.output_path:
            with open(args.output_path, "w") as f:
                json.dump({"results": summary}, f, indent=2, default=str)
            print(f"Wrote {args.output_path}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
