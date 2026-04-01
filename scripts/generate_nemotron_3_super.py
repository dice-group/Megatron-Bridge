#!/usr/bin/env python3
"""
Interactive text generation from a trained Nemotron 3 Super Megatron checkpoint.

Reuses the same recipe config as pretrain_nemotron_3_super.py so the model
architecture matches exactly (including overrides like num_layers, hybrid_override_pattern, etc.).

Usage:
  torchrun --nproc-per-node=1 examples/models/nemotron_3/generate_nemotron_3_super.py \
      checkpoint.load=/path/to/checkpoints \
      model.num_layers=7 \
      model.hybrid_override_pattern="MEME*ME" \
      model.num_moe_experts=8 \
      model.tensor_model_parallel_size=1 \
      model.expert_model_parallel_size=1 \
      model.sequence_parallel=False \
      model.context_parallel_size=1 \
      model.seq_length=2048 \
      dataset.sequence_length=2048

The script loads the checkpoint, then enters an interactive prompt loop.
Type 'quit' or 'exit' to stop.
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import torch
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from megatron.bridge.recipes.nemotronh.nemotron_3_super import (
    nemotron_3_super_pretrain_config as pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer, runtime_config_update
from megatron.bridge.training.setup import setup, initialize_megatron, _validate_and_set_vocab_size
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.checkpointing import load_checkpoint
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0

logger = logging.getLogger(__name__)


class SingleBatchIterator:
    """Yields a single batch then stops. Required by forward_backward_func."""

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


def forward_step(data_iterator, model, **kwargs):
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def generate(model, hf_tokenizer, prompt, max_new_tokens=100):
    """Greedy generation loop."""
    input_ids = hf_tokenizer.encode(prompt, return_tensors="pt").cuda()
    generated_ids = input_ids.clone()

    stop_tokens = [hf_tokenizer.eos_token_id]

    for step in range(max_new_tokens):
        position_ids = (
            torch.arange(generated_ids.size(1), dtype=torch.long, device=generated_ids.device)
            .unsqueeze(0)
            .expand_as(generated_ids)
        )

        with torch.no_grad():
            fwd_bwd_function = get_forward_backward_func()
            iterator = SingleBatchIterator(generated_ids, position_ids)

            output = fwd_bwd_function(
                forward_step_func=forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=generated_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                if world_size > 1:
                    gathered = [torch.zeros_like(output) for _ in range(world_size)]
                    torch.distributed.all_gather(
                        gathered, output, group=parallel_state.get_tensor_model_parallel_group()
                    )
                    output = torch.cat(gathered, dim=2)

                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            if next_token_ids.item() in stop_tokens:
                break

    return hf_tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained Nemotron 3 Super checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Max tokens to generate per prompt")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive mode)")
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main():
    args, cli_overrides = parse_cli_args()

    # Build the same config as pretraining — architecture will match the checkpoint
    cfg: ConfigContainer = pretrain_config()

    # Disable WandB for inference
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)
    if cli_overrides:
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    # Sync dataset.sequence_length to match model.seq_length if overridden
    model_seq_length = OmegaConf.select(merged_omega_conf, "model.seq_length", default=None)
    if model_seq_length is not None:
        OmegaConf.update(merged_omega_conf, "dataset.sequence_length", model_seq_length)

    # Force inference-friendly settings
    merged_omega_conf = OmegaConf.merge(
        merged_omega_conf,
        OmegaConf.create(
            {
                "logger": {"wandb_project": None, "wandb_entity": None},
                "train": {"train_iters": 0, "eval_iters": 0, "micro_batch_size": 1},
            }
        ),
    )

    final_overrides = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides, excluded_fields)

    # Apply runtime config (computes data_parallel_size, resolves precision, etc.)
    runtime_config_update(cfg)

    # Initialize distributed and model parallel
    state = GlobalState()
    state.cfg = cfg
    initialize_megatron(cfg=cfg)

    # Build tokenizer and set vocab_size (required before model creation)
    tokenizer = build_tokenizer(cfg.tokenizer)
    cfg.model.vocab_size, cfg.model.should_pad_vocab = _validate_and_set_vocab_size(
        model_vocab_size=cfg.model.vocab_size,
        tokenizer_vocab_size=tokenizer.vocab_size,
    )

    # Disable MTP before building the model so MTP layers are not created
    cfg.model.mtp_num_layers = 0

    # Build model
    model = cfg.model.provide_distributed_model(wrap_with_ddp=False)

    # Print model architecture
    print_rank_0(model[0])

    # Load checkpoint
    print_rank_0(f"Loading checkpoint from: {cfg.checkpoint.load}")
    load_checkpoint(state, model, None, None)

    # Disable MTP after checkpoint load — the checkpoint restores the training config
    # (mtp_num_layers > 0) but we don't need MTP for generation.
    for m in model:
        inner = m.module if hasattr(m, 'module') else m
        if hasattr(inner, 'config'):
            inner.config.mtp_num_layers = None
        if hasattr(inner, 'mtp_process'):
            inner.mtp_process = False

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()

    # Get HF tokenizer for encode/decode (the megatron tokenizer wraps it)
    hf_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.tokenizer_model, trust_remote_code=True)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    # Generation loop
    if args.prompt:
        # Single prompt mode
        result = generate(model, hf_tokenizer, args.prompt, args.max_new_tokens)
        print_rank_0(f"\n{'='*60}")
        print_rank_0(f"Prompt: {args.prompt}")
        print_rank_0(f"Generated: {result}")
        print_rank_0(f"{'='*60}")
    else:
        # Interactive mode
        print_rank_0("\n=== Nemotron 3 Super Interactive Generation ===")
        print_rank_0(f"Max new tokens: {args.max_new_tokens}")
        print_rank_0("Type 'quit' or 'exit' to stop.\n")

        while True:
            if parallel_state.is_pipeline_last_stage():
                try:
                    prompt = input(">>> ")
                except EOFError:
                    break
                if prompt.strip().lower() in ("quit", "exit", ""):
                    break
                # Broadcast prompt length then prompt to all ranks
                prompt_tensor = torch.tensor(
                    [ord(c) for c in prompt], dtype=torch.long, device="cuda"
                )
                length_tensor = torch.tensor([len(prompt)], dtype=torch.long, device="cuda")
            else:
                length_tensor = torch.tensor([0], dtype=torch.long, device="cuda")
                prompt_tensor = None
                prompt = None

            torch.distributed.broadcast(length_tensor, get_last_rank())
            if length_tensor.item() == 0:
                break

            if prompt_tensor is None:
                prompt_tensor = torch.zeros(length_tensor.item(), dtype=torch.long, device="cuda")
            torch.distributed.broadcast(prompt_tensor, get_last_rank())

            if prompt is None:
                prompt = "".join(chr(c) for c in prompt_tensor.tolist())

            result = generate(model, hf_tokenizer, prompt, args.max_new_tokens)
            print_rank_0(f"\n{result}\n")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
