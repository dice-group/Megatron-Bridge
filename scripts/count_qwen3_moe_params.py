#!/usr/bin/env python3
"""Count parameters for Qwen3 MoE and search for ~1B scaled config."""

V = 151936  # Qwen3 vocab size


def count_params(h, n_layers, n_heads, n_qg, ffn_h, moe_ffn, n_exp, topk):
    """
    Qwen3 MoE architecture (pure transformer, no Mamba):
    - GQA attention with QK layernorm
    - Gated SiLU FFN (gate_proj + up_proj + down_proj) per routed expert
    - Shared expert with same gated SiLU structure (using ffn_h)
    - Router: h -> n_exp

    Args:
        h:        hidden_size
        n_layers: number of transformer layers (all MoE)
        n_heads:  num_attention_heads
        n_qg:     num_query_groups (GQA)
        ffn_h:    shared expert intermediate size (= ffn_hidden_size)
        moe_ffn:  per-expert intermediate size (= moe_ffn_hidden_size)
        n_exp:    number of MoE experts
        topk:     router top-k (for active param count)
    """
    kv_ch = h // n_heads  # kv_channels

    # Embeddings (input + output, not shared in Qwen3 MoE)
    emb = 2 * V * h

    # Per-layer attention (GQA with QK layernorm)
    q_proj = h * h                    # h -> n_heads * kv_ch
    k_proj = h * n_qg * kv_ch        # h -> n_qg * kv_ch
    v_proj = h * n_qg * kv_ch        # h -> n_qg * kv_ch
    o_proj = h * h                    # n_heads * kv_ch -> h
    qk_norm = 2 * kv_ch              # QK layernorm (q_norm + k_norm)
    attn = q_proj + k_proj + v_proj + o_proj + qk_norm

    # Per-layer shared expert (gated SiLU: gate + up + down)
    shared_gate = h * ffn_h
    shared_up = h * ffn_h
    shared_down = ffn_h * h
    shared = shared_gate + shared_up + shared_down

    # Per-expert routed FFN (gated SiLU: gate + up + down)
    exp_gate = h * moe_ffn
    exp_up = h * moe_ffn
    exp_down = moe_ffn * h
    ep = exp_gate + exp_up + exp_down  # per expert

    # Router
    router = h * n_exp

    # Layer norms (input_layernorm + post_attention_layernorm)
    norms = 2 * h

    # Per-layer total
    layer_total = attn + shared + n_exp * ep + router + norms

    # Final layernorm
    final_norm = h

    total = emb + n_layers * layer_total + final_norm

    # Active params (only topk experts active per token)
    active_layer = attn + shared + topk * ep + router + norms
    active = emb + n_layers * active_layer + final_norm

    return total, active, attn, shared, ep, router, emb


# ============================================================
# Verify 30B-A3B
# ============================================================
print("=== Qwen3 MoE 30B-A3B (verification) ===")
t, act, attn, shared, ep, router, emb = count_params(
    h=2048, n_layers=48, n_heads=32, n_qg=4,
    ffn_h=6144, moe_ffn=768, n_exp=128, topk=8)

print(f"Total:  {t/1e9:.2f}B")
print(f"Active: {act/1e9:.2f}B (topk=8)")
print(f"  Embed+output:  {emb/1e9:.3f}B")
print(f"  Attention/layer: {attn/1e6:.1f}M")
print(f"  Shared exp/layer: {shared/1e6:.1f}M")
print(f"  Per expert:    {ep/1e6:.3f}M")
print(f"  128 experts:   {128*ep/1e6:.1f}M per layer")
print(f"  Router/layer:  {router/1e3:.0f}K")

# ============================================================
# Search for ~1B config
# ============================================================
print(f"\n{'='*60}")
print("=== Searching for ~1B Qwen3 MoE configs ===")
print(f"{'='*60}")

configs = []
# Constrain: kv_channels=64, keep 30B ratios (ffn=3*h, moe_ffn~0.375*h), 128 experts
KV_CH = 64
for n_heads in range(8, 33, 2):
    h = n_heads * KV_CH
    for n_qg in [2, 4, 8]:
        if n_heads % n_qg != 0:
            continue
        # Keep 30B ratio: ffn = 3*h, moe_ffn = 0.375*h
        ffn_h = 3 * h
        for moe_ffn in range(max(64, round(h * 0.25 / 64) * 64),
                             round(h * 0.5 / 64) * 64 + 1, 64):
            for n_exp in [64, 128]:
                for n_layers in range(8, 32):
                    t, act, _, _, _, _, _ = count_params(
                        h, n_layers, n_heads, n_qg,
                        ffn_h, moe_ffn, n_exp, topk=8)
                    if 0.90e9 <= t <= 1.10e9:
                        configs.append((
                            abs(t - 1e9), t, act, h, n_layers,
                            n_heads, n_qg, ffn_h, moe_ffn, n_exp
                        ))

configs.sort()
print(f"Found {len(configs)} configs in [0.90B, 1.10B]")

# Deduplicate by (h, n_layers, n_exp) and show top results
seen = set()
shown = 0
for c in configs:
    key = (c[3], c[4], c[9])  # h, n_layers, n_exp
    if key in seen:
        continue
    seen.add(key)
    shown += 1
    if shown > 25:
        break
    _, t, act, h, nl, nh, nqg, ffn, mffn, nexp = c
    print(f"  {t/1e9:.3f}B (active={act/1e9:.3f}B) | "
          f"H={h} L={nl} heads={nh} qg={nqg} "
          f"shared_ffn={ffn} moe_ffn={mffn} exp={nexp}")

# ============================================================
# Best config detailed breakdown
# ============================================================
if configs:
    print(f"\n{'='*60}")
    print("=== Best config breakdown ===")
    print(f"{'='*60}")
    _, t, act, h, nl, nh, nqg, ffn, mffn, nexp = configs[0]
    t, act, attn, shared, ep, router, emb = count_params(
        h, nl, nh, nqg, ffn, mffn, nexp, topk=8)
    kv_ch = h // nh
    print(f"Total:  {t/1e9:.3f}B")
    print(f"Active: {act/1e9:.3f}B (topk=8)")
    print(f"  Embed+output:    {emb/1e6:.1f}M ({100*emb/t:.1f}%)")
    print(f"  Attention ({nl}x): {nl*attn/1e6:.1f}M ({100*nl*attn/t:.1f}%)")
    print(f"  Shared exp ({nl}x): {nl*shared/1e6:.1f}M ({100*nl*shared/t:.1f}%)")
    print(f"  Routed experts ({nl}x{nexp}): {nl*nexp*ep/1e6:.1f}M ({100*nl*nexp*ep/t:.1f}%)")
    print(f"  Per expert:      {ep/1e3:.1f}K")
    print(f"  Router ({nl}x):   {nl*router/1e3:.0f}K")
    print()
    print(f"  Config for SLURM script:")
    print(f"    num_layers={nl}")
    print(f"    hidden_size={h}")
    print(f"    num_attention_heads={nh}")
    print(f"    num_query_groups={nqg}")
    print(f"    ffn_hidden_size={ffn}")
    print(f"    moe_ffn_hidden_size={mffn}")
    print(f"    num_moe_experts={nexp}")
    print(f"    moe_router_topk=8")
    print(f"    kv_channels={kv_ch}")
