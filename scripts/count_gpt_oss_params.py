#!/usr/bin/env python3
"""Count parameters for GPT OSS MoE and search for ~1B scaled config."""

V = 201088  # GPT OSS vocab size


def count_params(h, n_layers, n_heads, n_qg, kv_ch, moe_ffn, n_exp, topk, bias=True):
    """
    GPT OSS MoE architecture (pure transformer, all MoE layers):
    - GQA attention with explicit kv_channels (not h/n_heads)
    - Gated linear unit FFN (gate + up + down) per routed expert
    - No shared expert
    - Router: h -> n_exp
    - add_bias_linear=True

    Args:
        h:        hidden_size
        n_layers: number of transformer layers (all MoE)
        n_heads:  num_attention_heads
        n_qg:     num_query_groups (GQA)
        kv_ch:    kv_channels (explicit, independent of h/n_heads)
        moe_ffn:  per-expert intermediate size (= moe_ffn_hidden_size)
        n_exp:    number of MoE experts
        topk:     router top-k (for active param count)
        bias:     add_bias_linear (True for GPT OSS)
    """
    # Embeddings (input + output, not shared)
    emb = 2 * V * h

    # Per-layer attention (GQA with explicit kv_channels)
    q_dim = n_heads * kv_ch
    kv_dim = n_qg * kv_ch
    q_proj = h * q_dim + (q_dim if bias else 0)
    k_proj = h * kv_dim + (kv_dim if bias else 0)
    v_proj = h * kv_dim + (kv_dim if bias else 0)
    o_proj = q_dim * h + (h if bias else 0)
    attn = q_proj + k_proj + v_proj + o_proj

    # Per-expert routed FFN (gated linear unit: gate + up + down)
    exp_gate = h * moe_ffn + (moe_ffn if bias else 0)
    exp_up = h * moe_ffn + (moe_ffn if bias else 0)
    exp_down = moe_ffn * h + (h if bias else 0)
    ep = exp_gate + exp_up + exp_down  # per expert

    # Router
    router = h * n_exp + (n_exp if bias else 0)

    # Layer norms (input_layernorm + post_attention_layernorm)
    norms = 2 * h

    # Per-layer total
    layer_total = attn + n_exp * ep + router + norms

    # Final layernorm
    final_norm = h

    total = emb + n_layers * layer_total + final_norm

    # Active params (only topk experts active per token)
    active_layer = attn + topk * ep + router + norms
    active = emb + n_layers * active_layer + final_norm

    return total, active, attn, ep, router, emb


# ============================================================
# Verify 20B
# ============================================================
print("=== GPT OSS 20B (verification) ===")
t, act, attn, ep, router, emb = count_params(
    h=2880, n_layers=24, n_heads=64, n_qg=8, kv_ch=64,
    moe_ffn=2880, n_exp=32, topk=4)

print(f"Total:  {t/1e9:.2f}B")
print(f"Active: {act/1e9:.2f}B (topk=4)")
print(f"  Embed+output:    {emb/1e9:.3f}B")
print(f"  Attention/layer: {attn/1e6:.1f}M")
print(f"  Per expert:      {ep/1e6:.3f}M")
print(f"  32 experts:      {32*ep/1e6:.1f}M per layer")
print(f"  Router/layer:    {router/1e3:.0f}K")

# Verify 120B
print("\n=== GPT OSS 120B (verification) ===")
t120, act120, _, _, _, _ = count_params(
    h=2880, n_layers=36, n_heads=64, n_qg=8, kv_ch=64,
    moe_ffn=2880, n_exp=128, topk=4)
print(f"Total:  {t120/1e9:.2f}B")
print(f"Active: {act120/1e9:.2f}B (topk=4)")

# ============================================================
# Search for ~1B config
# ============================================================
print(f"\n{'='*60}")
print("=== Searching for ~1B GPT OSS MoE configs ===")
print(f"{'='*60}")

# 30B ratios to preserve:
#   kv_channels = 64 (fixed)
#   moe_ffn = h (same as hidden_size in 20B: 2880/2880)
#   n_qg = n_heads / 8
#   n_heads * kv_ch != h (decoupled in GPT OSS)

configs = []
for n_heads in range(8, 65, 4):
    for n_qg in [2, 4, 8]:
        if n_heads % n_qg != 0:
            continue
        for h in range(512, 2049, 64):
            kv_ch = 64
            # Scale moe_ffn proportionally: moe_ffn = h in the 20B
            for moe_ffn_ratio in [0.75, 1.0, 1.25]:
                moe_ffn = round(h * moe_ffn_ratio / 64) * 64
                if moe_ffn < 128:
                    continue
                for n_exp in [32, 64]:
                    for n_layers in range(8, 30):
                        t, act, _, _, _, _ = count_params(
                            h, n_layers, n_heads, n_qg, kv_ch,
                            moe_ffn, n_exp, topk=4)
                        if 0.90e9 <= t <= 1.10e9:
                            configs.append((
                                abs(t - 1e9), t, act, h, n_layers,
                                n_heads, n_qg, kv_ch, moe_ffn, n_exp
                            ))

configs.sort()
print(f"Found {len(configs)} configs in [0.90B, 1.10B]")

# Deduplicate by (h, n_layers, n_exp) and show top results
seen = set()
shown = 0
for c in configs:
    key = (c[3], c[4], c[9])
    if key in seen:
        continue
    seen.add(key)
    shown += 1
    if shown > 25:
        break
    _, t, act, h, nl, nh, nqg, kvch, mffn, nexp = c
    print(f"  {t/1e9:.3f}B (active={act/1e9:.3f}B) | "
          f"H={h} L={nl} heads={nh} qg={nqg} kv_ch={kvch} "
          f"moe_ffn={mffn} exp={nexp}")

# ============================================================
# Best config detailed breakdown
# ============================================================
if configs:
    print(f"\n{'='*60}")
    print("=== Best config breakdown ===")
    print(f"{'='*60}")
    _, t, act, h, nl, nh, nqg, kvch, mffn, nexp = configs[0]
    t, act, attn, ep, router, emb = count_params(
        h, nl, nh, nqg, kvch, mffn, nexp, topk=4)
    print(f"Total:  {t/1e9:.3f}B")
    print(f"Active: {act/1e9:.3f}B (topk=4)")
    print(f"  Embed+output:    {emb/1e6:.1f}M ({100*emb/t:.1f}%)")
    print(f"  Attention ({nl}x): {nl*attn/1e6:.1f}M ({100*nl*attn/t:.1f}%)")
    print(f"  Routed experts ({nl}x{nexp}): {nl*nexp*ep/1e6:.1f}M ({100*nl*nexp*ep/t:.1f}%)")
    print(f"  Per expert:      {ep/1e6:.3f}M")
    print(f"  Router ({nl}x):   {nl*router/1e3:.0f}K")
    print()
    print(f"  Config for SLURM script:")
    print(f"    num_layers={nl}")
    print(f"    hidden_size={h}")
    print(f"    num_attention_heads={nh}")
    print(f"    num_query_groups={nqg}")
    print(f"    kv_channels={kvch}")
    print(f"    ffn_hidden_size={mffn}")
    print(f"    moe_ffn_hidden_size={mffn}")
    print(f"    num_moe_experts={nexp}")
    print(f"    moe_router_topk=4")
