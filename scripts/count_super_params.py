#!/usr/bin/env python3
"""Count parameters for Nemotron-3 Super."""

V = 131072

def count_params(h, n_m, n_e, n_a, n_exp, moe_ffn, shared_int, lat,
                 m_heads, m_hd, m_sd, m_groups,
                 a_heads, a_qg, a_kv):
    """
    LatentMoE architecture:
    - Shared down-proj:  h -> lat  (fc1_latent_proj)
    - Per-expert fc1:    lat -> moe_ffn
    - Per-expert fc2:    moe_ffn -> lat
    - Shared up-proj:    lat -> h  (fc2_latent_proj)
    - Shared expert:     h -> shared_int -> h  (NOT through latent)
    - Router:            h -> lat -> n_exp (via latent projection)

    If lat=0, experts operate in full h space (standard MoE).
    """
    emb = 2 * V * h

    # Mamba-2 layer
    di = m_heads * m_hd
    dsp = m_groups * m_sd
    ip = h * (2*di + 2*dsp + m_heads)
    op = di * h
    cv = (di + 2*dsp) * 5  # conv1d k=4 + bias
    ms = 3 * m_heads + h   # A, D, dt_bias + norm
    ml = ip + op + cv + ms

    # Attention layer (GQA)
    qkv = h * (a_heads + 2*a_qg) * a_kv
    ao = a_heads * a_kv * h
    al = qkv + ao + h  # + norm

    # MoE layer (LatentMoE)
    if lat > 0:
        # Latent projections (shared per layer, not per expert)
        latent_down = h * lat   # fc1_latent_proj: h -> lat
        latent_up = lat * h     # fc2_latent_proj: lat -> h
        # Per expert: operates in latent space
        exp_fc1 = lat * moe_ffn   # lat -> moe_ffn
        exp_fc2 = moe_ffn * lat   # moe_ffn -> lat
        ep = exp_fc1 + exp_fc2
        # Router projects through latent: h -> lat (part of fc1_latent_proj) then lat -> n_exp
        rp = lat * n_exp + n_exp  # router weight + bias
    else:
        latent_down = 0
        latent_up = 0
        ep = 2 * h * moe_ffn
        rp = h * n_exp + n_exp

    # Shared expert: runs in FULL h space (NOT through latent)
    sp = 2 * h * shared_int

    # MoE layer total
    el = latent_down + latent_up + n_exp * ep + sp + rp + h  # +norm

    t = emb + h + n_m*ml + n_a*al + n_e*el
    return t, ml, al, el, emb, ep, latent_down + latent_up

# ============================================================
# Full model verification
# ============================================================
print("=== Full Nemotron-3 Super (LatentMoE corrected) ===")
t, ml, al, el, emb, ep, lat_proj = count_params(
    h=4096, n_m=40, n_e=40, n_a=8, n_exp=512,
    moe_ffn=2688, shared_int=5376, lat=1024,
    m_heads=128, m_hd=64, m_sd=128, m_groups=8,
    a_heads=32, a_qg=2, a_kv=128)

print(f"Total: {t/1e9:.2f}B")
print(f"  Embed+output:   {emb/1e9:.3f}B")
print(f"  Mamba (40x):    {40*ml/1e9:.3f}B  ({ml/1e6:.1f}M each)")
print(f"  Attn (8x):      {8*al/1e9:.3f}B  ({al/1e6:.1f}M each)")
print(f"  MoE (40x):      {40*el/1e9:.3f}B  ({el/1e9:.3f}B each)")
print(f"    Latent projs: {lat_proj/1e6:.1f}M per layer")
print(f"    Per expert:   {ep/1e6:.3f}M")
print(f"    512 experts:  {512*ep/1e9:.3f}B per layer")

# Active count (topk=22)
topk = 22
act_exp = topk * ep
lat_proj_per_layer = 2 * 4096 * 1024  # latent down + up
shared_p = 2 * 4096 * 5376
router_p = 1024 * 512 + 512
active_moe_l = lat_proj_per_layer + act_exp + shared_p + router_p + 4096
active = emb + 4096 + 40*ml + 8*al + 40*active_moe_l
print(f"\nActive: {active/1e9:.2f}B (topk={topk})")

# ============================================================
# Search for ~1B config
# ============================================================
print(f"\n{'='*60}")
print("=== Searching for ~1B configs ===")
print(f"{'='*60}")

configs = []
for h in range(256, 1281, 64):
    for nl in range(6, 30):
        na = max(1, round(nl * 8/88))
        ne = round(nl * 40/88)
        nm = nl - ne - na
        if nm < 1 or ne < 1:
            continue

        r = h / 4096
        for n_exp in [4, 8, 16, 32, 64, 128]:
            mf = max(64, round(2688 * r / 64) * 64)
            si = 2 * mf
            lt = max(64, round(1024 * r / 64) * 64)

            mh = max(4, round(128 * r / 4) * 4)
            mhd = 64
            msd = max(16, round(128 * r / 16) * 16)
            mg = max(1, round(8 * r))

            ah = max(2, round(32 * r / 2) * 2)
            aqg = max(1, min(ah, round(2 * r)))
            akv = h // ah if h % ah == 0 and (h // ah) >= 32 else 128

            t, _, _, _, _, _, _ = count_params(h, nm, ne, na, n_exp, mf, si, lt,
                                                mh, mhd, msd, mg, ah, aqg, akv)
            if 0.85e9 <= t <= 1.15e9:
                configs.append((abs(t-1e9), t, h, nl, nm, ne, na, n_exp, mf, si, lt,
                               mh, mhd, msd, mg, ah, aqg, akv))

configs.sort()
print(f"Found {len(configs)} configs in [0.85B, 1.15B]")
for i, c in enumerate(configs[:25]):
    _, t, h, nl, nm, ne, na, ne2, mf, si, lt, mh, mhd, msd, mg, ah, aqg, akv = c
    # Compute active params
    ep_per = 2 * lt * mf  # latent-space expert
    scaled_topk = max(1, round(22 * ne2 / 512))
    act_e = scaled_topk * ep_per + 2*h*si + 2*h*lt + lt*ne2 + ne2 + h
    act = 2*V*h + h + nm*c[14] + na*c[15] + ne*act_e if False else 0
    print(f"  #{i+1}: {t/1e9:.3f}B | H={h} L={nl}({nm}M+{ne}E+{na}A) exp={ne2} moe_ffn={mf} shared={si} lat={lt}")

# ============================================================
# Best config detailed breakdown
# ============================================================
if configs:
    print(f"\n{'='*60}")
    print("=== Best config breakdown ===")
    print(f"{'='*60}")
    _, tt, h, nl, nm, ne, na, ne2, mf, si, lt, mh, mhd, msd, mg, ah, aqg, akv = configs[0]
    t, ml, al, el, emb, ep, lp = count_params(h, nm, ne, na, ne2, mf, si, lt,
                                               mh, mhd, msd, mg, ah, aqg, akv)
    print(f"Total: {t/1e9:.3f}B")
    print(f"  Embed+output: {emb/1e6:.1f}M ({100*emb/t:.1f}%)")
    print(f"  Mamba ({nm}x): {nm*ml/1e6:.1f}M ({100*nm*ml/t:.1f}%)")
    print(f"  Attn ({na}x): {na*al/1e6:.1f}M ({100*na*al/t:.1f}%)")
    print(f"  MoE ({ne}x): {ne*el/1e6:.1f}M ({100*ne*el/t:.1f}%)")
    print(f"  Per expert: {ep/1e3:.1f}K")
    print(f"  Latent proj/layer: {lp/1e6:.1f}M")

    # Generate pattern
    print(f"\n  Suggested pattern:")
    # Build a pattern with proper interleaving
    total = nm + ne + na
    attn_interval = total // (na + 1)
    layer_list = []
    mi, ei, ai = 0, 0, 0
    for pos in range(total):
        # Insert attention at regular intervals
        if ai < na and (pos+1) % (attn_interval+1) == 0:
            layer_list.append('*')
            ai += 1
        elif mi <= ei and mi < nm:
            layer_list.append('M')
            mi += 1
        elif ei < ne:
            layer_list.append('E')
            ei += 1
        elif mi < nm:
            layer_list.append('M')
            mi += 1
    while mi < nm:
        layer_list.append('M')
        mi += 1
    while ei < ne:
        layer_list.append('E')
        ei += 1
    pat = ''.join(layer_list)
    print(f"  Pattern: {pat}")
    print(f"  Check: {pat.count('M')}M + {pat.count('E')}E + {pat.count('*')}A = {len(pat)}")
