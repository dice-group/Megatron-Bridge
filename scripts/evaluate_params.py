def filter_brute():
    h = 768
    nl = 12
    nh = 12
    nqg = 4
    ffn = 2304
    mffn = 192

    kv_ch = h // nh
    q_proj = h * h
    k_proj = h * nqg * kv_ch
    v_proj = h * nqg * kv_ch
    o_proj = h * h
    qk_norm = 2 * kv_ch
    attn = q_proj + k_proj + v_proj + o_proj + qk_norm

    shared = h * ffn * 3
    ep = h * mffn * 3
    router = h * 128
    norms = 2 * h
    
    target = 442193664
    
    for vocab in range(150000, 155000):
        if vocab % 64 != 0: continue
        for has_shared in [True, False]:
            for n_expert in [32, 128]:
                for tie in [True, False]:
                    for qk_norm_on in [True, False]:
                        layer = q_proj + k_proj + v_proj + o_proj + norms
                        if qk_norm_on:
                            layer += qk_norm
                        if has_shared:
                            layer += shared
                        layer += n_expert * ep + router
                        
                        total = h + layer * nl
                        total += vocab * h
                        if not tie:
                            total += vocab * h
                            
                        if total == target:
                            print(f"FOUND! Vocab={vocab}, Shared={has_shared}, experts on rank={n_expert}, tied={tie}, qk_norm={qk_norm_on}")

filter_brute()
