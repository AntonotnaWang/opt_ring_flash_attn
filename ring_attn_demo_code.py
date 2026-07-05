"""
Single-process Ring Attention demo.

The code below simulates `num_of_gpus` sequence-parallel ranks in one process:
each rank owns one local Q/K/V sequence shard. It verifies the simulated ring
attention forward and manual backward against normal full-sequence attention.
"""

import torch
import tqdm


def apply_causal_mask(scores, q_start, k_start, causal):
    """Apply causal mask to a local score block if `causal=True`."""
    if not causal:
        return scores

    local_q_len, local_k_len = scores.shape[-2:]
    q_pos = torch.arange(q_start, q_start + local_q_len, device=scores.device)
    k_pos = torch.arange(k_start, k_start + local_k_len, device=scores.device)
    mask = k_pos.view(1, 1, 1, -1) > q_pos.view(1, 1, -1, 1)
    return scores.masked_fill(mask, -float("inf"))


def safe_max(max_value):
    """Use 0 for -inf max values so fully masked blocks produce exp(-inf)=0."""
    return torch.where(torch.isfinite(max_value), max_value, torch.zeros_like(max_value))


def check_close(name, actual, expected, atol=1e-6):
    """Check equality and report max absolute error instead of error sum."""
    max_abs_diff = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, atol=atol), (
        f"{name} mismatch, max_abs_diff={max_abs_diff:.6e}"
    )
    return max_abs_diff


def run_validation(causal=False):
    # Hyperparameters for the toy problem.
    batch_size = 2
    total_seq_len = 256
    num_heads = 4
    head_dim = 64
    num_of_gpus = 8
    local_seq_len = total_seq_len // num_of_gpus
    assert total_seq_len % num_of_gpus == 0

    mode = "causal" if causal else "dense"
    print(f"\n=== {mode} ring attention validation ===")

    # qkv_total shape:
    #   (batch_size, total_seq_len, 3 * num_heads, head_dim)
    # dim=2 stores [all Q heads, all K heads, all V heads].
    qkv_total = torch.randn(
        (batch_size, total_seq_len, num_heads * 3, head_dim),
        requires_grad=True,
    )
    qkv_ref = qkv_total.detach().clone().requires_grad_()

    # doutput is upstream dL/dO, shape:
    #   (batch_size, total_seq_len, num_heads, head_dim)
    doutput = torch.randn(batch_size, total_seq_len, num_heads, head_dim)

    # qkv_total.chunk(num_of_gpus, dim=1) splits sequence length:
    #   item shape: (batch_size, local_seq_len, 3 * num_heads, head_dim)
    # item.chunk(3, dim=2) splits Q/K/V heads:
    #   qkv_chunks[i][0]: local Q_i, shape (batch_size, local_seq_len, num_heads, head_dim)
    #   qkv_chunks[i][1]: local K_i, shape (batch_size, local_seq_len, num_heads, head_dim)
    #   qkv_chunks[i][2]: local V_i, shape (batch_size, local_seq_len, num_heads, head_dim)
    qkv_chunks = [item.chunk(3, dim=2) for item in qkv_total.chunk(num_of_gpus, dim=1)]

    # =========================
    # Ring Attention forward
    # =========================
    # out_chunks[i]: local output O_i, shape (batch, local_seq, heads, head_dim)
    # max_chunks[i]: final row max for Q_i over all K shards, shape (batch, local_seq, heads, 1)
    # l_chunks[i]: final softmax denominator for Q_i, shape (batch, local_seq, heads, 1)
    out_chunks, max_chunks, l_chunks = [], [], []

    for i in tqdm.tqdm(range(num_of_gpus), desc=f"{mode} forward on each gpu"):
        # Q_i is fixed on rank i while K_j/V_j are visited one shard at a time.
        qi = qkv_chunks[i][0].transpose(1, 2)
        # local Q_i shape change:
        #   before transpose: (batch_size, local_seq_len, num_heads, head_dim)
        #   after transpose:  (batch_size, num_heads, local_seq_len, head_dim)
        q_start = i * local_seq_len

        # oi is the streaming softmax numerator:
        #   sum_j exp(score_ij - global_row_max_i) @ V_j
        # mi is the streaming row max, li is the streaming denominator.
        # oi shape: (batch_size, num_heads, local_seq_len, head_dim)
        # mi/li shape: (batch_size, num_heads, local_seq_len, 1)
        oi = torch.zeros_like(qi)
        mi = qi.new_full((batch_size, num_heads, local_seq_len, 1), -float("inf"))
        li = torch.zeros_like(mi)

        for j in range(num_of_gpus):
            # local K_j/V_j shape change:
            #   before transpose: (batch_size, local_seq_len, num_heads, head_dim)
            #   after transpose:  (batch_size, num_heads, local_seq_len, head_dim)
            kj = qkv_chunks[j][1].transpose(1, 2)
            vj = qkv_chunks[j][2].transpose(1, 2)
            k_start = j * local_seq_len

            # scores_ij is the local attention block Q_i @ K_j^T.
            # qi shape:                   (batch_size, num_heads, local_seq_len, head_dim)
            # kj.transpose(-2, -1) shape: (batch_size, num_heads, head_dim, local_seq_len)
            # shape: (batch_size, num_heads, local_seq_len, local_seq_len)
            scores_ij = (qi @ kj.transpose(-2, -1)) / (head_dim ** 0.5)
            scores_ij = apply_causal_mask(scores_ij, q_start, k_start, causal)

            # m_ij is the block-local row max. In causal mode an entire block can
            # be masked; safe_max keeps the exp calculation finite in that case.
            # m_ij shape:          (batch_size, num_heads, local_seq_len, 1)
            # exp_scores_ij shape: (batch_size, num_heads, local_seq_len, local_seq_len)
            m_ij = torch.max(scores_ij, dim=-1, keepdim=True).values
            exp_scores_ij = torch.exp(scores_ij - safe_max(m_ij))

            # Merge this block into the running softmax state. If m_ij is larger
            # than the old mi, the previous numerator/denominator are rescaled.
            # exp_scores_ij @ vj shape:
            #   (batch_size, num_heads, local_seq_len, head_dim)
            # oi/li/mi keep their original streaming-state shapes.
            mi_new = torch.maximum(mi, m_ij)
            oi = (
                oi * torch.exp(safe_max(mi) - safe_max(mi_new))
                + (exp_scores_ij @ vj) * torch.exp(safe_max(m_ij) - safe_max(mi_new))
            )
            li = (
                li * torch.exp(safe_max(mi) - safe_max(mi_new))
                + exp_scores_ij.sum(dim=-1, keepdim=True)
                * torch.exp(safe_max(m_ij) - safe_max(mi_new))
            )
            mi = mi_new

        # (oi / li) shape before transpose:
        #   (batch_size, num_heads, local_seq_len, head_dim)
        # output shard shape after transpose:
        #   (batch_size, local_seq_len, num_heads, head_dim)
        out_chunks.append((oi / li).transpose(1, 2))
        # mi/li shape before transpose: (batch_size, num_heads, local_seq_len, 1)
        # saved max/sum shape:          (batch_size, local_seq_len, num_heads, 1)
        max_chunks.append(mi.transpose(1, 2))
        l_chunks.append(li.transpose(1, 2))

    # Concatenate all local output shards along sequence dimension:
    #   out_ring shape: (batch_size, total_seq_len, num_heads, head_dim)
    out_ring = torch.cat(out_chunks, dim=1)

    # =========================
    # Ring Attention backward
    # =========================
    # Manual gradient accumulation does not need autograd graph construction.
    with torch.no_grad():
        # doutput_chunks[j] shape before transpose:
        #   (batch_size, local_seq_len, num_heads, head_dim)
        doutput_chunks = doutput.chunk(num_of_gpus, dim=1)
        # dq_chunks[j] accumulates local dQ_j in attention-matmul layout:
        #   (batch_size, num_heads, local_seq_len, head_dim)
        dq_chunks = [torch.zeros_like(qkv_chunks[i][0].transpose(1, 2)) for i in range(num_of_gpus)]
        dk_chunks, dv_chunks = [], []

        for i in tqdm.tqdm(range(num_of_gpus), desc=f"{mode} backward on each gpu"):
            # Fix local K_i/V_i and accumulate dK_i/dV_i from all query shards Q_j.
            ki = qkv_chunks[i][1].transpose(1, 2)
            vi = qkv_chunks[i][2].transpose(1, 2)
            # local K_i/V_i shape change:
            #   before transpose: (batch_size, local_seq_len, num_heads, head_dim)
            #   after transpose:  (batch_size, num_heads, local_seq_len, head_dim)
            k_start = i * local_seq_len

            # dki/dvi shape: (batch_size, num_heads, local_seq_len, head_dim)
            dki = torch.zeros_like(ki)
            dvi = torch.zeros_like(vi)

            for j in range(num_of_gpus):
                qj = qkv_chunks[j][0].transpose(1, 2)
                dout_j = doutput_chunks[j].transpose(1, 2)
                out_j = out_chunks[j].transpose(1, 2)
                q_start = j * local_seq_len
                # qj/dout_j/out_j shape change:
                #   before transpose: (batch_size, local_seq_len, num_heads, head_dim)
                #   after transpose:  (batch_size, num_heads, local_seq_len, head_dim)

                # scores_ji shape:
                #   (batch_size, num_heads, local_seq_len(query_j), local_seq_len(key_i))
                scores_ji = (qj @ ki.transpose(-2, -1)) / (head_dim ** 0.5)
                scores_ji = apply_causal_mask(scores_ji, q_start, k_start, causal)

                # Reconstruct the attention probability block A_{j,i}.
                # shape: (batch_size, num_heads, local_seq_len, local_seq_len)
                aji = torch.exp(scores_ji - max_chunks[j].transpose(1, 2)) / l_chunks[j].transpose(1, 2)

                # dV_i += A_{j,i}^T @ dO_j
                # aji.transpose(-2, -1) shape: (batch_size, num_heads, local_seq_len, local_seq_len)
                # dout_j shape:                (batch_size, num_heads, local_seq_len, head_dim)
                # dvi shape:                   (batch_size, num_heads, local_seq_len, head_dim)
                dvi += aji.transpose(-2, -1) @ dout_j

                # Softmax backward:
                # dS = A * (dA - sum(dA * A)).
                # Since O_j = A_j @ V_all, sum(dA * A) = sum(dO_j * O_j).
                # daji/dsji shape:
                #   (batch_size, num_heads, local_seq_len, local_seq_len)
                daji = dout_j @ vi.transpose(-2, -1)
                dsji = aji * (daji - torch.sum(dout_j * out_j, dim=-1, keepdim=True))

                # dK_i += dS_{j,i}^T @ Q_j / sqrt(head_dim)
                # dQ_j += dS_{j,i} @ K_i / sqrt(head_dim)
                dki += (dsji.transpose(-2, -1) @ qj) / (head_dim ** 0.5)
                dq_chunks[j] += (dsji @ ki) / (head_dim ** 0.5)

            dk_chunks.append(dki)
            dv_chunks.append(dvi)

        # Concatenate local gradient shards along sequence dimension, then restore
        # model layout from (batch, heads, seq, head_dim) to
        # (batch, seq, heads, head_dim).
        dq_ring_manual = torch.cat(dq_chunks, dim=2).transpose(1, 2)
        dk_ring_manual = torch.cat(dk_chunks, dim=2).transpose(1, 2)
        dv_ring_manual = torch.cat(dv_chunks, dim=2).transpose(1, 2)

    # =========================
    # Full attention reference
    # =========================
    # qkv_ref.transpose(1, 2) shape:
    #   (batch_size, 3 * num_heads, total_seq_len, head_dim)
    # q/k/v shape after chunk:
    #   (batch_size, num_heads, total_seq_len, head_dim)
    q, k, v = qkv_ref.transpose(1, 2).chunk(3, dim=1)
    # scores/attn shape:
    #   (batch_size, num_heads, total_seq_len, total_seq_len)
    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    scores = apply_causal_mask(scores, 0, 0, causal)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    # out_ref shape:
    #   before transpose: (batch_size, num_heads, total_seq_len, head_dim)
    #   after transpose:  (batch_size, total_seq_len, num_heads, head_dim)
    out_ref = (attn @ v).transpose(1, 2)

    output_diff = check_close(f"{mode} output", out_ring, out_ref)
    print(f"{mode} output max_abs_diff = {output_diff:.6e}")

    # Compare autograd gradients from ring forward and normal full attention.
    out_ref.backward(doutput)
    out_ring.backward(doutput)

    ring_dq, ring_dk, ring_dv = qkv_total.grad.chunk(3, dim=2)
    ref_dq, ref_dk, ref_dv = qkv_ref.grad.chunk(3, dim=2)
    grad_diff = check_close(f"{mode} ring grad vs reference grad", qkv_total.grad, qkv_ref.grad)
    print(f"{mode} reference grad max_abs_diff = {grad_diff:.6e}")

    # Extra check: manual full-attention backward should match reference autograd.
    with torch.no_grad():
        # doutput_h shape:
        #   (batch_size, num_heads, total_seq_len, head_dim)
        doutput_h = doutput.transpose(1, 2)
        dv_full = (attn.transpose(-2, -1) @ doutput_h).transpose(1, 2)
        da = doutput_h @ v.transpose(-2, -1)
        ds = attn * (da - torch.sum(da * attn, dim=-1, keepdim=True))
        # dq_full/dk_full/dv_full shape:
        #   (batch_size, total_seq_len, num_heads, head_dim)
        dq_full = ((ds @ k) / (head_dim ** 0.5)).transpose(1, 2)
        dk_full = ((ds.transpose(-2, -1) @ q) / (head_dim ** 0.5)).transpose(1, 2)

    check_close(f"{mode} reference dQ", ref_dq, dq_full)
    check_close(f"{mode} reference dK", ref_dk, dk_full)
    check_close(f"{mode} reference dV", ref_dv, dv_full)

    # Final check: manually accumulated ring gradients should match ring autograd.
    check_close(f"{mode} ring manual dQ", ring_dq, dq_ring_manual)
    check_close(f"{mode} ring manual dK", ring_dk, dk_ring_manual)
    check_close(f"{mode} ring manual dV", ring_dv, dv_ring_manual)


if __name__ == "__main__":
    torch.manual_seed(0)
    run_validation(causal=False)
    run_validation(causal=True)
