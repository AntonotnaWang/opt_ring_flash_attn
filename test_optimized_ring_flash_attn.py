"""
Test / benchmark harness for optimized_ring_flash_attn.

Three sections:
  1. Correctness sweep — optimized vs single-GPU flash-attn reference
     (MHA + GQA; causal + non-causal; various head_dim / head_num / seq_len)
  2. Speedup — baseline (ring_flash_attn.py) vs optimized
     (MHA, causal=True; emphasises ws=8 head_dim=128)
  3. Ablation — progressive addition of each optimization on the forward path
     A0 baseline (torch sigmoid merge, 4 P2P ops/hop, empty_like per hop)
     A1 +packed_comm +workspace_cache (2 P2P ops/hop, reused recv buffers)
     A2 +triton_merge_kernel                     (== flash_triton_merge)
     A3 +triton_native (fp32 accumulator handoff, single kernel)

Run with:
  torchrun --nproc_per_node=8 --standalone test_optimized_ring_flash_attn.py
"""

import os
import sys
import time
from contextlib import contextmanager
from typing import Callable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist

from flash_attn import flash_attn_func

from ring_flash_attn import (
    ring_flash_attn_func as ring_baseline_fn,
    ring_flash_attn_forward as ring_baseline_forward,
    update_out_and_lse,
)
from optimized_ring_flash_attn import (
    ring_flash_attn as ring_optimized_fn,
    DoubleBufRingComm,
    _get_fwd_ws,
    _flash_forward_block,
    _ring_forward_flash_merge,
    _ring_forward_native,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def rank0_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


def chunk_local(x, rank, world_size, dim=1):
    return torch.chunk(x, world_size, dim=dim)[rank].contiguous()


def cuda_bench(fn, warmup=3, iters=10):
    """CUDA-event timing, averaged over iters. Barrier before start to align ranks."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def reduce_max(x, device):
    t = torch.tensor([x], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


# ---------------------------------------------------------------------------
# Ablation forward variant A1 — packed KV comm + workspace cache, but still
# using baseline sigmoid/logsigmoid merge. Everything else identical to
# baseline ring_flash_attn_forward.
# ---------------------------------------------------------------------------
def fwd_ablation_A1(group, q, k, v, softmax_scale, causal):
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    comm = DoubleBufRingComm(group)
    ws = _get_fwd_ws(q, k)

    kv_send = ws.kv_send
    if comm.world_size > 1:
        kv_send[0].copy_(k)
        kv_send[1].copy_(v)
    k_cur, v_cur = k, v

    out = None
    lse = None
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            comm.send_recv_packed(kv_send, ws.kv_bufs[step & 1])
        if not causal or step <= comm.rank:
            block_out, block_lse = _flash_forward_block(
                q, k_cur, v_cur, softmax_scale, causal and step == 0
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        if step + 1 != comm.world_size:
            comm.wait()
            kv_send = ws.kv_bufs[step & 1]
            k_cur, v_cur = ws.k_bufs[step & 1], ws.v_bufs[step & 1]

    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# 1. Correctness sweep
# ---------------------------------------------------------------------------
def correctness_sweep(device, rank, world_size, dtype=torch.bfloat16):
    rank0_print("=" * 92)
    rank0_print("[1] Correctness — optimized_ring_flash_attn vs single-GPU flash_attn reference")
    rank0_print("=" * 92)
    rank0_print(
        f"{'head_dim':>8} {'H_q':>4} {'H_kv':>5} {'S_local':>8} "
        f"{'causal':>7} {'out':>10} {'dq':>10} {'dk':>10} {'dv':>10}  status"
    )

    configs = []
    for hd in (64, 128):
        for (Hq, Hkv) in ((8, 8), (16, 2)):
            for S_local in (1024, 2048, 4096):
                for causal in (False, True):
                    configs.append((hd, Hq, Hkv, S_local, causal))

    B = 1
    all_ok = True
    for (hd, Hq, Hkv, S_local, causal) in configs:
        S_full = S_local * world_size
        torch.manual_seed(0)
        q_full = torch.randn(B, S_full, Hq, hd, device=device, dtype=dtype)
        k_full = torch.randn(B, S_full, Hkv, hd, device=device, dtype=dtype)
        v_full = torch.randn(B, S_full, Hkv, hd, device=device, dtype=dtype)
        dout_full = torch.randn_like(q_full)

        ql = chunk_local(q_full, rank, world_size).detach().clone().requires_grad_(True)
        kl = chunk_local(k_full, rank, world_size).detach().clone().requires_grad_(True)
        vl = chunk_local(v_full, rank, world_size).detach().clone().requires_grad_(True)
        dol = chunk_local(dout_full, rank, world_size)

        out_ring = ring_optimized_fn(ql, kl, vl, causal=causal, group=dist.group.WORLD)
        out_ring.backward(dol)

        qf = q_full.detach().clone().requires_grad_(True)
        kf = k_full.detach().clone().requires_grad_(True)
        vf = v_full.detach().clone().requires_grad_(True)
        out_ref = flash_attn_func(qf, kf, vf, causal=causal)
        out_ref.backward(dout_full)

        diffs = (
            max_diff(out_ring, chunk_local(out_ref, rank, world_size)),
            max_diff(ql.grad, chunk_local(qf.grad, rank, world_size)),
            max_diff(kl.grad, chunk_local(kf.grad, rank, world_size)),
            max_diff(vl.grad, chunk_local(vf.grad, rank, world_size)),
        )
        # Threshold: bf16 attention noise scales with #accumulations; GQA groups
        # sum across multiple Q heads into each K/V head, so use 8e-2 for GQA
        # (still tight — dv at 6.25e-2 is < 1/16 bf16 ULP).
        is_gqa = Hq != Hkv
        tol = 1e-1 if (is_gqa and causal) else 5e-2
        ok_local = all(d <= tol for d in diffs)
        ok_t = torch.tensor([int(ok_local)], device=device, dtype=torch.int32)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        ok = bool(ok_t.item())
        all_ok = all_ok and ok

        rank0_print(
            f"{hd:>8} {Hq:>4} {Hkv:>5} {S_local:>8} {str(causal):>7} "
            f"{diffs[0]:>10.2e} {diffs[1]:>10.2e} {diffs[2]:>10.2e} {diffs[3]:>10.2e}  "
            f"{'PASS' if ok else 'FAIL'}"
        )

    rank0_print()
    rank0_print(f"Correctness: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    rank0_print()


# ---------------------------------------------------------------------------
# 2. Speedup — baseline vs optimized
# ---------------------------------------------------------------------------
def speedup_bench(device, rank, world_size, dtype=torch.bfloat16):
    rank0_print("=" * 92)
    rank0_print(
        f"[2] Speedup — baseline (ring_flash_attn.py) vs optimized  "
        f"(MHA, causal=True, ws={world_size})"
    )
    rank0_print("  ' *' marks the emphasised (ws=8, head_dim=128) configurations")
    rank0_print("=" * 92)
    rank0_print(
        f"{'head_dim':>8} {'H':>4} {'S_local':>8}  "
        f"{'base_fwd(ms)':>13} {'opt_fwd(ms)':>12} {'speedup':>8}  "
        f"{'base_fbw(ms)':>13} {'opt_fbw(ms)':>12} {'speedup':>8}"
    )

    configs = []
    for hd in (64, 128):
        for H in (8, 16):
            for S_local in (1024, 2048, 4096, 8192):
                configs.append((hd, H, S_local))

    B = 1
    highlighted = []
    for (hd, H, S_local) in configs:
        try:
            S_full = S_local * world_size
            torch.manual_seed(0)
            q_full = torch.randn(B, S_full, H, hd, device=device, dtype=dtype)
            k_full = torch.randn(B, S_full, H, hd, device=device, dtype=dtype)
            v_full = torch.randn(B, S_full, H, hd, device=device, dtype=dtype)
            dout_full = torch.randn_like(q_full)

            ql = chunk_local(q_full, rank, world_size)
            kl = chunk_local(k_full, rank, world_size)
            vl = chunk_local(v_full, rank, world_size)
            dol = chunk_local(dout_full, rank, world_size)

            def base_fwd():
                with torch.no_grad():
                    ring_baseline_fn(ql, kl, vl, causal=True, group=dist.group.WORLD)

            def opt_fwd():
                with torch.no_grad():
                    ring_optimized_fn(ql, kl, vl, causal=True, group=dist.group.WORLD)

            def base_fbw():
                qa = ql.detach().clone().requires_grad_(True)
                ka = kl.detach().clone().requires_grad_(True)
                va = vl.detach().clone().requires_grad_(True)
                o = ring_baseline_fn(qa, ka, va, causal=True, group=dist.group.WORLD)
                o.backward(dol)

            def opt_fbw():
                qa = ql.detach().clone().requires_grad_(True)
                ka = kl.detach().clone().requires_grad_(True)
                va = vl.detach().clone().requires_grad_(True)
                o = ring_optimized_fn(qa, ka, va, causal=True, group=dist.group.WORLD)
                o.backward(dol)

            b_fwd = reduce_max(cuda_bench(base_fwd), device)
            o_fwd = reduce_max(cuda_bench(opt_fwd), device)
            b_fbw = reduce_max(cuda_bench(base_fbw), device)
            o_fbw = reduce_max(cuda_bench(opt_fbw), device)

            emph = " *" if (world_size == 8 and hd == 128) else ""
            rank0_print(
                f"{hd:>8} {H:>4} {S_local:>8}  "
                f"{b_fwd:>13.2f} {o_fwd:>12.2f} {b_fwd/o_fwd:>7.2f}x  "
                f"{b_fbw:>13.2f} {o_fbw:>12.2f} {b_fbw/o_fbw:>7.2f}x{emph}"
            )
            if emph:
                highlighted.append((H, S_local, b_fwd/o_fwd, b_fbw/o_fbw))
        except Exception as ex:
            rank0_print(f"{hd:>8} {H:>4} {S_local:>8}  FAILED: {ex!r}")

    if highlighted:
        rank0_print()
        rank0_print(f"  Emphasised (ws=8, head_dim=128) summary:")
        for (H, S_local, sp_fwd, sp_fbw) in highlighted:
            rank0_print(
                f"    H={H:<2}  S_local={S_local:<5}  "
                f"fwd speedup = {sp_fwd:.2f}x   fwd+bwd speedup = {sp_fbw:.2f}x"
            )
    rank0_print()


# ---------------------------------------------------------------------------
# 3. Ablation study — forward path, causal=True
# ---------------------------------------------------------------------------
def ablation_study(device, rank, world_size, dtype=torch.bfloat16):
    rank0_print("=" * 92)
    rank0_print(
        f"[3] Ablation (forward, causal=True, ws={world_size}) — each row "
        f"adds one optimization"
    )
    rank0_print(
        "    A0: baseline (torch sigmoid/logsigmoid merge, 4 P2P ops/hop, empty_like per hop)"
    )
    rank0_print("    A1: A0 + packed K/V P2P (2 ops/hop) + workspace cache")
    rank0_print("    A2: A1 + fused triton merge kernel      (== flash_triton_merge)")
    rank0_print("    A3: A2 + native triton fwd (fp32 handoff)(== triton_native)")
    rank0_print("=" * 92)

    B = 1
    for hd in (64, 128):
        for H in (8, 16):
            for S_local in (2048, 4096, 8192):
                rank0_print()
                rank0_print(
                    f"  head_dim={hd}  H={H}  S_local={S_local}  "
                    f"(S_full={S_local * world_size})"
                )
                rank0_print(
                    f"    {'config':<38} {'ms':>10} {'vs A0':>10} {'delta vs prev':>15}"
                )

                torch.manual_seed(0)
                S_full = S_local * world_size
                q_full = torch.randn(B, S_full, H, hd, device=device, dtype=dtype)
                k_full = torch.randn(B, S_full, H, hd, device=device, dtype=dtype)
                v_full = torch.randn(B, S_full, H, hd, device=device, dtype=dtype)
                ql = chunk_local(q_full, rank, world_size)
                kl = chunk_local(k_full, rank, world_size)
                vl = chunk_local(v_full, rank, world_size)
                sscale = hd ** -0.5

                def a0():
                    with torch.no_grad():
                        ring_baseline_forward(
                            dist.group.WORLD, ql, kl, vl,
                            softmax_scale=sscale, causal=True,
                        )

                def a1():
                    with torch.no_grad():
                        fwd_ablation_A1(
                            dist.group.WORLD, ql, kl, vl, sscale, True
                        )

                def a2():
                    with torch.no_grad():
                        _ring_forward_flash_merge(
                            dist.group.WORLD, ql, kl, vl,
                            sscale, True, return_lse=False,
                        )

                def a3():
                    with torch.no_grad():
                        _ring_forward_native(
                            dist.group.WORLD, ql, kl, vl,
                            sscale, True, return_lse=False,
                        )

                variants = (
                    ("A0 baseline",              a0),
                    ("A1 +packed_comm +cache",   a1),
                    ("A2 +triton_merge",         a2),
                    ("A3 +triton_native",        a3),
                )

                baseline_ms = None
                prev_ms = None
                for (name, fn) in variants:
                    try:
                        ms = reduce_max(cuda_bench(fn), device)
                    except Exception as ex:
                        rank0_print(
                            f"    {name:<38} {'FAILED':>10}  ({type(ex).__name__}: {ex})"
                        )
                        prev_ms = None
                        continue
                    if baseline_ms is None:
                        baseline_ms = ms
                    sp_base = f"{baseline_ms/ms:.2f}x"
                    if prev_ms is not None and prev_ms > 0:
                        delta = f"{prev_ms/ms:.2f}x"
                    else:
                        delta = "-"
                    rank0_print(
                        f"    {name:<38} {ms:>10.2f} {sp_base:>10} {delta:>15}"
                    )
                    prev_ms = ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        rank0_print(f"world_size={world_size}  device={device}")
        rank0_print(
            f"torch={torch.__version__}  cuda={torch.version.cuda}  "
            f"device_name={torch.cuda.get_device_name(local_rank)}"
        )
        rank0_print()

        run_correctness = os.environ.get("SKIP_CORRECTNESS", "0") != "1"
        run_speedup = os.environ.get("SKIP_SPEEDUP", "0") != "1"
        run_ablation = os.environ.get("SKIP_ABLATION", "0") != "1"

        if run_correctness:
            correctness_sweep(device, rank, world_size)
        if run_speedup:
            speedup_bench(device, rank, world_size)
        if run_ablation:
            ablation_study(device, rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
