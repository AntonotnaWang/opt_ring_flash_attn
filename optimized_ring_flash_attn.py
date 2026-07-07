"""
Optimized Ring Flash Attention
==============================

A drop-in replacement for `ring_flash_attn.ring_flash_attn_func` with the same
autograd semantics. Measured on H100 NVLink (bf16, max-over-ranks median vs the
zhuzilin baseline): forward ~1.1-1.4x faster at world_size 4 and ~1.2-1.9x at
world_size 8; forward+backward ~1.05-1.2x and ~1.1-1.35x respectively; near
parity at world_size 2 (single hop). Speedup grows with world_size (ring hop
count) and is largest for GQA / causal shapes.

`ring_flash_attn.ring_flash_attn_func` 的直接替换实现，autograd 语义完全一致。
H100 NVLink 实测（bf16，跨 rank 取 median 后取最大值，对比 zhuzilin baseline）：
world_size=4 时 forward 快约 1.1-1.4 倍、world_size=8 约 1.2-1.9 倍；forward+backward
分别约 1.05-1.2 倍与 1.1-1.35 倍；world_size=2（单 hop）基本持平。加速随 world_size
（ring hop 数）增大，GQA / causal 形状收益最大。

Concept refresher / 原理简述
----------------------------
Ring (sequence-parallel) attention shards the sequence dim across `world_size`
ranks. Each rank holds a Q, K, V slice of length S_local. K/V rotate around
the ring once — every step, each rank computes flash-attn on (its Q, current
K/V), merges the partial output via online softmax, then sends K/V forward
while receiving K/V from the previous rank.

Ring attention 把 sequence 维度切成 `world_size` 份分散在多张卡上，每张卡拿一段
长度 S_local 的 Q/K/V。计算时 K/V 沿着环形通信路径转一整圈：每步用本地 Q 和
当前 K/V 算 flash-attn，通过 online softmax 累加到局部输出；同时把当前 K/V
发给下一个 rank、从上一个 rank 收下一份 K/V。

  step 0:  rank r has (K_r, V_r).   Compute attn(Q_r, K_r, V_r), start ring hop.
  step 1:  rank r has (K_{r-1}, V_{r-1}). Merge second partial output.
  ...
  step W-1: after W-1 hops all K/V blocks have contributed.

Where the baseline leaves perf on the table (this file addresses each):
Baseline 实现的性能损失点（本文件逐一优化）:
  (a) bf16 → fp32 → bf16 round-trip at every ring step
      每一步都在 bf16 与 fp32 之间往返
  (b) `torch.empty_like(k)` allocation per hop
      每 hop 都新分配 K/V 接收缓冲
  (c) 4 P2P ops per hop (send/recv K + send/recv V)
      每 hop 4 个 P2P op（K、V 各 send/recv）
  (d) `sigmoid + logsigmoid` JIT'd merge (4 kernel launches per merge)
      merge 用 pytorch op（每次合并 4 个 kernel launch）
  (e) backward re-allocates the fp32 dk/dv accumulators every ring step
      (`dk = block_dk + next_dk`) instead of reusing a pool
      backward 每步都重新分配 dk/dv 的 fp32 累加器，而不是复用缓冲池
  (Baseline DOES support GQA via flash-attn; this file additionally adds
   GQA-aware native triton kernels. / baseline 本身经 flash-attn 支持 GQA，
   本文件额外提供了 GQA 感知的 native triton kernel。)

File layout / 文件结构
----------------------
   1.  Flash-attn interface adapters                / flash-attn 接口适配
   2.  Ring communicator (double-buffered, packed)  / 环形通信器（双缓冲 + K/V 打包）
   3.  Workspace caches (fwd & bwd tensor pools)    / 按 shape 缓存的 tensor 池
   4.  Triton kernels (merge, grad-accum, fwd/bwd)  / Triton kernel 实现
   5.  Ring forward implementations + dispatcher    / forward 分派
   6.  Ring backward implementations + dispatcher   / backward 分派
   7.  Zigzag ring forward/backward                 / zigzag 变体
   8.  Autograd Function + `ring_flash_attn` entry  / autograd 入口
   9.  __main__ smoke test / micro-benchmark        / 冒烟测试

Environment knobs / 环境变量
----------------------------
  OPTIMIZED_RING_FORWARD_IMPL   : 'auto'|'triton_native'|'flash_triton_merge'
                                  强制 forward 实现
  OPTIMIZED_RING_BACKWARD_IMPL  : 'auto'|'triton_native'|'flash'
                                  强制 backward 实现
  OPTIMIZED_RING_BWD_COMM_BF16  : '1' → bf16 dk/dv transport (default fp32)
                                  dk/dv 通信用 bf16（默认 fp32；速度换精度）
"""

import inspect
import os
from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward


# ===========================================================================
# 1.  Flash-attn interface adapters / flash-attn 接口适配
# ---------------------------------------------------------------------------
# flash-attn's C API signature drifts between minor releases (window_size
# packed vs left/right, softcap presence, rng_state). We introspect once and
# build a fixed kwarg template so per-call cost is zero.
#
# flash-attn 的 C API 在各小版本间会变（window_size 是否合并、有没有 softcap、
# 是否需要 rng_state），启动时反射一次得到固定的 kwarg 模板，每次调用零成本。
# ===========================================================================
def _sig(fn):
    fn = fn._init_fn if hasattr(fn, "_init_fn") else fn
    return inspect.getfullargspec(fn).args


_FWD_ARGS = _sig(_flash_attn_forward)
_BWD_ARGS = _sig(_flash_attn_backward)


def _flash_kwargs(*, forward: bool) -> dict:
    """Build the shape-independent kwargs (window_size / softcap / rng_state)."""
    kwargs = {"dropout_p": 0.0, "alibi_slopes": None}
    args = _FWD_ARGS if forward else _BWD_ARGS
    if "window_size" in args:
        kwargs["window_size"] = (-1, -1)
    else:
        kwargs["window_size_left"] = -1
        kwargs["window_size_right"] = -1
    if "softcap" in args:
        kwargs["softcap"] = 0.0
    if forward:
        kwargs["return_softmax"] = False
    else:
        kwargs["deterministic"] = False
        if "rng_state" in args:
            kwargs["rng_state"] = None
    return kwargs


_FWD_KWARGS_TMPL = _flash_kwargs(forward=True)
_BWD_KWARGS_TMPL = _flash_kwargs(forward=False)


def _flash_forward_block(q, k, v, softmax_scale, causal):
    """Run flash-attn forward on (q, k, v). Returns (block_out, block_lse)."""
    outputs = _flash_attn_forward(
        q=q, k=k, v=v,
        softmax_scale=softmax_scale, causal=causal, **_FWD_KWARGS_TMPL,
    )
    # flash-attn returns (out, lse, S_dmask, rng_state) or 8-tuple variant.
    return outputs[0], outputs[1] if len(outputs) == 4 else outputs[5]


def _flash_backward_block(dout, q, k, v, out, softmax_lse,
                          dq, dk, dv, softmax_scale, causal):
    """Run flash-attn backward. Writes dq/dk/dv into caller's buffers."""
    _flash_attn_backward(
        dout=dout, q=q, k=k, v=v, out=out, softmax_lse=softmax_lse,
        dq=dq, dk=dk, dv=dv,
        softmax_scale=softmax_scale, causal=causal, **_BWD_KWARGS_TMPL,
    )


# ===========================================================================
# 2.  Ring communicator (double-buffered, K/V packed)
# ---------------------------------------------------------------------------
# Baseline `RingComm` issues 4 P2P ops per hop. By stacking K and V into one
# `[2, B, S, H, D]` tensor we cut to 2 ops, saving Python + NCCL launch time.
# The caller supplies TWO receive buffers and alternates between them so the
# next hop can start filling one buffer while compute still uses the other.
#
# Baseline `RingComm` 每 hop 发 4 个 P2P op（K/V 各 send/recv）。把 K 和 V
# 拼成一个 `[2, B, S, H, D]` tensor，就能压到 2 个 op，省下 Python 与 NCCL
# 的 launch 成本。调用方提供两个接收 buffer 交替使用，实现真正的双缓冲：
# 下一 hop 往 buffer A 收数据时，本 hop 的计算还在读 buffer B。
# ===========================================================================
class DoubleBufRingComm:
    def __init__(self, process_group: Optional[dist.ProcessGroup]):
        self._pg = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        if process_group is not None:
            self.send_rank = dist.get_global_rank(process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(process_group, self.recv_rank)
        self._pending = []  # FIFO of outstanding Work batches

    def send_recv_packed(self, send_buf: torch.Tensor, recv_buf: torch.Tensor):
        """Ship send_buf to next rank, receive into recv_buf from prev rank.
        Async; caller must call `wait()` before touching recv_buf."""
        ops = [
            dist.P2POp(dist.isend, send_buf, self.send_rank, group=self._pg),
            dist.P2POp(dist.irecv, recv_buf, self.recv_rank, group=self._pg),
        ]
        self._pending.append(dist.batch_isend_irecv(ops))

    def wait(self):
        """Wait for the oldest outstanding batch to complete."""
        if self._pending:
            for req in self._pending.pop(0):
                req.wait()


# ===========================================================================
# 3.  Workspace caches / 工作空间缓存
# ---------------------------------------------------------------------------
# Ring attention needs same-shape scratch tensors (K/V recv buffers, fp32
# accumulators, block-dk/dv). Allocating per call is ~5% overhead. We cache
# by (q shape, k shape, dtype, device); training loops hit 100%.
#
# Ring attention 需要一堆形状固定的 scratch tensor（K/V 接收缓冲、fp32 累加器、
# block-dk/dv 等）。每次调用重新分配大约 5% 的额外开销，我们按
# (q.shape, k.shape, dtype, device) 缓存，训练循环里命中率 100%。
# ===========================================================================
def _next_power_of_2(x): return 1 << (x - 1).bit_length()


def _backward_comm_dtype(local_dtype: torch.dtype) -> torch.dtype:
    """dk/dv transport dtype — default fp32 (safe); env for bf16."""
    return local_dtype if os.getenv("OPTIMIZED_RING_BWD_COMM_BF16", "0") == "1" \
                     else torch.float32


class _ForwardWorkspace:
    """Scratch pool for forward. All tensors keyed by (q shape, k shape,
    dtype, device); training loops re-use these at zero alloc cost."""
    __slots__ = ("kv_bufs", "k_bufs", "v_bufs", "kv_send",
                 "out_acc",          # fp32 output accumulator (Q-shaped)
                 "out",              # bf16 output tensor
                 "lse_acc",          # fp32 log-sum-exp accumulator (per Q head)
                 "acc",              # native path: fp32 [B, H_q, S, D] state
                 "m_state", "l_state",  # native path: fp32 [B, H_q, S] state
                 "q_shape", "kv_shape", "dtype", "device")

    def __init__(self, q, k):
        b, s, h, d = q.shape
        self.q_shape, self.kv_shape = q.shape, k.shape
        self.dtype, self.device = q.dtype, q.device
        # Two combined [2, B, S, H_kv, D] recv buffers, double-buffered.
        self.kv_bufs = [torch.empty((2,) + k.shape, device=q.device, dtype=k.dtype),
                        torch.empty((2,) + k.shape, device=q.device, dtype=k.dtype)]
        self.k_bufs = [self.kv_bufs[0][0], self.kv_bufs[1][0]]
        self.v_bufs = [self.kv_bufs[0][1], self.kv_bufs[1][1]]
        self.kv_send = torch.empty((2,) + k.shape, device=q.device, dtype=k.dtype)
        self.out_acc = torch.empty((b, s, h, d), device=q.device, dtype=torch.float32)
        self.out = torch.empty_like(q)
        self.lse_acc = torch.empty((b, h, s), device=q.device, dtype=torch.float32)
        # Native path also needs (m, l, acc) fp32 state across ring steps.
        self.acc = torch.empty((b, h, s, d), device=q.device, dtype=torch.float32)
        self.m_state = torch.empty((b, h, s), device=q.device, dtype=torch.float32)
        self.l_state = torch.empty_like(self.m_state)

    def matches(self, q, k):
        return (self.q_shape == q.shape and self.kv_shape == k.shape
                and self.dtype == q.dtype and self.device == q.device)


class _BackwardWorkspace:
    """Scratch pool for backward."""
    __slots__ = ("block_dq", "block_dk", "block_dv",
                 "kv_bufs", "k_bufs", "v_bufs", "kv_send",
                 "dkdv_bufs", "dk_bufs", "dv_bufs",
                 "dq_acc",           # fp32 Q-shaped dq accumulator
                 "D",                # fp32 [B, H_q, S] rowsum(dO * O) (native)
                 "block_dk_fp32",    # per-step scratch (native only)
                 "block_dv_fp32",
                 "q_shape", "kv_shape", "dtype", "device")

    def __init__(self, q, k, comm_dtype: torch.dtype):
        b, s, hq, d = q.shape
        _, _, hkv, _ = k.shape
        self.q_shape, self.kv_shape = q.shape, k.shape
        self.dtype, self.device = q.dtype, q.device
        self.block_dq = torch.empty_like(q)
        self.block_dk = torch.empty_like(k)
        self.block_dv = torch.empty_like(k)
        # K/V ring buffers (packed, local dtype).
        self.kv_bufs = [torch.empty((2,) + k.shape, device=q.device, dtype=k.dtype),
                        torch.empty((2,) + k.shape, device=q.device, dtype=k.dtype)]
        self.k_bufs = [self.kv_bufs[0][0], self.kv_bufs[1][0]]
        self.v_bufs = [self.kv_bufs[0][1], self.kv_bufs[1][1]]
        self.kv_send = torch.empty((2,) + k.shape, device=q.device, dtype=k.dtype)
        # dk/dv ring buffers (packed, comm dtype).
        self.dkdv_bufs = [torch.empty((2,) + k.shape, device=q.device, dtype=comm_dtype),
                          torch.empty((2,) + k.shape, device=q.device, dtype=comm_dtype)]
        self.dk_bufs = [self.dkdv_bufs[0][0], self.dkdv_bufs[1][0]]
        self.dv_bufs = [self.dkdv_bufs[0][1], self.dkdv_bufs[1][1]]
        # Q-shaped fp32 dq accumulator + preprocess D.
        self.dq_acc = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        self.D = torch.empty((b, hq, s), device=q.device, dtype=torch.float32)
        # Per-step fp32 scratch (used only by native backward).
        self.block_dk_fp32 = torch.empty(k.shape, device=q.device, dtype=torch.float32)
        self.block_dv_fp32 = torch.empty(k.shape, device=q.device, dtype=torch.float32)

    def matches(self, q, k, comm_dtype):
        return (self.q_shape == q.shape and self.kv_shape == k.shape
                and self.dtype == q.dtype and self.device == q.device
                and self.dkdv_bufs[0].dtype == comm_dtype)


_FWD_WS: Optional[_ForwardWorkspace] = None
_BWD_WS: Optional[_BackwardWorkspace] = None


def _get_fwd_ws(q, k):
    global _FWD_WS
    if _FWD_WS is None or not _FWD_WS.matches(q, k):
        _FWD_WS = _ForwardWorkspace(q, k)
    return _FWD_WS


def _get_bwd_ws(q, k, comm_dtype):
    global _BWD_WS
    if _BWD_WS is None or not _BWD_WS.matches(q, k, comm_dtype):
        _BWD_WS = _BackwardWorkspace(q, k, comm_dtype)
    return _BWD_WS


# ===========================================================================
# 4.  Triton kernels / Triton 内核
# ===========================================================================

# ---------------------------------------------------------------------------
# 4.1  Merge kernel (online-softmax accumulator update)
#      Merge 内核（online-softmax 累加更新）
# ---------------------------------------------------------------------------
# Combines a new flash-attn output block (bf16) with the running fp32
# accumulator (out_acc, lse_acc):
#
#   lse_new = max(lse_a, lse_b) + log(exp(lse_a-lse_new) + exp(lse_b-lse_new))
#   out_new = (out_a*exp(lse_a-lse_new) + out_b*exp(lse_b-lse_new))
#             / (exp(lse_a-lse_new) + exp(lse_b-lse_new))
#
# `final_step=True` writes result to bf16 `out` instead of fp32 accumulator.
#
# 把 flash-attn 新出的 block（bf16）和 fp32 累加器 (out_acc, lse_acc) 合并；
# 这就是 online-softmax 的数值稳定形式，等价于:
#     new_out = weighted_avg(old_out, new_out) 权重由 exp(lse) 决定
# `final_step=True` 时把结果直接写到 bf16 `out`（省一次 cast）。
# ---------------------------------------------------------------------------
@triton.jit
def _merge_kernel(
    block_out_ptr, block_lse_ptr,
    out_acc_ptr, lse_acc_ptr, out_ptr,
    slice_start, slice_len,     # runtime — where in accumulator to merge
    lse_stride_h,               # H-stride of lse_acc (== full local_seq)
    is_first: tl.constexpr,     # if True, no old accumulator to combine with
    is_final: tl.constexpr,     # if True, write to `out` (bf16), else out_acc
    local_seq: tl.constexpr,    # full local seq of destination tensors
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < slice_len
    d_mask = offs_d < head_dim

    # block layout: [B, slice_len, H, D] contiguous in slice_len.
    b_out_base = pid_b * slice_len * num_heads * head_dim + pid_h * head_dim
    b_lse_base = pid_b * num_heads * slice_len + pid_h * slice_len
    # accumulator layout: [B, local_seq, H, D] and lse [B, H, local_seq].
    acc_base = pid_b * local_seq * num_heads * head_dim + pid_h * head_dim
    lse_base = pid_b * num_heads * lse_stride_h + pid_h * lse_stride_h
    dest_m = offs_m + slice_start

    block_out = tl.load(
        block_out_ptr + b_out_base
        + offs_m[:, None] * num_heads * head_dim + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)
    block_lse = tl.load(block_lse_ptr + b_lse_base + offs_m,
                        mask=m_mask, other=-float("inf"))

    if is_first:
        merged_out = block_out
        merged_lse = block_lse
    else:
        old_out = tl.load(
            out_acc_ptr + acc_base
            + dest_m[:, None] * num_heads * head_dim + offs_d[None, :],
            mask=m_mask[:, None] & d_mask[None, :], other=0.0,
        )
        old_lse = tl.load(lse_acc_ptr + lse_base + dest_m,
                          mask=m_mask, other=-float("inf"))
        merged_lse = tl.maximum(old_lse, block_lse)
        old_scale = tl.exp(old_lse - merged_lse)
        block_scale = tl.exp(block_lse - merged_lse)
        denom = old_scale + block_scale
        merged_out = (old_out * old_scale[:, None]
                      + block_out * block_scale[:, None]) / denom[:, None]
        merged_lse = merged_lse + tl.log(denom)

    out_data = merged_out.to(out_ptr.dtype.element_ty) if is_final else merged_out
    dest_ptr = out_ptr if is_final else out_acc_ptr
    tl.store(
        dest_ptr + acc_base
        + dest_m[:, None] * num_heads * head_dim + offs_d[None, :],
        out_data, mask=m_mask[:, None] & d_mask[None, :],
    )
    tl.store(lse_acc_ptr + lse_base + dest_m, merged_lse, mask=m_mask)


def _launch_merge(block_out, block_lse, out_acc, lse_acc, out,
                  is_first, is_final, slice_start, slice_len):
    """Launch merge kernel over a (slice_start, slice_len) window of the
    full-length accumulator. Slice window is used by zigzag; regular ring
    passes (0, local_seq)."""
    batch, full_seq, num_heads, head_dim = out_acc.shape
    BLOCK_D = _next_power_of_2(head_dim)
    # Cap the per-program fp32 register tile (block_out + old_out + merged all
    # live in registers). A fixed BLOCK_M=128 spills badly for large head_dim
    # (e.g. D=256 -> 128x256 fp32 tile), making the merge ~4.5x slower. Keep the
    # tile near ~8192 fp32 elems so hd=64 -> 128, hd=128 -> 64, hd=256 -> 32.
    BLOCK_M = max(16, min(128, 8192 // BLOCK_D))
    grid = (triton.cdiv(slice_len, BLOCK_M), batch, num_heads)
    # num_warps: swept {1,2,4,8,16} across hd=64..256 at the BLOCK_M above; the
    # merge is HBM-bandwidth-bound and nw=4 already saturates it (nw=1/2 slower;
    # nw=8/16 tie within ~1-2us of nw=4 and flip with S, i.e. noise). Keep 4.
    _merge_kernel[grid](
        block_out, block_lse, out_acc, lse_acc, out,
        slice_start, slice_len, full_seq,
        is_first, is_final,
        full_seq, num_heads, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, num_warps=4, num_stages=2,
    )


# ---------------------------------------------------------------------------
# 4.2  Fused grad accumulation (backward, flash-attn path)
#      融合的梯度累加内核（backward 用）
# ---------------------------------------------------------------------------
# Per ring step we get block_dq/dk/dv (bf16 for flash path, fp32 for native).
# We must:
#   dq_fp32 += block_dq                          (Q-shaped)
#   dk_ship  = dk_prev + block_dk (comm dtype)   (KV-shaped, may be smaller)
#   dv_ship  = dv_prev + block_dv (comm dtype)   (KV-shaped)
# Split into two kernels since Q and KV numel differ under GQA. Kernel loads
# block_dk/dv in either bf16 or fp32 — triton figures out the dtype from
# the tensor pointer.
#
# 每个 ring step 会得到本地梯度 block_dq/dk/dv（flash 路径 bf16、native 路径 fp32）
# 需要做的累加:
#   dq_fp32 += block_dq                          (Q shape，Q 头数)
#   dk_ship  = dk_prev + block_dk (comm dtype)   (KV shape，可能因为 GQA 更小)
#   dv_ship  = dv_prev + block_dv (comm dtype)
# 拆成两个 kernel（Q 与 KV 元素数不同）；kernel 里对 bf16/fp32 都能读，triton
# 会自己从 pointer 推 dtype。
# ---------------------------------------------------------------------------
@triton.jit
def _dq_accum_kernel(block_dq_ptr, dq_ptr, numel,
                     is_first: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    b = tl.load(block_dq_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if is_first:
        tl.store(dq_ptr + offs, b, mask=mask)
    else:
        old = tl.load(dq_ptr + offs, mask=mask, other=0.0)
        tl.store(dq_ptr + offs, old + b, mask=mask)


@triton.jit
def _dkdv_accum_kernel(
    block_dk_ptr, block_dv_ptr,
    dk_prev_ptr, dv_prev_ptr, dk_out_ptr, dv_out_ptr,
    numel, is_first: tl.constexpr, BLOCK: tl.constexpr,
):
    """block_dk/dv can be either bf16 (flash path) or fp32 (native path).
    prev/out are in comm_dtype (fp32 or bf16). All casts happen inside the
    kernel — no intermediate tensors."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    b_dk = tl.load(block_dk_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b_dv = tl.load(block_dv_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if is_first:
        new_dk, new_dv = b_dk, b_dv
    else:
        prev_dk = tl.load(dk_prev_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        prev_dv = tl.load(dv_prev_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        new_dk, new_dv = prev_dk + b_dk, prev_dv + b_dv
    tl.store(dk_out_ptr + offs, new_dk.to(dk_out_ptr.dtype.element_ty), mask=mask)
    tl.store(dv_out_ptr + offs, new_dv.to(dv_out_ptr.dtype.element_ty), mask=mask)


def _launch_grad_accum(block_dq, block_dk, block_dv, dq,
                        dk_prev, dv_prev, dk_out, dv_out, is_first):
    """Flash-attn path: block_dq/dk/dv in the original dtype (typically bf16),
    dk/dv accumulators in comm_dtype."""
    BLOCK = 4096
    nq, nk = block_dq.numel(), block_dk.numel()
    _dq_accum_kernel[(triton.cdiv(nq, BLOCK),)](
        block_dq, dq, nq, is_first, BLOCK=BLOCK, num_warps=4)
    _dkdv_accum_kernel[(triton.cdiv(nk, BLOCK),)](
        block_dk, block_dv, dk_prev, dv_prev, dk_out, dv_out, nk,
        is_first, BLOCK=BLOCK, num_warps=4)


def _launch_dkdv_accum(block_dk_fp32, block_dv_fp32,
                       dk_prev, dv_prev, dk_out, dv_out, is_first):
    """Native path: block_dk/dv are already fp32 (triton kernel outputs).
    Fuses the (block.to(comm_dtype) + dk_prev.add_()) sequence into ONE
    kernel — saves 2 kernel launches + 2 intermediate tensors per ring hop.
    """
    BLOCK = 4096
    nk = block_dk_fp32.numel()
    _dkdv_accum_kernel[(triton.cdiv(nk, BLOCK),)](
        block_dk_fp32, block_dv_fp32,
        dk_prev, dv_prev, dk_out, dv_out, nk,
        is_first, BLOCK=BLOCK, num_warps=4)


# ---------------------------------------------------------------------------
# 4.3  Native forward kernel (all-in-one triton flash-attn)
#      Native forward 内核（triton 手写 flash-attn）
# ---------------------------------------------------------------------------
# Standard flash-attn forward, but split per ring step. State (m, l, acc) is
# fp32 in HBM across steps — no bf16 round-trip at ring boundaries.
# GQA: pid_hq iterates over Q heads; pid_hkv = pid_hq // (H_q / H_kv).
#
# 一个标准 flash-attn forward，但被拆成"每个 ring step 一次调用"。关键在于
# online-softmax 的状态 (m, l, acc) 全程 fp32 存 HBM，ring step 之间不 cast 到
# bf16 再读回 —— 省掉每步 bf16↔fp32 的往返带宽。
# GQA 处理：pid_hq 遍历所有 Q head；pid_hkv = pid_hq // group_size 定位对应 KV。
# ---------------------------------------------------------------------------
@triton.jit
def _native_fwd_step_kernel(
    q_ptr, k_ptr, v_ptr,
    acc_ptr, m_ptr, l_ptr,
    out_ptr, lse_ptr,
    local_seq: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale_log2: tl.constexpr,  # scale * log2(e), for exp2
    first_step: tl.constexpr,
    final_step: tl.constexpr,
    causal_mode: tl.constexpr,          # 0=full, 1=diagonal causal
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hq = tl.program_id(2)
    group_size: tl.constexpr = num_q_heads // num_kv_heads
    pid_hkv = pid_hq // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < local_seq
    d_mask = offs_d < head_dim

    q_base = pid_b * local_seq * num_q_heads * head_dim + pid_hq * head_dim
    q = tl.load(
        q_ptr + q_base + offs_m[:, None] * num_q_heads * head_dim + offs_d[None, :],
        mask=q_mask[:, None] & d_mask[None, :], other=0.0,
    )
    # State layout: [B, H_q, S] for m/l/lse; [B, H_q, S, D] for acc.
    state_base = pid_b * num_q_heads * local_seq
    acc_base = state_base * head_dim + pid_hq * local_seq * head_dim

    if first_step:
        m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    else:
        m = tl.load(m_ptr + state_base + pid_hq * local_seq + offs_m,
                    mask=q_mask, other=0.0)
        l = tl.load(l_ptr + state_base + pid_hq * local_seq + offs_m,
                    mask=q_mask, other=0.0)
        acc = tl.load(
            acc_ptr + acc_base + offs_m[:, None] * head_dim + offs_d[None, :],
            mask=q_mask[:, None] & d_mask[None, :], other=0.0,
        )

    for start_n in range(0, local_seq, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < local_seq
        # Causal early-out: skip whole N-tile if fully above the diagonal.
        if causal_mode == 0 or start_n <= (pid_m + 1) * BLOCK_M - 1:
            kv_base = (pid_b * local_seq * num_kv_heads * head_dim
                       + pid_hkv * head_dim)
            k = tl.load(
                k_ptr + kv_base + offs_n[:, None] * num_kv_heads * head_dim + offs_d[None, :],
                mask=n_mask[:, None] & d_mask[None, :], other=0.0,
            )
            v = tl.load(
                v_ptr + kv_base + offs_n[:, None] * num_kv_heads * head_dim + offs_d[None, :],
                mask=n_mask[:, None] & d_mask[None, :], other=0.0,
            )
            # scores in log2 space so we can use fast exp2.
            scores = tl.dot(q, tl.trans(k)) * softmax_scale_log2
            score_mask = q_mask[:, None] & n_mask[None, :]
            if causal_mode == 1:
                score_mask = score_mask & (offs_n[None, :] <= offs_m[:, None])
            scores = tl.where(score_mask, scores, -float("inf"))
            block_m_val = tl.max(scores, axis=1)
            new_m = tl.maximum(m, block_m_val)
            alpha = tl.exp2(m - new_m)
            p = tl.exp2(scores - new_m[:, None])
            new_l = l * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m = new_m
            l = new_l

    if final_step:
        out = acc / l[:, None]
        out_base = pid_b * local_seq * num_q_heads * head_dim + pid_hq * head_dim
        tl.store(
            out_ptr + out_base + offs_m[:, None] * num_q_heads * head_dim + offs_d[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=q_mask[:, None] & d_mask[None, :],
        )
        # Convert m from log2 back to natural log for lse output.
        lse = m * 0.6931471805599453 + tl.log(l)
        tl.store(lse_ptr + state_base + pid_hq * local_seq + offs_m, lse, mask=q_mask)
    else:
        tl.store(m_ptr + state_base + pid_hq * local_seq + offs_m, m, mask=q_mask)
        tl.store(l_ptr + state_base + pid_hq * local_seq + offs_m, l, mask=q_mask)
        tl.store(
            acc_ptr + acc_base + offs_m[:, None] * head_dim + offs_d[None, :],
            acc, mask=q_mask[:, None] & d_mask[None, :],
        )


def _native_fwd_cfg(head_dim):
    """(BLOCK_M, BLOCK_N, BLOCK_D, num_warps, num_stages) — H100 hand-tuned."""
    bd = _next_power_of_2(head_dim)
    if head_dim <= 32:  return (128, 128, bd, 4, 3)
    if head_dim <= 64:  return (128, 64,  bd, 4, 3)
    if head_dim <= 128: return (128, 128, bd, 8, 3)
    return (64, 32, bd, 8, 2)


# ---------------------------------------------------------------------------
# 4.4  Native backward kernels (hd <= 64 in the default dispatcher)
#      Native backward 内核（默认仅在 hd ≤ 64 时启用）
# ---------------------------------------------------------------------------
# Attention backward math / Attention 反向传播公式:
#   P = softmax(Q K^T * scale)
#   dP = dO V^T
#   D  = rowsum(dO * O)   (precomputed once per forward pass)
#   dS = P * (dP - D)
#   dQ = dS @ K * scale      (per Q row; sums over N)
#   dK = dS^T @ Q * scale    (per K row; sums over M)
#   dV = P^T @ dO            (per V row; sums over M)
#
# Three kernels: preprocess D, per-step dQ contribution, per-step dK/dV.
# 三个 kernel：先预计算 D，然后每个 ring step 分别累加 dQ、算出本步 dK/dV。
# 其中 dQ 每步 add 到 fp32 累加器；dK/dV 每步写到 fp32 输出，由 ring 通信在
# 世界间循环累加，最终在 K/V 起点 rank 汇总。
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_preprocess_kernel(
    out_ptr, dout_ptr, d_ptr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """D[b, h, m] = sum_d out[b,m,h,d] * dout[b,m,h,d]. Called once per fwd."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < seq_len
    d_mask = offs_d < head_dim
    base = pid_b * seq_len * num_heads * head_dim + pid_h * head_dim
    o = tl.load(out_ptr + base
                + offs_m[:, None] * num_heads * head_dim + offs_d[None, :],
                mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    do = tl.load(dout_ptr + base
                 + offs_m[:, None] * num_heads * head_dim + offs_d[None, :],
                 mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    tl.store(d_ptr + pid_b * num_heads * seq_len + pid_h * seq_len + offs_m,
             tl.sum(o * do, axis=1), mask=m_mask)


@triton.jit
def _bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, lse_ptr, d_ptr,
    dout_ptr, dq_acc_ptr,
    seq_len: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale,
    causal_mode: tl.constexpr,
    add_to_acc: tl.constexpr,  # False on first step (overwrite), True after (add)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Per Q-block, add this step's dQ contribution."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hq = tl.program_id(2)
    group_size: tl.constexpr = num_q_heads // num_kv_heads
    pid_hkv = pid_hq // group_size
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < seq_len
    d_mask = offs_d < head_dim

    q_base = pid_b * seq_len * num_q_heads * head_dim + pid_hq * head_dim
    kv_base = pid_b * seq_len * num_kv_heads * head_dim + pid_hkv * head_dim
    state_base = pid_b * num_q_heads * seq_len + pid_hq * seq_len

    q = tl.load(q_ptr + q_base
                + offs_m[:, None] * num_q_heads * head_dim + offs_d[None, :],
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    do = tl.load(dout_ptr + q_base
                 + offs_m[:, None] * num_q_heads * head_dim + offs_d[None, :],
                 mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    lse = tl.load(lse_ptr + state_base + offs_m, mask=m_mask, other=0.0)
    D_i = tl.load(d_ptr + state_base + offs_m, mask=m_mask, other=0.0)

    dq_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        if causal_mode == 0 or start_n <= (pid_m + 1) * BLOCK_M - 1:
            k = tl.load(k_ptr + kv_base
                        + offs_n[:, None] * num_kv_heads * head_dim + offs_d[None, :],
                        mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            v = tl.load(v_ptr + kv_base
                        + offs_n[:, None] * num_kv_heads * head_dim + offs_d[None, :],
                        mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * softmax_scale
            score_mask = m_mask[:, None] & n_mask[None, :]
            if causal_mode == 1:
                score_mask = score_mask & (offs_n[None, :] <= offs_m[:, None])
            scores = tl.where(score_mask, scores, -float("inf"))
            p = tl.exp(scores - lse[:, None])
            dp = tl.dot(do, tl.trans(v)).to(tl.float32)
            ds = p * (dp - D_i[:, None])
            dq_acc += tl.dot((ds * softmax_scale).to(k.dtype), k).to(tl.float32)

    ptr = dq_acc_ptr + q_base + offs_m[:, None] * num_q_heads * head_dim + offs_d[None, :]
    mask2 = m_mask[:, None] & d_mask[None, :]
    if add_to_acc:
        tl.store(ptr, tl.load(ptr, mask=mask2, other=0.0) + dq_acc, mask=mask2)
    else:
        tl.store(ptr, dq_acc, mask=mask2)


@triton.jit
def _bwd_dkdv_kernel(
    q_ptr, k_ptr, v_ptr, lse_ptr, d_ptr,
    dout_ptr, dk_out_ptr, dv_out_ptr,
    seq_len: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale,
    causal_mode: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Per KV-block, compute this step's dK/dV over all Q rows and all Q
    heads in the GQA group. Writes fp32 outputs (caller does ring-accum)."""
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)
    group_size: tl.constexpr = num_q_heads // num_kv_heads
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < seq_len
    d_mask = offs_d < head_dim

    kv_base = pid_b * seq_len * num_kv_heads * head_dim + pid_hkv * head_dim
    k = tl.load(k_ptr + kv_base
                + offs_n[:, None] * num_kv_heads * head_dim + offs_d[None, :],
                mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    v = tl.load(v_ptr + kv_base
                + offs_n[:, None] * num_kv_heads * head_dim + offs_d[None, :],
                mask=n_mask[:, None] & d_mask[None, :], other=0.0)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    for hq_off in range(0, group_size):
        pid_hq = pid_hkv * group_size + hq_off
        q_base = pid_b * seq_len * num_q_heads * head_dim + pid_hq * head_dim
        state_base = pid_b * num_q_heads * seq_len + pid_hq * seq_len
        for start_m in range(0, seq_len, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            m_mask = offs_m < seq_len
            if causal_mode == 0 or (pid_n * BLOCK_N) <= (start_m + BLOCK_M - 1):
                q = tl.load(q_ptr + q_base
                            + offs_m[:, None] * num_q_heads * head_dim + offs_d[None, :],
                            mask=m_mask[:, None] & d_mask[None, :], other=0.0)
                do = tl.load(dout_ptr + q_base
                             + offs_m[:, None] * num_q_heads * head_dim + offs_d[None, :],
                             mask=m_mask[:, None] & d_mask[None, :], other=0.0)
                lse = tl.load(lse_ptr + state_base + offs_m, mask=m_mask, other=0.0)
                D_i = tl.load(d_ptr + state_base + offs_m, mask=m_mask, other=0.0)

                scores = tl.dot(q, tl.trans(k)).to(tl.float32) * softmax_scale
                score_mask = m_mask[:, None] & n_mask[None, :]
                if causal_mode == 1:
                    score_mask = score_mask & (offs_n[None, :] <= offs_m[:, None])
                scores = tl.where(score_mask, scores, -float("inf"))
                p = tl.exp(scores - lse[:, None])
                dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do).to(tl.float32)
                dp = tl.dot(do, tl.trans(v)).to(tl.float32)
                ds = p * (dp - D_i[:, None])
                dk_acc += tl.dot(tl.trans((ds * softmax_scale).to(q.dtype)), q).to(tl.float32)

    tl.store(dk_out_ptr + kv_base
             + offs_n[:, None] * num_kv_heads * head_dim + offs_d[None, :],
             dk_acc, mask=n_mask[:, None] & d_mask[None, :])
    tl.store(dv_out_ptr + kv_base
             + offs_n[:, None] * num_kv_heads * head_dim + offs_d[None, :],
             dv_acc, mask=n_mask[:, None] & d_mask[None, :])


# ===========================================================================
# 5.  Ring forward / 环形 forward
# ===========================================================================

# ---------------------------------------------------------------------------
# 5.1  Native forward — triton flash-attn with fp32 handoff between ring
#      steps. Best for hd <= 64 always, and hd <= 128 with ws >= 4.
#      跨 ring step 保持 fp32 状态的 triton 手写 flash-attn。
#      hd ≤ 64 恒定优选；hd ≤ 128 需要 ws ≥ 4（ring hop 多才划算）。
# ---------------------------------------------------------------------------
def _ring_forward_native(process_group, q, k, v, softmax_scale, causal, return_lse):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, local_seq, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape
    assert num_q_heads % num_kv_heads == 0, "GQA requires H_q % H_kv == 0"
    comm = DoubleBufRingComm(process_group)
    BM, BN, BD, num_warps, num_stages = _native_fwd_cfg(head_dim)
    grid = (triton.cdiv(local_seq, BM), batch, num_q_heads)

    ws = _get_fwd_ws(q, k)
    out = ws.out
    # fp32 state across ring steps (cached in workspace).
    lse = ws.lse_acc
    acc = ws.acc
    m_state = ws.m_state
    l_state = ws.l_state

    # Prime the initial send buffer with the local K/V.
    kv_send = ws.kv_send
    if comm.world_size > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)
    k_cur, v_cur = k, v

    # Causal: rank r only sees K/V from ranks r, r-1, ..., 0 (steps 0..r).
    last_valid_step = comm.rank if causal else comm.world_size - 1
    valid_idx = 0
    scale_log2 = float(softmax_scale) * 1.4426950408889634

    for step in range(comm.world_size):
        # Start next hop before compute so NCCL overlaps.
        if step + 1 != comm.world_size:
            comm.send_recv_packed(kv_send, ws.kv_bufs[step & 1])
        if not causal or step <= comm.rank:
            # Only the diagonal (rank-owned) K/V block needs the causal mask.
            k_block_rank = (comm.rank - step) % comm.world_size
            causal_mode = 1 if (causal and k_block_rank == comm.rank) else 0
            _native_fwd_step_kernel[grid](
                q, k_cur, v_cur, acc, m_state, l_state, out, lse,
                local_seq, num_q_heads, num_kv_heads, head_dim,
                scale_log2, valid_idx == 0, step == last_valid_step, causal_mode,
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD,
                num_warps=num_warps, num_stages=num_stages,
            )
            valid_idx += 1
        if step + 1 != comm.world_size:
            comm.wait()
            kv_send = ws.kv_bufs[step & 1]
            k_cur, v_cur = ws.k_bufs[step & 1], ws.v_bufs[step & 1]

    return (out, lse) if return_lse else out


# ---------------------------------------------------------------------------
# 5.2  Flash-attn forward + triton merge — fallback for hd > 128 or when
#      forced. flash-attn's cutlass kernel wins single-step; we still win on
#      merge kernel + packed comm + workspace reuse.
#      flash-attn 单步 + triton merge。hd > 128 或强制 fallback 场景走这条。
#      单步 attention 用 flash-attn 的 cutlass kernel（更快），但 merge、
#      通信、workspace 缓存的收益仍然全部拿到。
# ---------------------------------------------------------------------------
def _ring_forward_flash_merge(process_group, q, k, v, softmax_scale, causal, return_lse):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, local_seq, num_q_heads, head_dim = q.shape
    comm = DoubleBufRingComm(process_group)

    ws = _get_fwd_ws(q, k)
    out = ws.out
    lse_acc = ws.lse_acc
    out_acc = ws.out_acc

    kv_send = ws.kv_send
    if comm.world_size > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)
    k_cur, v_cur = k, v

    last_valid_step = comm.rank if causal else comm.world_size - 1
    valid_idx = 0

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            comm.send_recv_packed(kv_send, ws.kv_bufs[step & 1])
        if not causal or step <= comm.rank:
            block_out, block_lse = _flash_forward_block(
                q, k_cur, v_cur, softmax_scale, causal and step == 0)
            _launch_merge(
                block_out, block_lse, out_acc, lse_acc, out,
                is_first=(valid_idx == 0), is_final=(step == last_valid_step),
                slice_start=0, slice_len=local_seq,
            )
            valid_idx += 1
        if step + 1 != comm.world_size:
            comm.wait()
            kv_send = ws.kv_bufs[step & 1]
            k_cur, v_cur = ws.k_bufs[step & 1], ws.v_bufs[step & 1]

    return (out, lse_acc) if return_lse else out


def _select_forward_impl(head_dim, num_q_heads, local_seq, world_size):
    """Pick between native triton fwd and flash-attn + triton merge.

    Native fwd runs one triton kernel per ring step with fp32 (m, l, acc)
    state held across steps — best when compute is small enough that launch
    overhead / bf16 round-trip savings dominate. It underperforms flash-attn
    cutlass on large per-step compute (large Hq × local_seq × head_dim).
    """
    forced = os.getenv("OPTIMIZED_RING_FORWARD_IMPL", "auto").strip().lower()
    if forced in ("triton_native", "flash_triton_merge"):
        return forced
    if head_dim > 128:
        return "flash_triton_merge"
    if head_dim > 64 and world_size < 4:
        return "flash_triton_merge"
    # Fall back to flash+merge when Hq × local_seq gets large: many CTAs
    # each with long N-loop, flash-attn cutlass wins single-step compute.
    if num_q_heads * local_seq >= 262144:  # e.g. Hq=16 × 16k local_seq
        return "flash_triton_merge"
    return "triton_native"


# ===========================================================================
# 6.  Ring backward / 环形 backward
# ===========================================================================

# ---------------------------------------------------------------------------
# 6.1  Native backward (hd <= 64) — our triton kernels, fp32 dQ across steps.
#      Native backward（hd ≤ 64）：手写 triton kernel，dQ 跨 ring step 保持 fp32。
# ---------------------------------------------------------------------------
def _ring_backward_native(process_group, dout, q, k, v, out, softmax_lse,
                          softmax_scale, causal):
    kv_comm = DoubleBufRingComm(process_group)
    d_kv_comm = DoubleBufRingComm(process_group)
    world_size = kv_comm.world_size
    rank = kv_comm.rank

    batch, local_seq, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape
    comm_dtype = _backward_comm_dtype(q.dtype)
    ws = _get_bwd_ws(q, k, comm_dtype)

    # Cached fp32 scratch tensors.
    block_dk_fp32 = ws.block_dk_fp32
    block_dv_fp32 = ws.block_dv_fp32
    dq_acc = ws.dq_acc
    D = ws.D

    # D once per full forward pass.
    _bwd_preprocess_kernel[
        (triton.cdiv(local_seq, 128), batch, num_q_heads)
    ](out, dout, D, local_seq, num_q_heads, head_dim,
      BLOCK_M=128, BLOCK_D=_next_power_of_2(head_dim),
      num_warps=4, num_stages=2)

    BM = 128 if head_dim <= 32 else 64
    BN, BD = 64, _next_power_of_2(head_dim)

    kv_send = ws.kv_send
    if world_size > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)
    k_cur, v_cur = k, v
    first_iter_done = False

    for step in range(world_size):
        if step + 1 != world_size:
            kv_comm.send_recv_packed(kv_send, ws.kv_bufs[step & 1])
        active = step <= rank or not causal
        prev_slot = (step - 1) & 1

        if active:
            causal_mode = 1 if (causal and step == 0) else 0
            _bwd_dq_kernel[
                (triton.cdiv(local_seq, BM), batch, num_q_heads)
            ](q, k_cur, v_cur, softmax_lse, D, dout, dq_acc,
              local_seq, num_q_heads, num_kv_heads, head_dim,
              float(softmax_scale), causal_mode, first_iter_done,
              BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD, num_warps=4, num_stages=2)
            _bwd_dkdv_kernel[
                (triton.cdiv(local_seq, BN), batch, num_kv_heads)
            ](q, k_cur, v_cur, softmax_lse, D, dout,
              block_dk_fp32, block_dv_fp32,
              local_seq, num_q_heads, num_kv_heads, head_dim,
              float(softmax_scale), causal_mode,
              BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD, num_warps=4, num_stages=2)
            if first_iter_done:
                d_kv_comm.wait()
            # One fused kernel: (fp32 block_dk/dv) [+ comm_dtype dk_prev/dv_prev]
            # -> comm_dtype dk_out/dv_out. Replaces the old
            # `dk_bufs[slot].add_(block.to(comm_dtype))` pair (4 launches +
            # 2 intermediate tensors) with a single kernel.
            _launch_dkdv_accum(
                block_dk_fp32, block_dv_fp32,
                ws.dk_bufs[prev_slot], ws.dv_bufs[prev_slot],
                ws.dk_bufs[prev_slot], ws.dv_bufs[prev_slot],
                is_first=not first_iter_done,
            )
            first_iter_done = True
        elif step != 0:
            d_kv_comm.wait()

        if step + 1 != world_size:
            kv_comm.wait()
            kv_send = ws.kv_bufs[step & 1]
            k_cur, v_cur = ws.k_bufs[step & 1], ws.v_bufs[step & 1]

        d_kv_comm.send_recv_packed(ws.dkdv_bufs[prev_slot],
                                    ws.dkdv_bufs[step & 1])

    d_kv_comm.wait()
    final_slot = (world_size - 1) & 1
    return (dq_acc.to(q.dtype),
            ws.dk_bufs[final_slot].to(k.dtype),
            ws.dv_bufs[final_slot].to(v.dtype))


# ---------------------------------------------------------------------------
# 6.2  Flash-attn backward + fused grad-accum kernel. For hd > 64 or forced.
#      Flash-attn backward + 融合梯度累加 kernel。hd > 64 或强制 fallback 场景走这条。
# ---------------------------------------------------------------------------
def _ring_backward_flash(process_group, dout, q, k, v, out, softmax_lse,
                         softmax_scale, causal):
    kv_comm = DoubleBufRingComm(process_group)
    d_kv_comm = DoubleBufRingComm(process_group)
    comm_dtype = _backward_comm_dtype(q.dtype)
    ws = _get_bwd_ws(q, k, comm_dtype)

    dq = ws.dq_acc  # fp32 Q-shaped, cached
    world_size = kv_comm.world_size
    rank = kv_comm.rank

    kv_send = ws.kv_send
    if world_size > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)
    k_cur, v_cur = k, v
    first_iter_done = False

    for step in range(world_size):
        if step + 1 != world_size:
            kv_comm.send_recv_packed(kv_send, ws.kv_bufs[step & 1])
        active = step <= rank or not causal
        prev_slot = (step - 1) & 1

        if active:
            _flash_backward_block(
                dout, q, k_cur, v_cur, out, softmax_lse,
                ws.block_dq, ws.block_dk, ws.block_dv,
                softmax_scale, causal and step == 0,
            )
            if first_iter_done:
                d_kv_comm.wait()
            # Fused: dq += block_dq;  dk_out = dk_prev + block_dk (in-place slot).
            _launch_grad_accum(
                ws.block_dq, ws.block_dk, ws.block_dv, dq,
                ws.dk_bufs[prev_slot], ws.dv_bufs[prev_slot],
                ws.dk_bufs[prev_slot], ws.dv_bufs[prev_slot],
                is_first=not first_iter_done,
            )
            first_iter_done = True
        elif step != 0:
            d_kv_comm.wait()

        if step + 1 != world_size:
            kv_comm.wait()
            kv_send = ws.kv_bufs[step & 1]
            k_cur, v_cur = ws.k_bufs[step & 1], ws.v_bufs[step & 1]

        d_kv_comm.send_recv_packed(ws.dkdv_bufs[prev_slot],
                                    ws.dkdv_bufs[step & 1])

    d_kv_comm.wait()
    final_slot = (world_size - 1) & 1
    return (dq.to(q.dtype),
            ws.dk_bufs[final_slot].to(k.dtype),
            ws.dv_bufs[final_slot].to(v.dtype))


def _select_backward_impl(head_dim, num_q_heads, num_kv_heads, local_seq):
    """Pick between native triton bwd and flash-attn bwd (with fused accum).

    Native wins on small/mid seq for hd<=64 (fp32 dQ handoff across ring steps
    avoids bf16 round-trips). It loses on large workloads where:
      (a) `_bwd_dkdv_kernel` inner loop over group_size Q heads makes each
          KV-tile CTA too serial (`group_size * local_seq` proxy), or
      (b) large Hq * local_seq means many CTAs and each has a long M-loop,
          amortizing kernel launch/pipeline setup badly vs. flash-attn cutlass.
    Both thresholds hand-tuned on H100.
    """
    forced = os.getenv("OPTIMIZED_RING_BACKWARD_IMPL", "auto").strip().lower()
    if forced in ("triton_native", "flash"):
        return forced
    if head_dim > 64 or num_q_heads % num_kv_heads != 0:
        return "flash"
    group_size = num_q_heads // num_kv_heads
    if group_size * local_seq > 16384:
        return "flash"
    # Total Q-row work across all Q heads: when large, `_bwd_dkdv_kernel`'s
    # per-CTA M-loop (∝ local_seq/BM) × Q-head fanout underperforms flash-attn.
    if num_q_heads * local_seq >= 65536:
        return "flash"
    return "triton_native"


# ===========================================================================
# 7.  Zigzag ring — causal load balancer / zigzag 环形（causal 负载均衡）
# ---------------------------------------------------------------------------
# Standard causal ring: rank r does r+1 steps, so rank W-1 is the tail.
# Zigzag: seq split into 2W chunks; rank r holds [chunk_r || chunk_{2W-1-r}].
# Each rank does W flash-attn calls per pass (each ~= half work).
#
# 标准 causal ring 中 rank r 做 r+1 步 flash-attn，rank W-1 是尾部瓶颈。
# zigzag 把 seq 切 2W 段：rank r 拿 [chunk_r || chunk_{2W-1-r}] 拼起来。
# 这样每个 rank 都做 W 次 flash-attn，每次约半量，负载均衡。
#
# Per-step logic (matches zhuzilin/ring-flash-attention semantics):
#   step 0:        causal on full local (q, k, v)
#   step <= rank:  full attn from Q_full onto K/V front-half only
#   step >  rank:  full attn from Q back-half onto K/V full
#
# 每步逻辑（与 zhuzilin/ring-flash-attention 一致）：
#   step 0        ：本地 (q, k, v) 完整 causal
#   step ≤ rank   ：Q_full × K/V 前半段做完整 attention
#   step > rank   ：Q 后半段 × K/V 全长做完整 attention
# ===========================================================================
def _zigzag_forward(process_group, q, k, v, softmax_scale, return_lse):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    comm = DoubleBufRingComm(process_group)
    batch, local_seq, num_heads, head_dim = q.shape
    assert local_seq % 2 == 0, "zigzag requires even local_seq"
    half = local_seq // 2

    ws = _get_fwd_ws(q, k)
    out = ws.out
    out_acc = ws.out_acc
    lse_acc = ws.lse_acc

    kv_send = ws.kv_send
    if comm.world_size > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)
    k_cur, v_cur = k, v
    q1 = q[:, half:].contiguous()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            comm.send_recv_packed(kv_send, ws.kv_bufs[step & 1])

        if step == 0:
            # Full local causal — Q_full × (K_full, V_full).
            block_out, block_lse = _flash_forward_block(
                q, k_cur, v_cur, softmax_scale, causal=True)
            _launch_merge(block_out, block_lse, out_acc, lse_acc, out,
                          is_first=True, is_final=False,
                          slice_start=0, slice_len=local_seq)
        elif step <= comm.rank:
            # Q_full × K/V front-half only.
            block_out, block_lse = _flash_forward_block(
                q, k_cur[:, :half].contiguous(), v_cur[:, :half].contiguous(),
                softmax_scale, causal=False)
            _launch_merge(block_out, block_lse, out_acc, lse_acc, out,
                          is_first=False, is_final=False,
                          slice_start=0, slice_len=local_seq)
        else:
            # Q back-half × K/V full.
            block_out, block_lse = _flash_forward_block(
                q1, k_cur, v_cur, softmax_scale, causal=False)
            _launch_merge(block_out, block_lse, out_acc, lse_acc, out,
                          is_first=False, is_final=False,
                          slice_start=half, slice_len=half)

        if step + 1 != comm.world_size:
            comm.wait()
            kv_send = ws.kv_bufs[step & 1]
            k_cur, v_cur = ws.k_bufs[step & 1], ws.v_bufs[step & 1]

    # Final cast fp32 accumulator -> bf16 output (one HBM pass).
    out.copy_(out_acc)
    return (out, lse_acc) if return_lse else out


def _zigzag_backward(process_group, dout, q, k, v, out, softmax_lse, softmax_scale):
    kv_comm = DoubleBufRingComm(process_group)
    d_kv_comm = DoubleBufRingComm(process_group)
    world_size = kv_comm.world_size
    rank = kv_comm.rank
    b, s, h, d = q.shape
    half = s // 2
    assert s % 2 == 0

    comm_dtype = _backward_comm_dtype(q.dtype)
    ws = _get_bwd_ws(q, k, comm_dtype)

    # Slice views (used for zigzag's back-half branches).
    dout1 = dout[:, half:].contiguous()
    q1 = q[:, half:].contiguous()
    out1 = out[:, half:].contiguous()
    softmax_lse1 = softmax_lse[:, :, half:].contiguous()

    kv_send = ws.kv_send
    if world_size > 1:
        kv_send[0].copy_(k); kv_send[1].copy_(v)
    k_cur, v_cur = k, v

    dq = dk_cur = dv_cur = None

    for step in range(world_size):
        if step + 1 != world_size:
            kv_comm.send_recv_packed(kv_send, ws.kv_bufs[step & 1])

        if step == 0:
            _flash_backward_block(
                dout, q, k_cur, v_cur, out, softmax_lse,
                ws.block_dq, ws.block_dk, ws.block_dv,
                softmax_scale, causal=True)
            dq = ws.block_dq.to(torch.float32)
            dk_cur = ws.block_dk.to(torch.float32)
            dv_cur = ws.block_dv.to(torch.float32)
        else:
            if step <= rank:
                _flash_backward_block(
                    dout, q, k_cur[:, :half].contiguous(), v_cur[:, :half].contiguous(),
                    out, softmax_lse,
                    ws.block_dq, ws.block_dk[:, :half], ws.block_dv[:, :half],
                    softmax_scale, causal=False)
                dq.add_(ws.block_dq)
            else:
                _flash_backward_block(
                    dout1, q1, k_cur, v_cur, out1, softmax_lse1,
                    ws.block_dq[:, :half], ws.block_dk, ws.block_dv,
                    softmax_scale, causal=False)
                dq[:, half:].add_(ws.block_dq[:, :half])

            # Merge received dk/dv with this step's contribution.
            d_kv_comm.wait()
            recv_slot = 1 - ((step - 1) & 1)
            dk_cur = ws.dk_bufs[recv_slot].to(torch.float32)
            dv_cur = ws.dv_bufs[recv_slot].to(torch.float32)
            if step <= rank:
                dk_cur[:, :half].add_(ws.block_dk[:, :half])
                dv_cur[:, :half].add_(ws.block_dv[:, :half])
            else:
                dk_cur.add_(ws.block_dk)
                dv_cur.add_(ws.block_dv)

        if step + 1 != world_size:
            kv_comm.wait()
            kv_send = ws.kv_bufs[step & 1]
            k_cur, v_cur = ws.k_bufs[step & 1], ws.v_bufs[step & 1]

        # Ship dk/dv forward.
        send_slot = step & 1
        ws.dk_bufs[send_slot].copy_(dk_cur)
        ws.dv_bufs[send_slot].copy_(dv_cur)
        d_kv_comm.send_recv_packed(ws.dkdv_bufs[send_slot],
                                    ws.dkdv_bufs[1 - send_slot])

    d_kv_comm.wait()
    final_slot = 1 - ((world_size - 1) & 1)
    return (dq.to(q.dtype),
            ws.dk_bufs[final_slot].to(k.dtype),
            ws.dv_bufs[final_slot].to(v.dtype))


# ===========================================================================
# 8.  Autograd Function + user entry point
#     Autograd Function 与用户入口
# ---------------------------------------------------------------------------
# `_RingFlashAttnFunc` implements the standard torch.autograd.Function
# contract: `forward` saves inputs + intermediate for backward, `backward`
# reads saved tensors and returns per-input gradients. It dispatches to the
# native / flash-triton-merge / zigzag path based on shape.
#
# `ring_flash_attn(...)` is the sole user-facing API — call it just like
# `flash_attn_func`, with an extra `causal` / `zigzag` / `group` argument.
#
# `_RingFlashAttnFunc` 是标准 torch.autograd.Function：forward 保存输入 +
# 中间量供 backward 使用，backward 读回并返回每个输入的梯度。内部按 shape
# 选择 native / flash+merge / zigzag 路径。
#
# `ring_flash_attn(...)` 是唯一的用户入口，用法与 `flash_attn_func` 一致，
# 额外接受 `causal` / `zigzag` / `group` 三个参数。
# ===========================================================================
class _RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, causal, zigzag, group):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1

        if zigzag:
            assert causal, "zigzag mode requires causal=True"
            out, lse = _zigzag_forward(group, q, k, v, softmax_scale, return_lse=True)
        else:
            impl = _select_forward_impl(q.shape[-1], q.shape[2], q.shape[1], world_size)
            fwd = _ring_forward_native if impl == "triton_native" \
                                        else _ring_forward_flash_merge
            out, lse = fwd(group, q, k, v, softmax_scale, causal, return_lse=True)

        ctx.save_for_backward(q, k.contiguous(), v.contiguous(), out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.zigzag = zigzag
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        if ctx.zigzag:
            dq, dk, dv = _zigzag_backward(
                ctx.group, dout, q, k, v, out, lse, ctx.softmax_scale)
        else:
            impl = _select_backward_impl(
                q.shape[-1], q.shape[2], k.shape[2], q.shape[1])
            bwd = _ring_backward_native if impl == "triton_native" \
                                          else _ring_backward_flash
            dq, dk, dv = bwd(ctx.group, dout, q, k, v, out, lse,
                             ctx.softmax_scale, ctx.causal)
        # No grad w.r.t. softmax_scale/causal/zigzag/group.
        return dq, dk, dv, None, None, None, None


def ring_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    zigzag: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Ring flash attention (sequence-parallel).

    Args:
      q:  [B, S_local, H_q, D]    bf16/fp16
      k:  [B, S_local, H_kv, D]   bf16/fp16 (H_kv can be < H_q for GQA)
      v:  [B, S_local, H_kv, D]   bf16/fp16
      softmax_scale: 1/sqrt(D) by default.
      causal: causal mask over the FULL (global) sequence.
      zigzag: if True, expect zigzag-tiled input layout — for global seq of
              length S split into 2*W chunks, rank r holds
              [chunk_r, chunk_{2W-1-r}] concatenated along seq dim.
              Only meaningful with causal=True; better load balance.
      group:  NCCL process group. Defaults to WORLD.

    Returns:
      out: [B, S_local, H_q, D] same dtype as q.

    Notes:
      Requires H_q % H_kv == 0 (GQA). Head_dim up to 256 (hd > 128 runs on
      the flash+merge path; hd <= 128 can use the native triton path).
    """
    return _RingFlashAttnFunc.apply(q, k, v, softmax_scale, causal, zigzag, group)


# ===========================================================================
# 9. __main__ — smoke test + micro-benchmark / 冒烟测试 + 微基准
# ---------------------------------------------------------------------------
# Run with / 运行方式:
#   torchrun --nproc_per_node=<N> --standalone optimized_ring_flash_attn.py
#
# What it does:
#   * builds a small (B=1, S=8192, H_q=8, H_kv=2, D=64) bf16 GQA test case
#   * compares against single-GPU flash-attn reference (fwd + bwd)
#   * runs a zigzag pass for demonstration
#   * measures fwd-only and fwd+bwd latency
#
# 运行内容：
#   * 构造 (B=1, S=8192, H_q=8, H_kv=2, D=64) 的 bf16 GQA 测试用例
#   * 与单卡 flash-attn 参考实现比对 forward + backward 数值一致性
#   * 跑一次 zigzag 路径确认能通
#   * 分别测量 fwd-only 和 fwd+bwd 的延迟
# ===========================================================================
def _demo():
    from flash_attn import flash_attn_func

    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        B, S_full, H_q, H_kv, D = 1, 8192, 8, 2, 64
        assert S_full % world_size == 0

        torch.manual_seed(1234)
        dtype = torch.bfloat16
        q_full = torch.randn(B, S_full, H_q, D, device=device, dtype=dtype)
        k_full = torch.randn(B, S_full, H_kv, D, device=device, dtype=dtype)
        v_full = torch.randn(B, S_full, H_kv, D, device=device, dtype=dtype)
        dout_full = torch.randn_like(q_full)

        def _chunk_local(x):
            return torch.chunk(x, world_size, dim=1)[rank].contiguous()

        # ---------- Correctness: fwd + bwd for both causal and non-causal ----------
        for causal in (False, True):
            ql = _chunk_local(q_full).detach().clone().requires_grad_(True)
            kl = _chunk_local(k_full).detach().clone().requires_grad_(True)
            vl = _chunk_local(v_full).detach().clone().requires_grad_(True)
            dol = _chunk_local(dout_full)

            out_ring = ring_flash_attn(ql, kl, vl, causal=causal, group=dist.group.WORLD)
            out_ring.backward(dol)

            # Reference on full seq, then slice.
            qf = q_full.detach().clone().requires_grad_(True)
            kf = k_full.detach().clone().requires_grad_(True)
            vf = v_full.detach().clone().requires_grad_(True)
            out_ref = flash_attn_func(qf, kf, vf, causal=causal)
            out_ref.backward(dout_full)

            def _md(a, b):
                return (a.float() - b.float()).abs().max().item()

            diffs = (
                _md(out_ring, torch.chunk(out_ref, world_size, dim=1)[rank]),
                _md(ql.grad, torch.chunk(qf.grad, world_size, dim=1)[rank]),
                _md(kl.grad, torch.chunk(kf.grad, world_size, dim=1)[rank]),
                _md(vl.grad, torch.chunk(vf.grad, world_size, dim=1)[rank]),
            )
            # Wide atol: bf16 flash-attn numerics differ under ring reordering.
            ok_local = all(d <= 5e-2 for d in diffs)
            ok = torch.tensor([int(ok_local)], device=device, dtype=torch.int32)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if rank == 0:
                print(f"[correctness causal={causal!s:<5}] "
                      f"out={diffs[0]:.2e} dq={diffs[1]:.2e} "
                      f"dk={diffs[2]:.2e} dv={diffs[3]:.2e} "
                      f"ok={bool(ok.item())}", flush=True)

        # ---------- Zigzag: informational (needs zigzag-laid-out inputs) ----------
        if world_size >= 2:
            def _zz(x):
                c = x.chunk(2 * world_size, dim=1)
                return torch.cat([c[rank], c[2 * world_size - 1 - rank]], dim=1).contiguous()
            q_zz = _zz(q_full).detach().clone().requires_grad_(True)
            k_zz = _zz(k_full).detach().clone().requires_grad_(True)
            v_zz = _zz(v_full).detach().clone().requires_grad_(True)
            out_zz = ring_flash_attn(q_zz, k_zz, v_zz,
                                     causal=True, zigzag=True, group=dist.group.WORLD)
            if rank == 0:
                print(f"[zigzag causal]      out shape={tuple(out_zz.shape)} "
                      f"dtype={out_zz.dtype}", flush=True)

        # ---------- Latency ----------
        def _bench(fn, warmup=3, iters=10):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / iters

        ql = _chunk_local(q_full)
        kl = _chunk_local(k_full)
        vl = _chunk_local(v_full)
        dol = _chunk_local(dout_full)

        def fwd_only():
            with torch.no_grad():
                ring_flash_attn(ql, kl, vl, causal=True, group=dist.group.WORLD)

        def fwd_bwd():
            qa = ql.detach().clone().requires_grad_(True)
            ka = kl.detach().clone().requires_grad_(True)
            va = vl.detach().clone().requires_grad_(True)
            o = ring_flash_attn(qa, ka, va, causal=True, group=dist.group.WORLD)
            o.backward(dol)

        ms_fwd = _bench(fwd_only)
        ms_fwbw = _bench(fwd_bwd)

        # Worst-case wall-clock across ranks.
        t = torch.tensor([ms_fwd, ms_fwbw], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"[latency ws={world_size}] "
                  f"fwd={t[0].item():.2f}ms  fwbw={t[1].item():.2f}ms  "
                  f"(B={B} S_local={S_full // world_size} H_q={H_q} H_kv={H_kv} D={D})",
                  flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _demo()
