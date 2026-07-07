# Optimized Ring Flash Attention

On 8×H100 with bf16 causal workloads, this drop-in replacement of the open-source baseline `ring_flash_attn.py` runs up to **~3.0×** faster forward and up to **~2.6×** faster fwd+bwd on **short-sequence shards**, where communication and kernel-launch overhead dominate. As the per-shard sequence length grows and attention compute dominates, the speedup tapers toward **~1.1×**. Behavior is 100% compatible, GQA is supported natively, and `head_dim` up to **256** is supported.

---

## 1. Ring Attention Primer

Ring attention shards the sequence dimension into W chunks across W GPUs. Each rank holds one Q/K/V slice; K/V then rotate around a ring for W-1 hops. At each step every rank runs one attention over its local Q and the current K/V, merges the partial output into an accumulator via **online softmax**, and non-blockingly forwards K/V to the next rank while receiving the next slice from the previous rank. After W steps every Q has seen every K/V — equivalent to full-sequence attention.

```
for step in range(W):
    if step + 1 < W: async send(K,V → next);  recv(K,V ← prev)
    block_out, block_lse = attn(Q_local, K_cur, V_cur)
    out_acc, lse_acc = online_softmax_merge(out_acc, lse_acc, block_out, block_lse)
    if step + 1 < W: wait()
```

Backward is analogous: K/V stay local, dK/dV accumulate around the ring, dQ accumulates in place.

**Reference implementations of the underlying method (easiest → hardest):**
- `ring_attn_demo_code.py` — single-process 8-rank simulation with a hand-derived forward + backward
- `ring_attn.py` — real NCCL-distributed pure-PyTorch version (no flash-attn, plain matmul)
- `ring_attention_explained.html` — detailed derivation with diagrams

---

## 2. Usage

`optimized_ring_flash_attn.py` is our deeply-optimized **fast ring flash attention** implementation, built on top of the open-source [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention). It fuses Triton kernels, packs P2P communication, reuses workspace buffers, and keeps softmax state in fp32 across ring steps. On 8×H100 it delivers up to ~3.0× forward and up to ~2.6× fwd+bwd speedup over baseline on short-sequence shards (tapering to ~1.1× as sequences grow), is 100% API-compatible, and natively supports GQA (see §3 and §4 for details).

A **standalone flash+merge-only variant** is also provided as `ring_flash_attn_flash_merge.py` — same API, but with the hand-written native-Triton attention kernels and the dispatcher stripped out (every step uses flash-attn's cutlass kernel + the fused Triton merge). It is validated bit-identical to the full file's `flash_triton_merge` path (see §4.4).

**User API:**

```python
import torch.distributed as dist
from optimized_ring_flash_attn import ring_flash_attn

# q: [B, S_local, H_q,  D]   bf16 / fp16
# k: [B, S_local, H_kv, D]   H_kv may be < H_q (GQA)
# v: [B, S_local, H_kv, D]
out = ring_flash_attn(
    q, k, v,
    softmax_scale=None,      # defaults to 1/sqrt(D)
    causal=True,
    group=None,              # dist.ProcessGroup, defaults to WORLD
)
```

Constraints: `H_q % H_kv == 0`, `head_dim ≤ 256` (`head_dim > 128` runs on the flash+merge path; `head_dim ≤ 128` can use the native Triton path). `ring_flash_attn` is a standard `torch.autograd.Function` — drop it into any training graph.

**Run the tests (correctness + speedup + ablation):**

```bash
torchrun --nproc_per_node=8 --standalone test_optimized_ring_flash_attn.py
# ablation only:
SKIP_CORRECTNESS=1 SKIP_SPEEDUP=1 torchrun --nproc_per_node=8 --standalone \
    test_optimized_ring_flash_attn.py
```

**Environment knobs:**
| Variable | Effect |
|---|---|
| `OPTIMIZED_RING_FORWARD_IMPL` | `auto` / `triton_native` / `flash_triton_merge` — force forward impl |
| `OPTIMIZED_RING_BACKWARD_IMPL` | `auto` / `triton_native` / `flash` — force backward impl |
| `OPTIMIZED_RING_BWD_COMM_BF16` | `1` → send dk/dv in bf16 (default fp32) |

**Dependencies:** `torch (with distributed)` · `flash-attn` · `triton`

---

## 3. Optimizations

`ring_flash_attn.py` is ported directly from [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention). On top of it, `optimized_ring_flash_attn.py` applies:

| # | Optimization | Baseline | Optimized |
|---|---|---|---|
| a | **online-softmax merge** | `sigmoid/logsigmoid` combo, 4+ elementwise kernels per merge; out_acc casts bf16⇄fp32 every step | one fused Triton kernel does max-scale-add-log; out_acc stays fp32 throughout. `BLOCK_M` adapts to `head_dim` so the fp32 register tile never spills (see §4.4) |
| b | **packed K/V P2P** | 4 P2P ops per hop (K/V send + recv separately) | packed into `[2,B,S,H_kv,D]` → **2 P2P ops per hop** |
| c | **workspace cache** | `torch.empty_like(k)` per hop for the recv buffer | single-slot cache keyed by shape+dtype; 100% hit rate in training loops |
| d | **double-buffered comm** | single buffer, no comm/compute overlap | two recv buffers alternated; next-hop recv fully overlaps current-hop compute |
| e | **native Triton fwd kernel** | flash-attn every step; `(m,l,acc)` round-trips bf16⇄fp32 through HBM | hand-written Triton flash-attn; `(m,l,acc)` **stay fp32 across ring steps**; uses `exp2` instead of `exp` |
| f | **fused grad accumulation (bwd)** | per hop `block_dk.to(fp32) + dk_prev.add_()`: 4 launches + 2 intermediate tensors | one Triton kernel fuses dtype cast + add + store |
| g | **GQA handling** | works via flash-attn, but no GQA-specific kernels; dk/dv comm is full-size | native Triton kernels are GQA-aware (iterate Q-heads per KV-head); packed dk/dv comm is KV-sized |
| h | **adaptive dispatcher** | one hard-coded impl | picks between `triton_native` and `flash_triton_merge` based on `(head_dim, seq_len, num_heads, world_size)` |

---

## 4. Performance

Test rig: 8×H100 80GB HBM3 · bf16 · causal=True · MHA · slowest-rank wall-clock (`cuda_bench`, warmup=3, iters=10, max-reduced across ranks). Generated by `test_optimized_ring_flash_attn.py`.

### 4.1 Correctness

24 configs (`head_dim ∈ {64,128}` × `(H_q,H_kv) ∈ {(8,8),(16,2)}` × `S_local ∈ {1024,2048,4096}` × `causal ∈ {False,True}`) **all pass**. `max_diff` of out/dq/dk against a single-GPU flash-attn reference sits stably in `1e-3 ~ 3e-2` (bf16); GQA+causal dv peaks at `6.25e-2`, which is normal bf16 accumulation noise (both the ring impl and baseline agree with each other far more tightly than either agrees with the single-kernel reference). `head_dim=256` is separately validated correct on the flash+merge path.

### 4.2 Baseline vs Optimized speedup

**⭐ Emphasized configuration: ws=8 · head_dim=128**

| S_local | H | base_fwd (ms) | opt_fwd (ms) | **fwd speedup** | base_fbw (ms) | opt_fbw (ms) | **fbw speedup** |
|---:|---:|---:|---:|:---:|---:|---:|:---:|
| 1024 | 8 | 1.64 | 0.55 | **2.97×** | 5.89 | 4.28 | 1.38× |
| 2048 | 8 | 1.59 | 1.17 | 1.35× | 6.19 | 4.36 | **1.42×** |
| 4096 | 8 | 3.03 | 2.47 | 1.23× | 9.39 | 8.86 | 1.06× |
| 8192 | 8 | 7.93 | 6.95 | 1.14× | 28.31 | 27.47 | 1.03× |
| 1024 | 16 | 1.69 | 1.07 | 1.58× | 6.48 | 4.62 | 1.40× |
| 2048 | 16 | 2.44 | 2.14 | 1.14× | 8.33 | 7.93 | 1.05× |
| 4096 | 16 | 5.76 | 5.24 | 1.10× | 17.61 | 17.34 | 1.02× |
| 8192 | 16 | 16.46 | 14.64 | 1.12× | 55.09 | 52.76 | 1.04× |

**Key observations:**
- **Big fwd win on short sequences (2.97× @ S=1024)** — comm/kernel-launch overhead dominates, and packed P2P + workspace caching + the native fwd kernel absorb all of it.
- **fwd+bwd gains are also concentrated on short sequences** (up to 1.42× here; up to **2.62×** for `head_dim=64` below). As `S_local` grows the workload becomes compute-bound (flash-attn cutlass kernels dominate), so the fixed comm/merge savings shrink to ~1.0–1.1×.
- **Note:** an earlier version of this README reported a 3.09× long-sequence fbw. That came from a one-off baseline anomaly (baseline fbw spiked to ~84 ms at S=8192); it does **not** reproduce — the current measurement is 28.3/27.5 = 1.03×. Long sequences are compute-bound and near parity.

**head_dim=64 reference table (ws=8):**

| S_local | H | fwd speedup | fbw speedup | | S_local | H | fwd speedup | fbw speedup |
|---:|---:|:---:|:---:|:---:|---:|---:|:---:|:---:|
| 1024 | 8 | 2.79× | 2.62× | | 1024 | 16 | 2.35× | 2.27× |
| 2048 | 8 | 2.32× | 2.16× | | 2048 | 16 | 1.49× | 1.29× |
| 4096 | 8 | 1.21× | 1.03× | | 4096 | 16 | 1.09× | 1.01× |
| 8192 | 8 | 1.12× | 1.04× | | 8192 | 16 | 1.09× | 1.02× |

### 4.3 Ablation study (forward, ws=8, causal=True)

Optimizations are added cumulatively to isolate each contribution:

- **A0**: baseline (sigmoid/logsigmoid merge + 4 P2P/hop + empty_like/hop)
- **A1**: A0 + packed P2P (2 ops/hop) + workspace cache
- **A2**: A1 + fused Triton merge kernel
- **A3**: A2 + native Triton fwd (fp32 handoff, eliminates bf16⇄fp32 round-trips)

**⭐ head_dim=128:**

| S_local | H | A0 (ms) | A1 vs A0 | A2 vs A0 | **A3 vs A0** |
|---:|---:|---:|:---:|:---:|:---:|
| 2048 | 8 | 1.88 | 1.28× | 1.25× | **1.66×** |
| 4096 | 8 | 3.10 | 1.05× | 1.06× | **1.24×** |
| 8192 | 8 | 8.03 | 1.03× | 1.03× | **1.15×** |
| 2048 | 16 | 2.43 | 1.06× | 1.07× | **1.15×** |
| 4096 | 16 | 5.66 | 1.04× | 1.03× | **1.08×** |
| 8192 | 16 | 16.49 | 1.01× | 1.02× | **1.13×** |

**head_dim=64:**

| S_local | H | A0 (ms) | A1 vs A0 | A2 vs A0 | **A3 vs A0** |
|---:|---:|---:|:---:|:---:|:---:|
| 2048 | 8 | 1.60 | 1.28× | 1.39× | **2.24×** |
| 4096 | 8 | 1.76 | 1.00× | 1.05× | **1.16×** |
| 8192 | 8 | 4.57 | 1.03× | 1.02× | **1.10×** |
| 2048 | 16 | 1.57 | 1.27× | 1.30× | 1.29× |
| 4096 | 16 | 3.20 | 1.05× | 1.05× | 1.09× |
| 8192 | 16 | 9.31 | 1.03× | 1.01× | 1.09× |

**Takeaways:**
- **A1 (packed comm + workspace cache)** — dominates on short sequences (~25-30% gain); the win is a fixed O(W) overhead reduction, so it dilutes as compute grows.
- **A2 (triton merge kernel)** — near-zero raw speed gain by itself; its real value is keeping out_acc in fp32 so the native fwd path can plug in cleanly (and, for `head_dim > 128`, its adaptive `BLOCK_M` avoids a large-tile register spill — see §4.4).
- **A3 (native triton fwd)** — the single biggest forward accelerator, especially `head_dim=64` short sequences (2.24× cumulative); the fp32 state carried across ring steps saves per-step bf16⇄fp32 HBM bandwidth. On long sequences it settles to a steady ~5–15%.
- **Backward speedups (§4.2)** come from fused grad-accum + packed dk/dv comm + fp32 dq accumulator, and — like forward — are largest on short sequences.

### 4.4 `head_dim=256` and the merge-kernel tiling fix

`head_dim > 128` runs on the flash+merge path. The fused merge kernel originally hard-coded `BLOCK_M=128`, so at `D=256` each program materialized a `[128, 256]` fp32 tile that overflowed the register file and spilled to local memory — making the merge ~4.5× slower and turning `D=256` at small `world_size` into a net regression vs baseline. Adapting `BLOCK_M` to `head_dim` (targeting ~8192 fp32 elems/program: hd64→128, hd128→64, hd256→32) removes the spill.

Independently benchmarked (bf16, causal, `D=256`, slowest-rank median), the fix turns the former regression into a win: at ws=4 `D=256` forward went from 0.82× → **1.10×**; ws=8 → **1.28×**. `num_warps=4` was separately swept `{1,2,4,8,16}` and is already optimal (the merge is HBM-bandwidth-bound; 8/16 tie within noise), so it is left at 4.

### 4.5 Flash+merge standalone file equivalence

`ring_flash_attn_flash_merge.py` (the stripped flash+merge-only variant) was validated against the full file forced to `flash_triton_merge`, across `head_dim ∈ {64,128,256}` × `world_size ∈ {2,4,8}` × `seq_len`:
- **Deterministic shapes** (e.g. `D128` non-causal, `D256`): forward is **bit-identical** (max|diff| = 0).
- **`D128` causal**: flash-attn's bf16 kernels are slightly non-deterministic run-to-run (~1e-2); the cross-file difference equals each file's own run-to-run difference — i.e. no code divergence.
- **Speed**: faster than baseline across the whole grid (ws=2 → 1.04–1.11×, ws=4 → 1.05–1.34×, ws=8 → 1.09–1.49×).

---
---

# 中文版本 (Chinese)

# Optimized Ring Flash Attention

在 8×H100 上以 bf16 causal 场景为基准，相对开源 baseline `ring_flash_attn.py`，在**短序列分片**（通信/kernel-launch 开销占主导）上 forward 最高 **~3.0×**、fwd+bwd 最高 **~2.6×**；随着每卡序列变长、attention 计算占主导，加速逐渐收敛到 **~1.1×**。行为 100% 兼容，原生支持 **GQA**，`head_dim` 支持到 **256**。

---

## 1. Ring Attention 原理简介

Ring attention 把序列维切成 W 段分散到 W 张卡。每张卡持有 Q/K/V 的一段；计算时 K/V 沿环状通信路径转 W-1 步：每步用本地 Q 和当前 K/V 算一次 attention，通过 **online softmax** 把每步的部分输出并入累加器，同时非阻塞地把 K/V 发给下一张卡、从上一张卡收下一份。W 步后每个 rank 的 Q 已看过全部 K/V，等价于全序列 attention。

```
for step in range(W):
    if step + 1 < W: async send(K,V → next);  recv(K,V ← prev)
    block_out, block_lse = attn(Q_local, K_cur, V_cur)
    out_acc, lse_acc = online_softmax_merge(out_acc, lse_acc, block_out, block_lse)
    if step + 1 < W: wait()
```

Backward 类似，K/V 保持本地不动，dK/dV 沿环累加、dQ 就地累积。

**方法原理实现（从易到难）：**
- `ring_attn_demo_code.py` — 单进程 8-rank 模拟，含手写前后向数学推导
- `ring_attn.py` — NCCL 分布式 PyTorch 版（无 flash-attn，纯 matmul）
- `ring_attention_explained.html` — 原理详解与图示

---

## 2. 使用方式

`optimized_ring_flash_attn.py` 是我们对开源 [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) 做了深度优化的 **fast ring flash attention** 实现：融合 Triton kernel、打包 P2P 通信、workspace 复用、跨 ring step 保持 fp32 状态等，在 8×H100 上短序列分片 forward 最高 ~3.0×、fwd+bwd 最高 ~2.6×（随序列变长收敛到 ~1.1×）；行为 100% 兼容原 API 并原生支持 GQA（详见 §3、§4）。

另外提供一个**只含 flash+merge 的独立版本** `ring_flash_attn_flash_merge.py` —— API 相同，但去掉了手写 native-Triton attention 内核与 dispatcher（每步都用 flash-attn cutlass kernel + 融合 Triton merge）。已验证与完整文件的 `flash_triton_merge` 路径逐元素等价（见 §4.5）。

**用户 API：**

```python
import torch.distributed as dist
from optimized_ring_flash_attn import ring_flash_attn

# q: [B, S_local, H_q,  D]   bf16 / fp16
# k: [B, S_local, H_kv, D]   H_kv 可 < H_q（GQA）
# v: [B, S_local, H_kv, D]
out = ring_flash_attn(
    q, k, v,
    softmax_scale=None,      # 默认 1/sqrt(D)
    causal=True,
    group=None,              # dist.ProcessGroup，默认 WORLD
)
```

限制：`H_q % H_kv == 0`，`head_dim ≤ 256`（`head_dim > 128` 走 flash+merge 路径，`head_dim ≤ 128` 可用 native Triton 路径）；`ring_flash_attn` 是标准 `torch.autograd.Function`，可以直接放到训练图里。

**跑测试（正确性 + 加速比 + ablation）：**

```bash
torchrun --nproc_per_node=8 --standalone test_optimized_ring_flash_attn.py
# 只跑 ablation：
SKIP_CORRECTNESS=1 SKIP_SPEEDUP=1 torchrun --nproc_per_node=8 --standalone \
    test_optimized_ring_flash_attn.py
```

**可调环境变量：**
| 变量 | 作用 |
|---|---|
| `OPTIMIZED_RING_FORWARD_IMPL` | `auto` / `triton_native` / `flash_triton_merge` — 强制 forward 实现 |
| `OPTIMIZED_RING_BACKWARD_IMPL` | `auto` / `triton_native` / `flash` — 强制 backward 实现 |
| `OPTIMIZED_RING_BWD_COMM_BF16` | `1` → dk/dv 通信用 bf16（默认 fp32）|

**依赖：** `torch (with distributed)` · `flash-attn` · `triton`

---

## 3. 优化点

`ring_flash_attn.py` 直接移植自 [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention)。在此基础上，`optimized_ring_flash_attn.py` 做了以下优化：

| # | 优化项 | Baseline | 优化后 |
|---|---|---|---|
| a | **online-softmax merge** | `sigmoid/logsigmoid` 组合，4+ elementwise kernel/次；out_acc 每步 bf16⇄fp32 | 一个 fused Triton kernel 完成 max-scale-add-log；out_acc 全程 fp32。`BLOCK_M` 随 `head_dim` 自适应，避免 fp32 register tile 溢出（见 §4.4）|
| b | **K/V P2P 打包** | 4 个 P2P op/hop（K/V 各 send/recv） | 打包成 `[2,B,S,H_kv,D]` → **2 个 P2P op/hop** |
| c | **Workspace 缓存** | 每 hop `torch.empty_like(k)` 分配 recv buffer | 按 shape+dtype 单槽缓存，训练循环命中率 100% |
| d | **双缓冲通信** | 单缓冲，无 comm/compute overlap | 两份 recv buffer 交替使用，下一 hop recv 与本 hop compute 完全重叠 |
| e | **Native Triton fwd 内核** | 每步调 flash-attn，`(m,l,acc)` 每步 bf16⇄fp32 往返 HBM | 手写 Triton flash-attn，`(m,l,acc)` **跨 ring step 保持 fp32**；用 `exp2` 而非 `exp` |
| f | **融合 grad 累加（bwd）** | 每 hop `block_dk.to(fp32) + dk_prev.add_()`：4 launch + 2 中间 tensor | 单个 Triton kernel 融合 dtype cast + add + 写回 |
| g | **GQA 处理** | 经 flash-attn 支持，但无 GQA 专用 kernel，dk/dv 通信为全尺寸 | native Triton kernel 是 GQA 感知的（每个 KV-head 遍历其 Q-heads）；打包 dk/dv 通信按 KV 尺寸 |
| h | **自适应 dispatcher** | 单一实现 | 按 `(head_dim, seq_len, num_heads, world_size)` 自动在 `triton_native` / `flash_triton_merge` 间切换 |

---

## 4. 性能数据

测试机器：8×H100 80GB HBM3 · bf16 · causal=True · MHA · 报告最慢 rank wall-clock（`cuda_bench`，warmup=3、iters=10，跨 rank 取 max）。由 `test_optimized_ring_flash_attn.py` 生成。

### 4.1 正确性

24 个 config（`head_dim ∈ {64,128}` × `(H_q,H_kv) ∈ {(8,8),(16,2)}` × `S_local ∈ {1024,2048,4096}` × causal ∈ {False,True}）**全部通过**。out/dq/dk 与单卡 flash-attn 参考实现的 bf16 `max_diff` 稳定在 `1e-3 ~ 3e-2`；GQA+causal 的 dv 最大到 `6.25e-2`，属 bf16 累加噪声（ring 实现与 baseline 彼此的吻合度远高于二者对单核参考的吻合度）。`head_dim=256` 在 flash+merge 路径上单独验证正确。

### 4.2 Baseline vs Optimized 加速比

**⭐ 强调场景：ws=8 · head_dim=128**

| S_local | H | base_fwd (ms) | opt_fwd (ms) | **fwd 加速** | base_fbw (ms) | opt_fbw (ms) | **fbw 加速** |
|---:|---:|---:|---:|:---:|---:|---:|:---:|
| 1024 | 8 | 1.64 | 0.55 | **2.97×** | 5.89 | 4.28 | 1.38× |
| 2048 | 8 | 1.59 | 1.17 | 1.35× | 6.19 | 4.36 | **1.42×** |
| 4096 | 8 | 3.03 | 2.47 | 1.23× | 9.39 | 8.86 | 1.06× |
| 8192 | 8 | 7.93 | 6.95 | 1.14× | 28.31 | 27.47 | 1.03× |
| 1024 | 16 | 1.69 | 1.07 | 1.58× | 6.48 | 4.62 | 1.40× |
| 2048 | 16 | 2.44 | 2.14 | 1.14× | 8.33 | 7.93 | 1.05× |
| 4096 | 16 | 5.76 | 5.24 | 1.10× | 17.61 | 17.34 | 1.02× |
| 8192 | 16 | 16.46 | 14.64 | 1.12× | 55.09 | 52.76 | 1.04× |

**关键观察：**
- **短序列 fwd 大赢 (2.97× @ S=1024)** — 通信/kernel launch 占比高，打包 P2P + workspace 缓存 + native fwd kernel 全部命中。
- **fwd+bwd 收益同样集中在短序列**（此处最高 1.42×；`head_dim=64` 下最高 **2.62×**，见下表）。S_local 变大后进入 compute-bound（flash-attn cutlass 内核主导），固定的通信/merge 收益被稀释到 ~1.0–1.1×。
- **勘误：** 本 README 早期版本报告过 3.09× 的长序列 fbw，那来自一次性的 baseline 异常（S=8192 时 baseline fbw 飙到 ~84ms），**无法复现** —— 当前测量为 28.3/27.5 = 1.03×。长序列 compute-bound，基本持平。

**head_dim=64 对照 (ws=8)：**

| S_local | H | fwd 加速 | fwbw 加速 | | S_local | H | fwd 加速 | fwbw 加速 |
|---:|---:|:---:|:---:|:---:|---:|---:|:---:|:---:|
| 1024 | 8 | 2.79× | 2.62× | | 1024 | 16 | 2.35× | 2.27× |
| 2048 | 8 | 2.32× | 2.16× | | 2048 | 16 | 1.49× | 1.29× |
| 4096 | 8 | 1.21× | 1.03× | | 4096 | 16 | 1.09× | 1.01× |
| 8192 | 8 | 1.12× | 1.04× | | 8192 | 16 | 1.09× | 1.02× |

### 4.3 Ablation Study（forward, ws=8, causal=True）

按累加顺序拆开每个优化的边际贡献：

- **A0**: baseline（sigmoid/logsigmoid merge + 4 P2P/hop + empty_like/hop）
- **A1**: A0 + 打包 P2P (2 ops/hop) + workspace 缓存
- **A2**: A1 + fused Triton merge kernel
- **A3**: A2 + native Triton fwd（fp32 handoff，消除 bf16⇄fp32 往返）

**⭐ head_dim=128：**

| S_local | H | A0 (ms) | A1 vs A0 | A2 vs A0 | **A3 vs A0** |
|---:|---:|---:|:---:|:---:|:---:|
| 2048 | 8 | 1.88 | 1.28× | 1.25× | **1.66×** |
| 4096 | 8 | 3.10 | 1.05× | 1.06× | **1.24×** |
| 8192 | 8 | 8.03 | 1.03× | 1.03× | **1.15×** |
| 2048 | 16 | 2.43 | 1.06× | 1.07× | **1.15×** |
| 4096 | 16 | 5.66 | 1.04× | 1.03× | **1.08×** |
| 8192 | 16 | 16.49 | 1.01× | 1.02× | **1.13×** |

**head_dim=64：**

| S_local | H | A0 (ms) | A1 vs A0 | A2 vs A0 | **A3 vs A0** |
|---:|---:|---:|:---:|:---:|:---:|
| 2048 | 8 | 1.60 | 1.28× | 1.39× | **2.24×** |
| 4096 | 8 | 1.76 | 1.00× | 1.05× | **1.16×** |
| 8192 | 8 | 4.57 | 1.03× | 1.02× | **1.10×** |
| 2048 | 16 | 1.57 | 1.27× | 1.30× | 1.29× |
| 4096 | 16 | 3.20 | 1.05× | 1.05× | 1.09× |
| 8192 | 16 | 9.31 | 1.03× | 1.01× | 1.09× |

**结论：**
- **A1 (packed comm + workspace cache)** — 短序列的主要来源 (~25-30%)；开销固定 O(W)，S 变大后被 compute 稀释。
- **A2 (triton merge kernel)** — 纯速度增益接近 0；价值在于保证 out_acc 全程 fp32，与 native path 拼接（且 `head_dim > 128` 时其自适应 `BLOCK_M` 避免了大 tile 的寄存器溢出，见 §4.4）。
- **A3 (native triton fwd)** — 最大的单项 forward 加速，尤其 `head_dim=64` 短序列（累计 2.24×）；跨 step 的 fp32 状态省下每步 bf16⇄fp32 的 HBM 带宽。长序列稳定在 ~5–15%。
- **Backward 的加速（§4.2）** 来自 fused grad-accum + packed dk/dv 通信 + fp32 dq 累加器，与 forward 一样在短序列上最大。

### 4.4 `head_dim=256` 与 merge 内核 tiling 修复

`head_dim > 128` 走 flash+merge 路径。融合 merge kernel 原本写死 `BLOCK_M=128`，`D=256` 时每个 program 要在寄存器里放 `[128, 256]` 的 fp32 tile，超出寄存器文件、溢出到 local memory —— 使 merge 慢约 4.5×，并把小 `world_size` 的 `D=256` 变成相对 baseline 的净退化。将 `BLOCK_M` 随 `head_dim` 自适应（每 program 目标 ~8192 个 fp32：hd64→128、hd128→64、hd256→32）即可消除溢出。

独立基准（bf16、causal、`D=256`、最慢 rank median）显示修复把退化变成净赢：ws=4 `D=256` forward 从 0.82× → **1.10×**；ws=8 → **1.28×**。`num_warps` 另外扫过 `{1,2,4,8,16}`，4 已是最优（merge 是 HBM 带宽瓶颈，8/16 与 4 的差异在噪声内），故保持 4。

### 4.5 flash+merge 独立文件等价性

`ring_flash_attn_flash_merge.py`（精简的 flash+merge-only 版本）已与完整文件强制 `flash_triton_merge` 对拍，覆盖 `head_dim ∈ {64,128,256}` × `world_size ∈ {2,4,8}` × `seq_len`：
- **确定性 shape**（如 `D128` non-causal、`D256`）：forward **逐元素完全一致**（max|diff| = 0）。
- **`D128` causal**：flash-attn 的 bf16 内核本身 run-to-run 有 ~1e-2 的非确定性；跨文件差异等于各自 run-to-run 的差异 —— 即无代码层面分歧。
- **速度**：全网格都快于 baseline（ws=2 → 1.04–1.11×，ws=4 → 1.05–1.34×，ws=8 → 1.09–1.49×）。
