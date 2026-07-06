# Optimized Ring Flash Attention

On 8×H100 with bf16 causal workloads, this drop-in replacement of the open-source baseline `ring_flash_attn.py` achieves up to **3.0×** forward and up to **3.1×** fwd+bwd speedup on long sequences; behavior is 100% compatible and GQA is supported natively.

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

`optimized_ring_flash_attn.py` is our deeply-optimized **fast ring flash attention** implementation, built on top of the open-source [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention). It fuses Triton kernels, packs P2P communication, reuses workspace buffers, and keeps softmax state in fp32 across ring steps. On 8×H100 it delivers up to 3.0× forward and up to 3.1× long-sequence fwd+bwd speedup over baseline, is 100% API-compatible, and natively supports GQA (see §3 and §4 for details).

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

Constraints: `H_q % H_kv == 0`, `head_dim ≤ 128`. `ring_flash_attn` is a standard `torch.autograd.Function` — drop it into any training graph.

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
| a | **online-softmax merge** | `sigmoid/logsigmoid` combo, 4+ elementwise kernels per merge; out_acc casts bf16⇄fp32 every step | one fused Triton kernel does max-scale-add-log; out_acc stays fp32 throughout |
| b | **packed K/V P2P** | 4 P2P ops per hop (K/V send + recv separately) | packed into `[2,B,S,H_kv,D]` → **2 P2P ops per hop** |
| c | **workspace cache** | `torch.empty_like(k)` per hop for the recv buffer | single-slot cache keyed by shape+dtype; 100% hit rate in training loops |
| d | **double-buffered comm** | single buffer, no comm/compute overlap | two recv buffers alternated; next-hop recv fully overlaps current-hop compute |
| e | **native Triton fwd kernel** | flash-attn every step; `(m,l,acc)` round-trips bf16⇄fp32 through HBM | hand-written Triton flash-attn; `(m,l,acc)` **stay fp32 across ring steps**; uses `exp2` instead of `exp` |
| f | **fused grad accumulation (bwd)** | per hop `block_dk.to(fp32) + dk_prev.add_()`: 4 launches + 2 intermediate tensors | one Triton kernel fuses dtype cast + add + store |
| g | **GQA support** | assumes `K.shape == Q.shape` | supports any `H_q % H_kv == 0` |
| h | **adaptive dispatcher** | one hard-coded impl | picks between `triton_native` and `flash_triton_merge` based on `(head_dim, seq_len, num_heads, world_size)` |

---

## 4. Performance

Test rig: 8×H100 80GB HBM3 · bf16 · causal=True · MHA · reporting slowest-rank wall-clock.

### 4.1 Correctness

24 configs (`head_dim ∈ {64,128}` × `(H_q,H_kv) ∈ {(8,8),(16,2)}` × `S_local ∈ {1024,2048,4096}` × `causal ∈ {False,True}`) **all pass**. `max_diff` of out/dq/dk against a single-GPU flash-attn reference sits stably in `1e-3 ~ 3e-2` (bf16); GQA+causal dv peaks at `6.25e-2`, which is normal bf16 accumulation noise.

### 4.2 Baseline vs Optimized speedup

**⭐ Emphasized configuration: ws=8 · head_dim=128**

| S_local | H | base_fwd (ms) | opt_fwd (ms) | **fwd speedup** | base_fbw (ms) | opt_fbw (ms) | **fbw speedup** |
|---:|---:|---:|---:|:---:|---:|---:|:---:|
| 1024 | 8 | 1.57 | 0.53 | **2.96×** | 5.27 | 3.87 | 1.36× |
| 2048 | 8 | 1.57 | 1.14 | 1.38× | 5.17 | 4.09 | 1.26× |
| 4096 | 8 | 3.10 | 2.53 | 1.23× | 9.42 | 8.76 | 1.08× |
| 8192 | 8 | 7.99 | 6.93 | 1.15× | 84.61 | 27.41 | **3.09×** |
| 1024 | 16 | 1.78 | 1.00 | 1.78× | 5.44 | 3.93 | 1.38× |
| 2048 | 16 | 2.53 | 2.15 | 1.18× | 8.30 | 7.96 | 1.04× |
| 4096 | 16 | 5.70 | 5.23 | 1.09× | 17.67 | 17.25 | 1.02× |
| 8192 | 16 | 16.39 | 14.71 | 1.11× | 87.80 | 52.96 | **1.66×** |

**Key observations:**
- **Big fwd win on short sequences (2.96× @ S=1024)** — comm/kernel-launch overhead dominates, and packed P2P + workspace caching absorb all of it.
- **Explosive fwbw gain on long sequences (3.09× @ S=8192, H=8)** — baseline backward produces a stream of fp32 intermediates for chained `add` per hop; this cost scales with S and H_kv. Our fused grad-accum Triton kernel eliminates every KV-elementwise intermediate.

**head_dim=64 reference table:**

| S_local | H | fwd speedup | fbw speedup | | S_local | H | fwd speedup | fbw speedup |
|---:|---:|:---:|:---:|:---:|---:|---:|:---:|:---:|
| 1024 | 8 | 2.78× | 2.32× | | 1024 | 16 | 2.92× | 2.20× |
| 2048 | 8 | 2.29× | 1.97× | | 2048 | 16 | 1.33× | 1.13× |
| 4096 | 8 | 1.22× | 0.94× | | 4096 | 16 | 1.09× | 1.00× |
| 8192 | 8 | 1.11× | 1.05× | | 8192 | 16 | 1.10× | 1.35× |

### 4.3 Ablation study (forward, ws=8, causal=True)

Optimizations are added cumulatively to isolate each contribution:

- **A0**: baseline (sigmoid/logsigmoid merge + 4 P2P/hop + empty_like/hop)
- **A1**: A0 + packed P2P (2 ops/hop) + workspace cache
- **A2**: A1 + fused Triton merge kernel
- **A3**: A2 + native Triton fwd (fp32 handoff, eliminates bf16⇄fp32 round-trips)

**⭐ head_dim=128:**

| S_local | H | A0 (ms) | A1 vs A0 | A2 vs A0 | **A3 vs A0** |
|---:|---:|---:|:---:|:---:|:---:|
| 2048 | 8 | 1.52 | 1.25× | 1.24× | **1.34×** |
| 4096 | 8 | 3.11 | 1.05× | 1.07× | **1.23×** |
| 8192 | 8 | 7.98 | 1.02× | 1.02× | **1.16×** |
| 2048 | 16 | 2.44 | 1.06× | 1.07× | **1.12×** |
| 4096 | 16 | 5.68 | 1.05× | 1.04× | **1.08×** |
| 8192 | 16 | 16.39 | 1.01× | 1.01× | **1.12×** |

**head_dim=64:**

| S_local | H | A0 (ms) | A1 vs A0 | A2 vs A0 | **A3 vs A0** |
|---:|---:|---:|:---:|:---:|:---:|
| 2048 | 8 | 1.49 | 1.22× | 1.21× | **1.93×** |
| 4096 | 8 | 1.75 | 1.07× | 1.09× | **1.14×** |
| 8192 | 8 | 4.60 | 1.02× | 1.03× | **1.11×** |
| 2048 | 16 | 1.54 | 1.29× | 1.19× | 1.29× |
| 4096 | 16 | 3.24 | 1.07× | 1.05× | 1.09× |
| 8192 | 16 | 9.30 | 1.02× | 1.01× | 1.09× |

**Takeaways:**
- **A1 (packed comm + workspace cache)** — dominates on short sequences (~20-30% gain); the win is a fixed O(W) overhead reduction, so it dilutes as compute grows.
- **A2 (triton merge kernel)** — near-zero raw speed gain by itself; its real value is keeping out_acc in fp32 so the native fwd path can plug in cleanly.
- **A3 (native triton fwd)** — a steady **5–15% marginal gain and the only reliable long-sequence forward accelerator**; the fp32 state carried across ring steps saves per-step bf16⇄fp32 HBM bandwidth.
- **The huge backward speedup (§4.2, 3.09× on long seq)** is not visible in this forward-only ablation. It comes from fused grad-accum + packed dk/dv comm + fp32 dq accumulator.

---
---

# 中文版本 (Chinese)

# Optimized Ring Flash Attention

在 8×H100 上以 bf16 causal 场景为基准，相对开源 baseline `ring_flash_attn.py` 的 forward 最高 **3.0×**、fwd+bwd 长序列最高 **3.1×**；行为 100% 兼容且原生支持 **GQA**。

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

`optimized_ring_flash_attn.py` 是我们对开源 [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) 做了深度优化的 **fast ring flash attention** 实现：融合 Triton kernel、打包 P2P 通信、workspace 复用、跨 ring step 保持 fp32 状态等，在 8×H100 上 forward 最高 3.0×、fwd+bwd 长序列最高 3.1× 于 baseline；行为 100% 兼容原 API 并原生支持 GQA（详见 §3、§4）。

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

限制：`H_q % H_kv == 0`，`head_dim ≤ 128`；`ring_flash_attn` 是标准 `torch.autograd.Function`，可以直接放到训练图里。

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
| a | **online-softmax merge** | `sigmoid/logsigmoid` 组合，4+ elementwise kernel/次；out_acc 每步 bf16⇄fp32 | 一个 fused Triton kernel 完成 max-scale-add-log；out_acc 全程 fp32 |
| b | **K/V P2P 打包** | 4 个 P2P op/hop（K/V 各 send/recv） | 打包成 `[2,B,S,H_kv,D]` → **2 个 P2P op/hop** |
| c | **Workspace 缓存** | 每 hop `torch.empty_like(k)` 分配 recv buffer | 按 shape+dtype 单槽缓存，训练循环命中率 100% |
| d | **双缓冲通信** | 单缓冲，无 comm/compute overlap | 两份 recv buffer 交替使用，下一 hop recv 与本 hop compute 完全重叠 |
| e | **Native Triton fwd 内核** | 每步调 flash-attn，`(m,l,acc)` 每步 bf16⇄fp32 往返 HBM | 手写 Triton flash-attn，`(m,l,acc)` **跨 ring step 保持 fp32**；用 `exp2` 而非 `exp` |
| f | **融合 grad 累加（bwd）** | 每 hop `block_dk.to(fp32) + dk_prev.add_()`：4 launch + 2 中间 tensor | 单个 Triton kernel 融合 dtype cast + add + 写回 |
| g | **GQA 支持** | 假设 `K.shape == Q.shape` | 支持任意 `H_q % H_kv == 0` |
| h | **自适应 dispatcher** | 单一实现 | 按 `(head_dim, seq_len, num_heads, world_size)` 自动在 `triton_native` / `flash_triton_merge` 间切换 |

---

## 4. 性能数据

测试机器：8×H100 80GB HBM3 · bf16 · causal=True · MHA · 报告最慢 rank wall-clock。

### 4.1 正确性

24 个 config（`head_dim ∈ {64,128}` × `(H_q,H_kv) ∈ {(8,8),(16,2)}` × `S_local ∈ {1024,2048,4096}` × causal ∈ {False,True}）**全部通过**。out/dq/dk 与单卡 flash-attn 参考实现的 bf16 `max_diff` 稳定在 `1e-3 ~ 3e-2`；GQA+causal 的 dv 最大到 `6.25e-2`，属 bf16 累加噪声。

### 4.2 Baseline vs Optimized 加速比

**⭐ 强调场景：ws=8 · head_dim=128**

| S_local | H | base_fwd (ms) | opt_fwd (ms) | **fwd 加速** | base_fbw (ms) | opt_fbw (ms) | **fbw 加速** |
|---:|---:|---:|---:|:---:|---:|---:|:---:|
| 1024 | 8 | 1.57 | 0.53 | **2.96×** | 5.27 | 3.87 | 1.36× |
| 2048 | 8 | 1.57 | 1.14 | 1.38× | 5.17 | 4.09 | 1.26× |
| 4096 | 8 | 3.10 | 2.53 | 1.23× | 9.42 | 8.76 | 1.08× |
| 8192 | 8 | 7.99 | 6.93 | 1.15× | 84.61 | 27.41 | **3.09×** |
| 1024 | 16 | 1.78 | 1.00 | 1.78× | 5.44 | 3.93 | 1.38× |
| 2048 | 16 | 2.53 | 2.15 | 1.18× | 8.30 | 7.96 | 1.04× |
| 4096 | 16 | 5.70 | 5.23 | 1.09× | 17.67 | 17.25 | 1.02× |
| 8192 | 16 | 16.39 | 14.71 | 1.11× | 87.80 | 52.96 | **1.66×** |

**关键观察：**
- **短序列 fwd 大赢 (2.96× @ S=1024)** — 通信/kernel launch 占比高，打包 + 缓存全部命中
- **长序列 fwbw 收益爆炸 (3.09× @ S=8192, H=8)** — baseline backward 每 hop 都产生 fp32 中间 tensor 做 chained add，S、H_kv 越大越亏；优化版的 fused grad-accum Triton kernel 消掉所有 KV 元素级中间 tensor

**head_dim=64 对照：**

| S_local | H | fwd 加速 | fwbw 加速 | | S_local | H | fwd 加速 | fwbw 加速 |
|---:|---:|:---:|:---:|:---:|---:|---:|:---:|:---:|
| 1024 | 8 | 2.78× | 2.32× | | 1024 | 16 | 2.92× | 2.20× |
| 2048 | 8 | 2.29× | 1.97× | | 2048 | 16 | 1.33× | 1.13× |
| 4096 | 8 | 1.22× | 0.94× | | 4096 | 16 | 1.09× | 1.00× |
| 8192 | 8 | 1.11× | 1.05× | | 8192 | 16 | 1.10× | 1.35× |

### 4.3 Ablation Study（forward, ws=8, causal=True）

按累加顺序拆开每个优化的边际贡献：

- **A0**: baseline（sigmoid/logsigmoid merge + 4 P2P/hop + empty_like/hop）
- **A1**: A0 + 打包 P2P (2 ops/hop) + workspace 缓存
- **A2**: A1 + fused Triton merge kernel
- **A3**: A2 + native Triton fwd（fp32 handoff，消除 bf16⇄fp32 往返）

**⭐ head_dim=128：**

| S_local | H | A0 (ms) | A1 vs A0 | A2 vs A0 | **A3 vs A0** |
|---:|---:|---:|:---:|:---:|:---:|
| 2048 | 8 | 1.52 | 1.25× | 1.24× | **1.34×** |
| 4096 | 8 | 3.11 | 1.05× | 1.07× | **1.23×** |
| 8192 | 8 | 7.98 | 1.02× | 1.02× | **1.16×** |
| 2048 | 16 | 2.44 | 1.06× | 1.07× | **1.12×** |
| 4096 | 16 | 5.68 | 1.05× | 1.04× | **1.08×** |
| 8192 | 16 | 16.39 | 1.01× | 1.01× | **1.12×** |

**head_dim=64：**

| S_local | H | A0 (ms) | A1 vs A0 | A2 vs A0 | **A3 vs A0** |
|---:|---:|---:|:---:|:---:|:---:|
| 2048 | 8 | 1.49 | 1.22× | 1.21× | **1.93×** |
| 4096 | 8 | 1.75 | 1.07× | 1.09× | **1.14×** |
| 8192 | 8 | 4.60 | 1.02× | 1.03× | **1.11×** |
| 2048 | 16 | 1.54 | 1.29× | 1.19× | 1.29× |
| 4096 | 16 | 3.24 | 1.07× | 1.05× | 1.09× |
| 8192 | 16 | 9.30 | 1.02× | 1.01× | 1.09× |

**结论：**
- **A1 (packed comm + workspace cache)** — 短序列的主要来源 (~20-30%)；开销固定 O(W)，S 变大后被 compute 稀释。
- **A2 (triton merge kernel)** — 纯速度增益接近 0；价值在于保证 out_acc 全程 fp32，与 native path 拼接。
- **A3 (native triton fwd)** — **恒定 5–15% 增益，长序列唯一可靠的 fwd 加速来源**；跨 step 的 fp32 状态省下每步 bf16⇄fp32 的 HBM 带宽。
- **Backward 的巨大加速（4.2 长序列 3.09×）** 不在此 forward-only ablation 中体现，主要来自 fused grad-accum + packed dk/dv 通信 + fp32 dq 累加器。
