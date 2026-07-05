import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist


# This file implements a reference Ring Attention operator using plain PyTorch
# tensor math plus torch.distributed point-to-point communication. The sequence
# dimension is sharded across ranks. Each rank owns a local Q/K/V block, streams
# remote K/V blocks around a ring during forward, and streams query-side
# backward state around a ring during backward.


def setup(rank, world_size):
    """init dist env"""
    # Initialize one NCCL process per GPU. The test launcher is expected to set
    # RANK/WORLD_SIZE and the NCCL rendezvous environment variables.
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    # Bind this process to the CUDA device with the same index as its rank.
    torch.cuda.set_device(rank)


def cleanup():
    # Tear down the distributed process group after the test finishes.
    dist.destroy_process_group()


_CAUSAL_MASK_CACHE = {}


def _attention_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    # Keep fp32/fp64 inputs in their existing precision, but promote fp16/bf16
    # attention math and softmax statistics to fp32 for numerical stability.
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _finite_or_zero(tensor: torch.Tensor) -> torch.Tensor:
    # Fully masked causal rows can have a row max of -inf. Replacing non-finite
    # values only inside scale factors avoids inf - inf -> NaN in online softmax.
    return torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))


def _get_causal_block_info(
    q_block_rank: int,
    k_block_rank: int,
    q_len: int,
    k_len: int,
    causal: bool,
) -> Tuple[bool, bool, int, int]:
    # Returns:
    #   skip_block: the whole K block is in the future, so this block contributes
    #               nothing and should not launch matmul.
    #   needs_mask: only part of the K block is valid, so apply an elementwise
    #               causal mask before softmax.
    #   q_start/k_start: global token offsets for mask construction.
    q_start = q_block_rank * q_len
    k_start = k_block_rank * k_len

    if not causal:
        return False, False, q_start, k_start

    q_end = q_start + q_len
    k_end = k_start + k_len

    # All keys are strictly after every query token in this block.
    if k_start >= q_end:
        return True, False, q_start, k_start

    # All keys are strictly before the first query token, so the whole block is
    # valid and no elementwise mask is needed.
    needs_mask = k_end > q_start
    return False, needs_mask, q_start, k_start


def _get_causal_mask(
    q_len: int,
    k_len: int,
    q_start: int,
    k_start: int,
    device: torch.device,
) -> torch.Tensor:
    # Cache masks by shape, relative position, and device. Only diagonal or
    # partially overlapping causal blocks call this helper; fully past blocks do
    # not need a mask, and fully future blocks are skipped before matmul.
    device = torch.device(device)
    key = (q_len, k_len, q_start - k_start, device.type, device.index)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None or mask.device != device:
        q_positions = torch.arange(q_start, q_start + q_len, device=device)
        k_positions = torch.arange(k_start, k_start + k_len, device=device)
        mask = k_positions.view(1, 1, 1, k_len) > q_positions.view(1, 1, q_len, 1)
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


# use for ring attn communication
class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        # RingComm is a small helper around non-blocking P2P sends/receives.
        # It always sends tensors to the next rank and receives tensors from the
        # previous rank, so after world_size steps every rank has seen every
        # shard exactly once.
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        # Local rank ids inside the given process group form a logical ring:
        # rank r sends to r+1 and receives from r-1, with wrap-around.
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            # P2POp expects global ranks when a subgroup is provided, so convert
            # the ring-neighbor group ranks back to their global rank ids.
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Queue one async send and one async receive. The actual operations are
        # launched later by commit(), which allows callers to batch multiple
        # tensor transfers such as K and V together.
        if recv_tensor is None:
            # Allocate a fresh receive buffer with the same shape, dtype, and
            # device as the tensor being sent.
            res = torch.empty_like(to_send)
        else:
            # Reuse the caller-provided buffer to avoid repeated allocations.
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        # Launch all queued P2P operations as one batched call.
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        # Block until all queued sends/receives complete, then reset the helper
        # so it can be reused for the next ring step.
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # send and receive kv for forward pass
        # Forward streams the current K and V shards together. Dense mode keeps
        # K/V in sequence-first layout:
        #   (batch, local_seq, heads, head_dim)
        # Causal mode keeps K/V in head-first layout:
        #   (batch, heads, local_seq, head_dim)
        # The returned tensors preserve the caller's layout.
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v
        
    def send_recv_q_sum_max_sumodo_dout(
        self,
        q: torch.Tensor,
        sum_attn_scores: torch.Tensor,
        max_attn_scores: torch.Tensor,
        sumodo: torch.Tensor,
        dout: torch.Tensor,
        q_buffer: Optional[torch.Tensor] = None,
        sum_attn_scores_buffer: Optional[torch.Tensor] = None,
        max_attn_scores_buffer: Optional[torch.Tensor] = None,
        sumodo_buffer: Optional[torch.Tensor] = None,
        dout_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # send and receive q, sum_attn_scores, max_attn_scores, sumodo, dout for backward pass
        # Backward fixes local K/V and streams query-side state around the ring.
        # For each remote Q shard we also need the forward softmax statistics
        # (sum and max), the row-wise term delta=sum(dO * O), and that shard's
        # dO. Dense and causal paths use different layouts; the caller-side
        # comments in forward/backward mark the exact shapes.
        next_q, next_sum_attn_scores, next_max_attn_scores, next_sumodo, next_dout = \
            self.send_recv(q, q_buffer), self.send_recv(sum_attn_scores, sum_attn_scores_buffer), self.send_recv(max_attn_scores, max_attn_scores_buffer), self.send_recv(sumodo, sumodo_buffer), self.send_recv(dout, dout_buffer)
        self.commit()
        return next_q, next_sum_attn_scores, next_max_attn_scores, next_sumodo, next_dout

  
# ring attn forward and backward by pytorch
class RingAttentionFunction(torch.autograd.Function):
    """
    Reference Ring Attention autograd function with dense and causal support.
    """
    @staticmethod
    def forward(
        ctx,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        process_group: dist.ProcessGroup = None,
        causal: bool = False):
        """
        forward
        Args:
            ctx: data saved for backward
            q_local, k_local, v_local: q, k, v on local gpu, shape: (batch_size, seq_len_local, num_heads, head_dim), where seq_len_local = seq_len_total // world_size
            process_group: gpu group for ring attn
            causal: whether to apply autoregressive attention masking
        Returns:
            output: shape: (batch_size, seq_len_local, num_heads, head_dim)
        """
        # Each rank owns Q_i, K_i, V_i for a contiguous sequence shard. During
        # forward Q_i stays local, while K/V blocks circulate around the ring:
        #   step 0: attend Q_i to local K_i/V_i
        #   step 1: attend Q_i to the K/V block received from rank i-1
        #   ...
        # After world_size steps, Q_i has attended to every key/value shard,
        # which is equivalent to full attention over the global sequence.
        
        comm = RingComm(process_group)
        
        batch_size, seq_len_local, num_heads, head_dim = q_local.shape
        device = q_local.device
        dtype = q_local.dtype

        if not causal:
            # Dense fast path: keep the original lightweight sequence-first
            # implementation. This avoids extra contiguous copies used by the
            # causal path, which only pay off when future K blocks can be
            # skipped.
            #
            # Public/local layout:
            #   q_local/k_local/v_local: (batch, local_seq, heads, head_dim)
            # Internal matmul layout:
            #   q_dense/k_dense/v_dense: (batch, heads, local_seq, head_dim)
            softmax_scale = head_dim ** -0.5
            out_local = torch.zeros(
                batch_size,
                num_heads,
                seq_len_local,
                head_dim,
                device=device,
                dtype=dtype,
            )
            sum_local = torch.zeros(
                batch_size,
                num_heads,
                seq_len_local,
                1,
                device=device,
                dtype=dtype,
            )
            max_local = torch.full(
                (batch_size, num_heads, seq_len_local, 1),
                -float('inf'),
                device=device,
                dtype=dtype,
            )

            # Dense mode sends sequence-first K/V shards around the ring.
            k_cur = k_local.clone()
            v_cur = v_local.clone()
            q_dense = q_local.transpose(1, 2)

            for step in range(comm.world_size):
                if step + 1 != comm.world_size:
                    next_k, next_v = comm.send_recv_kv(k_cur, v_cur)

                # Shape change:
                #   k_cur/v_cur: (batch, local_seq, heads, head_dim)
                #   k_dense/v_dense: (batch, heads, local_seq, head_dim)
                k_dense = k_cur.transpose(1, 2)
                v_dense = v_cur.transpose(1, 2)

                # Block scores shape:
                #   (batch, heads, local_q, local_k)
                attn_scores = (q_dense @ k_dense.transpose(-2, -1)) * softmax_scale
                max_scores = torch.max(attn_scores, dim=-1, keepdim=True).values
                exp_scores = torch.exp(attn_scores - max_scores)

                # Online softmax merge in input dtype, matching the old dense
                # implementation exactly.
                max_scores_update = torch.maximum(max_local, max_scores)
                out_local = (
                    out_local * torch.exp(max_local - max_scores_update)
                    + (exp_scores @ v_dense)
                    * torch.exp(max_scores - max_scores_update)
                )
                sum_local = (
                    sum_local * torch.exp(max_local - max_scores_update)
                    + torch.sum(exp_scores, dim=-1, keepdim=True)
                    * torch.exp(max_scores - max_scores_update)
                )
                max_local = max_scores_update

                if step + 1 != comm.world_size:
                    comm.wait()
                    k_cur, v_cur = next_k, next_v

            sum_local = torch.where(sum_local == 0, torch.ones_like(sum_local), sum_local)

            # Save dense softmax stats in public sequence-first layout:
            #   out_local: (batch, local_seq, heads, head_dim)
            #   sum_local/max_local: (batch, local_seq, heads, 1)
            out_local = out_local.transpose(1, 2).contiguous()
            sum_local = sum_local.transpose(1, 2).contiguous()
            max_local = max_local.transpose(1, 2).contiguous()
            out_local = out_local / sum_local

            ctx.save_for_backward(q_local, k_local, v_local, out_local, sum_local, max_local)
            ctx.process_group = process_group
            ctx.causal = causal

            return out_local

        compute_dtype = _attention_compute_dtype(dtype)
        softmax_scale = head_dim ** -0.5
        
        # Internally use head-first layout:
        #   q_head/k_head/v_head shape: (batch, heads, local_seq, head_dim)
        # This avoids repeated transpose(1, 2) inside the ring loop.
        q_head = q_local.transpose(1, 2).contiguous()
        q_compute = q_head.to(compute_dtype)
        
        # Running softmax numerator:
        #   sum_j exp(score_ij - global_row_max_i) @ V_j
        # Shape: (batch, heads, local_query_len, head_dim). Kept in fp32 for
        # fp16/bf16 inputs, then cast back to the input dtype at return time.
        out_acc = torch.zeros(
            batch_size,
            num_heads,
            seq_len_local,
            head_dim,
            device=device,
            dtype=compute_dtype,
        )
        
        # Running softmax denominator:
        #   sum_j exp(score_ij - global_row_max_i)
        # Shape: (batch, heads, local_query_len, 1).
        sum_acc = torch.zeros(
            batch_size,
            num_heads,
            seq_len_local,
            1,
            device=device,
            dtype=compute_dtype,
        )
        
        # Running row-wise max over all visible key shards. Shape:
        # (batch, heads, local_query_len, 1). This is what makes the online
        # softmax merge numerically stable.
        max_acc = torch.full(
            (batch_size, num_heads, seq_len_local, 1),
            -float('inf'),
            device=device,
            dtype=compute_dtype,
        )

        # K/V are also sent around the ring in head-first layout:
        #   k_cur/v_cur shape: (batch, heads, local_key_len, head_dim)
        k_cur = k_local.transpose(1, 2).contiguous()
        v_cur = v_local.transpose(1, 2).contiguous()

        q_block_rank = comm.rank

        for step in range(comm.world_size):
            
            if step + 1 != comm.world_size:
                # Start sending the current K/V shard to the next rank and
                # receiving the next shard from the previous rank. Compute on
                # k_cur/v_cur while this transfer is in flight.
                next_k, next_v = comm.send_recv_kv(k_cur, v_cur)

            # Ring order for this code is local rank, rank-1, rank-2, ... with
            # wrap-around. This group-rank id maps to the global sequence block
            # position for causal skip/mask decisions.
            k_block_rank = (comm.rank - step) % comm.world_size
            skip_block, needs_mask, q_start, k_start = _get_causal_block_info(
                q_block_rank,
                k_block_rank,
                seq_len_local,
                seq_len_local,
                causal,
            )

            if not skip_block:
                # Current block logits:
                #   q_compute shape: (batch, heads, local_q, head_dim)
                #   k_compute shape: (batch, heads, local_k, head_dim)
                #   attn_scores shape: (batch, heads, local_q, local_k)
                k_compute = k_cur.to(compute_dtype)
                v_compute = v_cur.to(compute_dtype)
                attn_scores = (q_compute @ k_compute.transpose(-2, -1)) * softmax_scale

                if needs_mask:
                    # Only diagonal/partially overlapping causal blocks need an
                    # elementwise mask. Fully future blocks were skipped above,
                    # and fully past blocks are already entirely visible.
                    causal_mask = _get_causal_mask(
                        seq_len_local,
                        seq_len_local,
                        q_start,
                        k_start,
                        device,
                    )
                    attn_scores = attn_scores.masked_fill(causal_mask, -float('inf'))
                
                # Block-local row max. Shape: (batch, heads, local_q, 1).
                max_scores = torch.max(attn_scores, dim=-1, keepdim=True).values
                safe_max_scores = _finite_or_zero(max_scores)
                
                # Unnormalized positive weights for this block:
                #   exp(score_block - block_row_max)
                exp_scores = torch.exp(attn_scores - safe_max_scores)
                
                # Online softmax merge. Shape of max_scores_update:
                # (batch, heads, local_q, 1).
                max_scores_update = torch.maximum(max_acc, max_scores)
                safe_max_acc = _finite_or_zero(max_acc)
                safe_max_scores_update = _finite_or_zero(max_scores_update)
                
                # Running numerator update:
                #   old_num * exp(old_max - new_max)
                # + block_num * exp(block_max - new_max)
                out_acc = (
                    out_acc * torch.exp(safe_max_acc - safe_max_scores_update)
                    + (exp_scores @ v_compute)
                    * torch.exp(safe_max_scores - safe_max_scores_update)
                )
                
                # Running denominator update with the same rescaling.
                sum_acc = (
                    sum_acc * torch.exp(safe_max_acc - safe_max_scores_update)
                    + torch.sum(exp_scores, dim=-1, keepdim=True)
                    * torch.exp(safe_max_scores - safe_max_scores_update)
                )
                
                max_acc = max_scores_update
            
            if step + 1 != comm.world_size:
                # Ensure the next K/V shard has arrived before moving to the
                # next ring step.
                comm.wait()
                k_cur, v_cur = next_k, next_v
        
        # Guard against division by zero. In normal dense attention this should
        # not happen, but it keeps the function robust for degenerate inputs.
        sum_acc = torch.where(sum_acc == 0, torch.ones_like(sum_acc), sum_acc)
        
        # Normalize the accumulated numerator in compute dtype. Shape:
        #   out_head: (batch, heads, local_seq, head_dim)
        out_head = out_acc / sum_acc
        
        # Public output layout is sequence-first:
        #   out_local: (batch, local_seq, heads, head_dim)
        out_local = out_head.to(dtype).transpose(1, 2).contiguous()
        
        # Save Q/K/V, output, and softmax statistics so backward can reconstruct
        # each attention probability block without saving the full attention
        # matrix.
        # Saved out_head/sum_acc/max_acc are head-first and compute dtype:
        #   out_head: (batch, heads, local_seq, head_dim)
        #   sum_acc/max_acc: (batch, heads, local_seq, 1)
        ctx.save_for_backward(q_local, k_local, v_local, out_head, sum_acc, max_acc)
        ctx.process_group = process_group
        ctx.causal = causal

        return out_local
    
    
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is dL/dO for this rank's local output shard.
        q_local, k_local, v_local, saved_out, saved_sum, saved_max = ctx.saved_tensors
        process_group = ctx.process_group 
        causal = ctx.causal
        
        batch_size, seq_len_local, num_heads, head_dim = q_local.shape
        device = q_local.device
        softmax_scale = head_dim ** -0.5

        comm = RingComm(process_group)

        if not causal:
            # Dense fast path: mirror the original lightweight backward layout.
            # Saved tensors from dense forward are sequence-first:
            #   saved_out: (batch, local_seq, heads, head_dim)
            #   saved_sum/saved_max: (batch, local_seq, heads, 1)
            dk_local = torch.zeros_like(k_local)
            dv_local = torch.zeros_like(v_local)
            
            # Query-side state rotates in sequence-first layout:
            #   q_cur/grad_output_cur/grad_q_cur:
            #       (batch, local_seq, heads, head_dim)
            #   sum_cur/max_cur:
            #       (batch, local_seq, heads, 1)
            q_cur = q_local.clone()
            sum_cur = saved_sum.clone()
            max_cur = saved_max.clone()
            grad_q_cur = torch.zeros_like(q_local)
            grad_output_cur = grad_output.clone()

            # delta_cur is head-first because it is directly broadcast across
            # the key dimension in softmax backward:
            #   delta_cur shape: (batch, heads, local_q, 1)
            delta_cur = torch.sum(
                grad_output.transpose(1, 2) * saved_out.transpose(1, 2),
                dim=-1,
                keepdim=True,
            )
            
            # Local K/V stay fixed and are viewed as head-first for matmul:
            #   k_dense/v_dense shape: (batch, heads, local_k, head_dim)
            k_dense = k_local.transpose(1, 2)
            v_dense = v_local.transpose(1, 2)
                    
            for step in range(comm.world_size):
                if step + 1 != comm.world_size:
                    next_q_cur, next_sum_cur, next_max_cur, next_delta_cur, next_grad_output_cur = \
                        comm.send_recv_q_sum_max_sumodo_dout(q_cur, sum_cur, max_cur, delta_cur, grad_output_cur)

                # Shape change for the current query shard:
                #   q_cur/grad_output_cur:
                #       (batch, local_seq, heads, head_dim)
                #   q_dense/grad_output_dense:
                #       (batch, heads, local_q, head_dim)
                q_dense = q_cur.transpose(1, 2)
                grad_output_dense = grad_output_cur.transpose(1, 2)
                
                # Reconstruct A_{q_cur,k_local}. Shape:
                #   attn_probs: (batch, heads, local_q, local_k)
                attn_probs = (
                    torch.exp(
                        (q_dense @ k_dense.transpose(-2, -1)) * softmax_scale
                        - max_cur.transpose(1, 2)
                    )
                    / sum_cur.transpose(1, 2)
                )

                # dV_i += A^T @ dO.
                dv_local += (
                    attn_probs.transpose(-2, -1) @ grad_output_dense
                ).transpose(1, 2).contiguous()

                # dA = dO @ V_i^T.
                grad_attn_probs = grad_output_dense @ v_dense.transpose(-2, -1)

                # dScore = A * (dA - delta), where delta=sum(dO*O) per query row.
                grad_scores = attn_probs * (grad_attn_probs - delta_cur)

                # dK_i += dScore^T @ Q * softmax_scale.
                dk_local += (
                    grad_scores.transpose(-2, -1) @ q_dense * softmax_scale
                ).transpose(1, 2).contiguous()

                # dQ_cur += dScore @ K_i * softmax_scale.
                grad_q_cur += (
                    grad_scores @ k_dense * softmax_scale
                ).transpose(1, 2).contiguous()
                
                if step + 1 != comm.world_size:
                    comm.wait()
                    q_cur, sum_cur, max_cur, delta_cur, grad_output_cur = \
                        next_q_cur, next_sum_cur, next_max_cur, next_delta_cur, next_grad_output_cur
                
                # Rotate accumulated dQ so after a full ring each rank gets the
                # dQ shard corresponding to its original local Q.
                next_grad_q_cur = comm.send_recv(grad_q_cur, None)
                comm.commit()
                comm.wait()
                grad_q_cur = next_grad_q_cur
            
            return grad_q_cur, dk_local, dv_local, None, None

        out_head = saved_out
        sum_head = saved_sum
        max_head = saved_max
        compute_dtype = out_head.dtype
        
        # This rank owns K_i/V_i, so it accumulates final gradients for only its
        # local K/V shards. Internal shape is head-first:
        #   dk_head/dv_head shape: (batch, heads, local_seq, head_dim)
        dk_head = torch.zeros(
            batch_size,
            num_heads,
            seq_len_local,
            head_dim,
            device=device,
            dtype=compute_dtype,
        )
        dv_head = torch.zeros_like(dk_head)
        
        # Query-side state is rotated through the ring. For the current
        # q_cur/dO_cur shard, this rank computes that shard's contribution to
        # local dK_i/dV_i and a partial dQ contribution.
        # q_cur shape: (batch, heads, local_seq, head_dim), input dtype.
        q_cur = q_local.transpose(1, 2).contiguous()
        # sum_cur/max_cur shape: (batch, heads, local_seq, 1), compute dtype.
        sum_cur = sum_head.clone()
        max_cur = max_head.clone()
        # grad_q_cur shape: (batch, heads, local_seq, head_dim), compute dtype.
        grad_q_cur = torch.zeros(
            batch_size,
            num_heads,
            seq_len_local,
            head_dim,
            device=device,
            dtype=compute_dtype,
        )
        # grad_output_cur shape: (batch, heads, local_seq, head_dim).
        grad_output_cur = grad_output.transpose(1, 2).contiguous()
        # For softmax backward, each query row needs:
        #   sum_k dA_k * A_k = sum_head_dim(dO * O)
        # because O = A @ V. This row-wise scalar is saved/communicated with Q.
        # delta_cur shape: (batch, heads, local_seq, 1). It depends only on the
        # query shard, so it is computed once and moved around the ring.
        delta_cur = torch.sum(
            grad_output_cur.to(compute_dtype) * out_head,
            dim=-1,
            keepdim=True,
        )
        
        # Local K_i/V_i stay fixed on this rank throughout the backward loop.
        # k_head/v_head shape: (batch, heads, local_seq, head_dim).
        k_head = k_local.transpose(1, 2).contiguous()
        v_head = v_local.transpose(1, 2).contiguous()
        k_compute = k_head.to(compute_dtype)
        v_compute = v_head.to(compute_dtype)
        k_block_rank = comm.rank
                
        for step in range(comm.world_size):
            
            if step + 1 != comm.world_size:
                # Rotate query-side backward state so every rank eventually
                # pairs its local K/V shard with every Q shard.
                next_q_cur, next_sum_cur, next_max_cur, next_delta_cur, next_grad_output_cur = \
                    comm.send_recv_q_sum_max_sumodo_dout(q_cur, sum_cur, max_cur, delta_cur, grad_output_cur)

            # Query blocks rotate in the same rank, rank-1, rank-2, ... order.
            q_block_rank = (comm.rank - step) % comm.world_size
            skip_block, needs_mask, q_start, k_start = _get_causal_block_info(
                q_block_rank,
                k_block_rank,
                seq_len_local,
                seq_len_local,
                causal,
            )

            if not skip_block:
                q_compute = q_cur.to(compute_dtype)
                grad_output_compute = grad_output_cur.to(compute_dtype)
                
                # Reconstruct attention probability block A_{q_cur,k_local}
                # from logits plus the global row max/denominator saved in
                # forward:
                #   score shape: (batch, heads, local_q, local_k)
                #   A = exp(score - global_row_max) / global_row_sum
                scores = (q_compute @ k_compute.transpose(-2, -1)) * softmax_scale
                if needs_mask:
                    causal_mask = _get_causal_mask(
                        seq_len_local,
                        seq_len_local,
                        q_start,
                        k_start,
                        device,
                    )
                    scores = scores.masked_fill(causal_mask, -float('inf'))

                attn_probs = torch.exp(scores - max_cur) / sum_cur
                
                # dV_i accumulates A^T @ dO for every visible query shard.
                # Shape: (batch, heads, local_k, head_dim).
                dv_head += attn_probs.transpose(-2, -1) @ grad_output_compute
                
                # dA = dO @ V_i^T for this K/V block.
                # Shape: (batch, heads, local_q, local_k).
                grad_attn_probs = grad_output_compute @ v_compute.transpose(-2, -1)
                
                # Softmax backward for this block:
                #   dScore = A * (dA - delta)
                # delta is sum(dO * O) over head_dim for the query row and is
                # shared across all key blocks for that query shard.
                grad_scores = attn_probs * (grad_attn_probs - delta_cur)
                
                # Local K gradient receives dScore^T @ Q * softmax_scale.
                # Shape: (batch, heads, local_k, head_dim).
                dk_head += (grad_scores.transpose(-2, -1) @ q_compute) * softmax_scale
                
                # Current query shard receives dScore @ K_i * softmax_scale.
                # This is only the contribution from local K_i; the shard will
                # collect other K-block contributions as it visits other ranks.
                grad_q_cur += (grad_scores @ k_compute) * softmax_scale
            
            if step + 1 != comm.world_size:
                # Move to the next remote query shard.
                comm.wait()
                q_cur, sum_cur, max_cur, delta_cur, grad_output_cur = \
                    next_q_cur, next_sum_cur, next_max_cur, next_delta_cur, next_grad_output_cur
            
            # Rotate the accumulated dQ shard as well. After a full ring, each
            # rank receives the dQ corresponding to its original local Q shard.
            next_grad_q_cur = comm.send_recv(grad_q_cur, None)
            comm.commit()
            comm.wait()
            grad_q_cur = next_grad_q_cur
        
        # Return gradients for q_local, k_local, v_local, and None for the
        # non-tensor process_group/causal arguments. Public gradient layout is
        # sequence-first: (batch, local_seq, heads, head_dim).
        grad_q_local = grad_q_cur.to(q_local.dtype).transpose(1, 2).contiguous()
        dk_local = dk_head.to(k_local.dtype).transpose(1, 2).contiguous()
        dv_local = dv_head.to(v_local.dtype).transpose(1, 2).contiguous()
        return grad_q_local, dk_local, dv_local, None, None
    

def ring_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    process_group: dist.ProcessGroup = None,
    causal: bool = False):
    
    # Public wrapper around the custom autograd Function. Inputs and output use
    # local sequence layout: (batch, seq_len_per_rank, num_heads, head_dim).
    return RingAttentionFunction.apply(
        q, k, v, process_group, causal
    )



def test_ring_attn(causal: bool = False):
    # Distributed correctness test. Launch with torchrun so each rank owns one
    # CUDA device and one contiguous sequence shard.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size)
    mode = "causal" if causal else "dense"
    print(f"Rank {rank} is initialized on GPU {torch.cuda.current_device()} for {mode} ring attention")
    
    # torch.manual_seed(0)
    
    ## hyper paras
    # ==========
    # Keep total_seq_len divisible by world_size because this reference code
    # assumes equal-size sequence shards.
    batch_size = 2
    total_seq_len = 128
    num_heads = 4
    head_dim = 8
    block_size = total_seq_len // world_size
    # ==========
    
    
    # qkv and grad of output
    # ==========
    # Rank 0 creates the full reference tensors, then broadcasts them so every
    # rank can compare its local ring result against full attention.
    if rank == 0:
        # qkv = torch.rand((batch_size, total_seq_len, num_heads*3, head_dim), dtype = torch.bfloat16, device=f'cuda:{rank}', requires_grad=True)
        qkv = torch.randn((batch_size, total_seq_len, num_heads*3, head_dim), device=f'cuda:{rank}', requires_grad=True)
        dout = torch.randn((batch_size, total_seq_len, num_heads, head_dim), device=f'cuda:{rank}', requires_grad=False)
    else:
        # qkv = torch.zeros((batch_size, total_seq_len, num_heads*3, head_dim), dtype = torch.bfloat16, device=f'cuda:{rank}', requires_grad=True)
        qkv = torch.zeros((batch_size, total_seq_len, num_heads*3, head_dim), device=f'cuda:{rank}', requires_grad=True)
        dout = torch.zeros((batch_size, total_seq_len, num_heads, head_dim), device=f'cuda:{rank}', requires_grad=False)
    torch.distributed.broadcast(qkv, src = 0)  
    torch.distributed.broadcast(dout, src = 0)  

    # Split the full Q/K/V tensors along the sequence dimension and keep only
    # the shard owned by this rank for ring attention.
    q, k, v = qkv.chunk(3, dim=-2)
    q_local = torch.chunk(q, chunks=world_size, dim=1)[rank]
    k_local = torch.chunk(k, chunks=world_size, dim=1)[rank]
    v_local = torch.chunk(v, chunks=world_size, dim=1)[rank]
    q_local.retain_grad()
    k_local.retain_grad()
    v_local.retain_grad()
    
    dout_local = torch.chunk(dout, chunks=world_size, dim=1)[rank]
    
    # Separate clone for the ordinary full-attention reference path, so its
    # gradients can be compared against the custom Function's gradients.
    qkv_normal_attn = qkv.detach().clone()
    qkv_normal_attn.requires_grad = True
    q_normal_attn, k_normal_attn, v_normal_attn = qkv_normal_attn.chunk(3, dim=-2)
    q_normal_attn.retain_grad()
    k_normal_attn.retain_grad()
    v_normal_attn.retain_grad()
    # ==========
    
    
    # --- ring attention (forward) ---
    # ==========
    # Computes only this rank's local output chunk.
    ring_output = ring_attn(q_local, k_local, v_local, causal=causal)
    # ==========
    
    
    # --- ring attention (backward) ---
    # ==========
    # Backpropagate only the local upstream gradient shard. The custom backward
    # uses ring communication to recover the global-attention gradients.
    ring_output.backward(dout_local)
    # ==========
    
    
    # -- normal attention (forward) --
    # ==========
    # Full reference attention materializes the complete attention matrix on
    # every rank for correctness checking only.
    normal_attn_scores = (q_normal_attn.transpose(1,2) @ k_normal_attn.transpose(1,2).transpose(-2, -1)) / (head_dim ** 0.5)
    if causal:
        normal_attn_scores = normal_attn_scores.masked_fill(
            _get_causal_mask(total_seq_len, total_seq_len, 0, 0, normal_attn_scores.device),
            -float('inf'),
        )
    normal_attn_scores = torch.nn.functional.softmax(normal_attn_scores, dim=-1)
    normal_attn_output = (normal_attn_scores @ v_normal_attn.transpose(1,2)).transpose(1,2)
    normal_attn_output_chunk = torch.chunk(normal_attn_output, chunks=world_size, dim=1)
    # ==========
    
    
    # -- normal attention (backward) --
    # ==========
    normal_attn_output.backward(dout)
    # ==========
    
    
    # -- normal attention (backward by hand) --
    # ==========
    # Manual dense-attention backward formulas provide a second reference in
    # addition to PyTorch autograd.
    attn_scores = normal_attn_scores
    dv = (attn_scores.transpose(-2, -1) @ dout.transpose(1,2)).transpose(1,2)
    grad_attn_scores = dout.transpose(1,2) @ v.transpose(1,2).transpose(-2, -1)
    grad_qk = attn_scores * (grad_attn_scores - torch.sum(dout.transpose(1,2) * normal_attn_output.transpose(1,2), dim=-1, keepdim=True))
    dq = (grad_qk @ k_normal_attn.transpose(1,2) / (head_dim ** 0.5)).transpose(1,2)
    dk = (grad_qk.transpose(-2, -1) @ q_normal_attn.transpose(1,2) / (head_dim ** 0.5)).transpose(1,2)
    # ==========
    
    
    # compare forward output dif between ring attn and normal attn
    # ==========
    # The ring output chunk should match the corresponding slice of full
    # attention.
    print(f"[rank: {rank}] {(ring_output - normal_attn_output_chunk[rank]).sum()}")
    
    assert torch.allclose(ring_output, normal_attn_output_chunk[rank], atol = 1e-3), 'out is not the same'
    # ==========
    
    
    # compare backward output dif between normal attn (auto) and normal attn (by hand)
    # ==========
    # These prints check that the manual reference formulas agree with standard
    # autograd for dense attention.
    print(f"dq diff {(dq - q_normal_attn.grad).sum()}")
    print(f"dk diff {(dk - k_normal_attn.grad).sum()}")
    print(f"dv diff {(dv - v_normal_attn.grad).sum()}")
    # ==========
    
    
    # compare backward output dif between normal attn (auto) and ring attn
    # ==========
    # Rank 0 performs strict gradient equality checks against its local shard.
    # Other ranks still compute and print earlier diagnostics.
    if rank == 0:
        print(f"[rank: {rank}] dq = {(q_local.grad - torch.chunk(q_normal_attn.grad, chunks=world_size, dim=1)[rank]).sum()}")
        print(f"[rank: {rank}] dk = {(k_local.grad - torch.chunk(k_normal_attn.grad, chunks=world_size, dim=1)[rank]).sum()}")
        print(f"[rank: {rank}] dv = {(v_local.grad - torch.chunk(v_normal_attn.grad, chunks=world_size, dim=1)[rank]).sum()}")
        assert torch.allclose(q_local.grad, torch.chunk(q_normal_attn.grad, chunks=world_size, dim=1)[rank], atol = 1e-3), 'grad is not the same'
        assert torch.allclose(k_local.grad, torch.chunk(k_normal_attn.grad, chunks=world_size, dim=1)[rank], atol = 1e-3), 'grad is not the same'
        assert torch.allclose(v_local.grad, torch.chunk(v_normal_attn.grad, chunks=world_size, dim=1)[rank], atol = 1e-3), 'grad is not the same'
    # ==========


def test_ring_comm():
    # Minimal communication smoke test: each rank starts with a scalar value,
    # then repeatedly sends it around the ring.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    setup(rank, world_size)
    
    print(f"Rank {rank} is initialized on GPU {torch.cuda.current_device()}")
    
    k = torch.ones(1, device = torch.device(f"cuda:{rank}")) * (rank+1)
    v = torch.ones(1, device = torch.device(f"cuda:{rank}")) * (rank+1) * 100
    
    comm = RingComm(None)
    print(f"[rank: {rank}] [send_rank: {comm.send_rank}] [recv_rank: {comm.recv_rank}]")
    for step in range(comm.world_size):
        # After each step, rank 0 observes the next value received from its
        # previous neighbor in the ring.
        if rank == 0:
            print(f"[rank: {rank}, step: {step+1} | {comm.world_size}] before send_recv, k = {k}, v = {v}")
        next_k, next_v = comm.send_recv_kv(k, v)
        comm.wait()
        k, v = next_k, next_v
        if rank == 0:
            print(f"[rank: {rank}, step: {step+1} | {comm.world_size}] after send_recv, k = {k}, v = {v}")
            
  
if __name__ == "__main__":
    test_ring_comm()
    test_ring_attn()
