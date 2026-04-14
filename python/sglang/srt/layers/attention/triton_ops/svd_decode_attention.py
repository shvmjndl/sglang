"""
Fused SVD-decompression + decode attention Triton kernel for AttentionPack.

This kernel fuses the low-rank decompression of compressed KV cache with
the attention computation, avoiding HBM round-trips for full-size K/V tensors.

For each block of compressed tokens:
  1. Load K̄[block] from HBM → SRAM          (compact: BLOCK_T × R_k)
  2. Load D_k slice for this KV head         (R_k × D, shared across token blocks)
  3. Decompress in SRAM: K_block = K̄[block] @ D_k_head_slice
  4. Compute QK^T for this block
  5. Track running max/sum for online softmax
  6. Similarly decompress V̄ and accumulate output

Memory access pattern:
  - K̄/V̄ are read ONCE from HBM (compact representation)
  - D_k/D_v are read ONCE per head (shared across token blocks)
  - Full K/V (T_c × H*D) are NEVER materialized in HBM

Reference: arXiv:2603.23914, Section 3.2 and Algorithm 1
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _svd_fused_decode_attention_kernel(
    # Query: [B, H_q, D]
    Q,
    # Compressed key factors
    K_bar,  # [T_c, R_k] — compressed keys
    D_k,  # [R_k, H*D] — key decompression matrix
    # Compressed value factors
    V_bar,  # [T_c, R_v] — compressed values
    D_v,  # [R_v, H*D] — value decompression matrix
    # Uncompressed K/V (text + recent tokens): [T_u, H, D]
    K_uncomp,
    V_uncomp,
    # Output: [B, H_q, D]
    O,
    # Importance scores (read-only here): [T_c]
    Importance,
    # Strides
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kr,
    stride_dkr,
    stride_dkd,
    stride_vb,
    stride_vr,
    stride_dvr,
    stride_dvd,
    stride_kub,
    stride_kuh,
    stride_kud,
    stride_vub,
    stride_vuh,
    stride_vud,
    stride_ob,
    stride_oh,
    stride_od,
    # Dims
    T_c,
    T_u,
    H_kv: tl.constexpr,
    H_q: tl.constexpr,
    D: tl.constexpr,
    R_k: tl.constexpr,
    R_v: tl.constexpr,
    SM_SCALE: tl.constexpr,
    # Block sizes
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # GQA: map query head to KV head
    kv_head_id = head_id * H_kv // H_q

    # Load query for this batch/head: [D]
    q_offset = batch_id * stride_qb + head_id * stride_qh
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D
    q = tl.load(Q + q_offset + d_range * stride_qd, mask=d_mask, other=0.0).to(
        tl.float32
    )

    # Online softmax accumulators
    m_prev = -float("inf")
    l_prev = 0.0
    o_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Precompute D_k head slice offset: D_k is [R_k, H*D], we need columns [kv_head_id*D : (kv_head_id+1)*D]
    dk_head_offset = kv_head_id * D
    dv_head_offset = kv_head_id * D

    # === Process compressed tokens in blocks ===
    for t_start in range(0, T_c, BLOCK_T):
        t_range = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_range < T_c

        # Load K̄[block, :R_k] from compressed cache
        # We accumulate the decompressed K block using tiled matmul over R dimension
        k_block = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

        for r_start in range(0, R_k, BLOCK_R):
            r_range = r_start + tl.arange(0, BLOCK_R)
            r_mask = r_range < R_k

            # Load K̄ tile: [BLOCK_T, BLOCK_R]
            k_bar_tile = tl.load(
                K_bar + t_range[:, None] * stride_kb + r_range[None, :] * stride_kr,
                mask=t_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Load D_k tile: [BLOCK_R, BLOCK_D]
            dk_tile = tl.load(
                D_k
                + r_range[:, None] * stride_dkr
                + (dk_head_offset + d_range[None, :]) * stride_dkd,
                mask=r_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Accumulate: k_block += k_bar_tile @ dk_tile
            k_block += tl.dot(k_bar_tile, dk_tile)

        # Compute attention scores: q @ k_block^T → [BLOCK_T]
        scores = tl.sum(q[None, :] * k_block, axis=1)  # [BLOCK_T]
        scores = scores * SM_SCALE
        scores = tl.where(t_mask, scores, -float("inf"))

        # Online softmax update
        m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
        correction = tl.exp(m_prev - m_new)
        p = tl.exp(scores - m_new)
        l_new = correction * l_prev + tl.sum(p, axis=0)

        # Decompress V̄ block and accumulate output
        v_block = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

        for r_start in range(0, R_v, BLOCK_R):
            r_range = r_start + tl.arange(0, BLOCK_R)
            r_mask = r_range < R_v

            v_bar_tile = tl.load(
                V_bar + t_range[:, None] * stride_vb + r_range[None, :] * stride_vr,
                mask=t_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            dv_tile = tl.load(
                D_v
                + r_range[:, None] * stride_dvr
                + (dv_head_offset + d_range[None, :]) * stride_dvd,
                mask=r_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            v_block += tl.dot(v_bar_tile, dv_tile)

        # Accumulate: O = correction * O_prev + P @ V_block
        o_acc = correction * o_acc + tl.sum(p[:, None] * v_block, axis=0)

        m_prev = m_new
        l_prev = l_new

    # === Process uncompressed tokens (standard path) ===
    for t_start in range(0, T_u, BLOCK_T):
        t_range = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_range < T_u

        # Load K directly: [BLOCK_T, D]
        k_block = tl.load(
            K_uncomp
            + t_range[:, None] * stride_kub
            + kv_head_id * stride_kuh
            + d_range[None, :] * stride_kud,
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(q[None, :] * k_block, axis=1) * SM_SCALE
        scores = tl.where(t_mask, scores, -float("inf"))

        m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
        correction = tl.exp(m_prev - m_new)
        p = tl.exp(scores - m_new)
        l_new = correction * l_prev + tl.sum(p, axis=0)

        v_block = tl.load(
            V_uncomp
            + t_range[:, None] * stride_vub
            + kv_head_id * stride_vuh
            + d_range[None, :] * stride_vud,
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        o_acc = correction * o_acc + tl.sum(p[:, None] * v_block, axis=0)

        m_prev = m_new
        l_prev = l_new

    # Final normalization (guard against T_c=T_u=0 edge case)
    l_prev = tl.maximum(l_prev, 1e-6)
    o_acc = o_acc / l_prev

    # Store output
    o_offset = batch_id * stride_ob + head_id * stride_oh
    tl.store(O + o_offset + d_range * stride_od, o_acc.to(O.dtype.element_ty), mask=d_mask)


def svd_fused_decode_attention(
    q: torch.Tensor,  # [B, H_q, D]
    k_bar: torch.Tensor,  # [T_c, R_k]
    d_k: torch.Tensor,  # [R_k, H_kv*D]
    v_bar: torch.Tensor,  # [T_c, R_v]
    d_v: torch.Tensor,  # [R_v, H_kv*D]
    k_uncomp: torch.Tensor,  # [T_u, H_kv, D]
    v_uncomp: torch.Tensor,  # [T_u, H_kv, D]
    importance: torch.Tensor,  # [T_c]
    sm_scale: float,
) -> torch.Tensor:
    """Python wrapper for the fused SVD decode attention Triton kernel.

    Performs attention computation where keys and values are stored in
    SVD-compressed form, fusing the decompression with attention to avoid
    materializing full-size K/V in HBM.

    Args:
        q: Query tensor [B, H_q, D]
        k_bar: Compressed keys [T_c, R_k]
        d_k: Key decompression matrix [R_k, H_kv*D]
        v_bar: Compressed values [T_c, R_v]
        d_v: Value decompression matrix [R_v, H_kv*D]
        k_uncomp: Uncompressed keys [T_u, H_kv, D]
        v_uncomp: Uncompressed values [T_u, H_kv, D]
        importance: Importance scores [T_c] (for future partial decompression)
        sm_scale: Softmax scaling factor (1/sqrt(D))

    Returns:
        Output tensor [B, H_q, D]
    """
    B, H_q, D = q.shape
    T_c, R_k = k_bar.shape
    R_v = v_bar.shape[1]
    T_u = k_uncomp.shape[0]
    H_kv = d_k.shape[1] // D

    o = torch.empty_like(q)

    # Choose block sizes — tl.dot requires K >= 16
    BLOCK_T = max(min(triton.next_power_of_2(max(T_c, T_u, 1)), 64), 16)
    BLOCK_D = max(triton.next_power_of_2(D), 16)
    BLOCK_R = max(min(triton.next_power_of_2(max(R_k, R_v, 1)), 64), 16)

    grid = (B, H_q)

    _svd_fused_decode_attention_kernel[grid](
        q,
        k_bar,
        d_k,
        v_bar,
        d_v,
        k_uncomp,
        v_uncomp,
        o,
        importance,
        # Q strides
        q.stride(0),
        q.stride(1),
        q.stride(2),
        # K_bar strides
        k_bar.stride(0),
        k_bar.stride(1),
        # D_k strides
        d_k.stride(0),
        d_k.stride(1),
        # V_bar strides
        v_bar.stride(0),
        v_bar.stride(1),
        # D_v strides
        d_v.stride(0),
        d_v.stride(1),
        # K_uncomp strides
        k_uncomp.stride(0),
        k_uncomp.stride(1),
        k_uncomp.stride(2),
        # V_uncomp strides
        v_uncomp.stride(0),
        v_uncomp.stride(1),
        v_uncomp.stride(2),
        # O strides
        o.stride(0),
        o.stride(1),
        o.stride(2),
        # Dims
        T_c=T_c,
        T_u=T_u,
        H_kv=H_kv,
        H_q=H_q,
        D=D,
        R_k=R_k,
        R_v=R_v,
        SM_SCALE=sm_scale,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        BLOCK_R=BLOCK_R,
    )

    return o
