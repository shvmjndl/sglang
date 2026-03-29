"""
TurboQuant memory pool for KV cache compression.

Implements Google's TurboQuant (ICLR 2026) KV cache quantization.
Stores bit-packed centroid indices + L2 norms per head per token.

Two operating modes for reads:
  1. Selective dequant (default): call set_active_indices(indices) before
     get_kv_buffer to dequant only the needed positions into a shared
     workspace.  O(active_tokens) per read.
  2. Write-through fallback: if set_active_indices is never called,
     set_kv_buffer writes through to per-layer bf16 buffers so reads
     are O(1) pointer returns.  Uses more memory but always correct.

Follows the same pattern as MHATokenToKVPoolFP4 for buffer management.
"""

from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.quantization.turboquant_kernels import (
    HadamardTransform,
    _next_power_of_2,
    compute_packed_dim,
    compute_packed_dim_mixed,
    initialize_centroids_cache,
    parse_bits,
    turboquant_dequantize,
    turboquant_dequantize_mixed,
    turboquant_quantize,
    turboquant_quantize_mixed,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    get_tensor_size_bytes,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

# Deterministic seeds for the randomized Hadamard rotation.
_HADAMARD_SEED_K = 42
_HADAMARD_SEED_K_LO = 43
_HADAMARD_SEED_V = 137
_HADAMARD_SEED_V_LO = 138


class MHATokenToKVPoolTurboQuant(MHATokenToKVPool):
    """Memory pool that stores KV cache compressed via TurboQuant.

    Storage per token per head per layer:
      - Bit-packed centroid indices (uint8, packed at b bits/coord)
      - L2 norm (float32, 1 per token-head)

    Reads use per-layer write-through buffers that are updated on every
    set_kv_buffer call, so _get_key_buffer / _get_value_buffer are O(1).
    The compressed storage is authoritative for move_kv_cache.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        bits: float = 4,
        mode: str = "mse",
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        self.bits = bits
        self.mode = mode
        self.is_mixed, self.bits_hi, self.bits_lo = parse_bits(bits)

        # Cache padded dimensions (needed for prod mode QJL buffers)
        self.padded_head_dim = _next_power_of_2(head_dim)
        effective_v = v_head_dim if v_head_dim is not None else head_dim
        self.v_padded_head_dim = _next_power_of_2(effective_v)

        # Hadamard transforms: for mixed-precision each group gets its own.
        torch_device = torch.device(device)
        if self.is_mixed:
            k_split = head_dim // 2
            v_split = effective_v // 2
            self.k_hadamard_hi = HadamardTransform(
                k_split, seed=_HADAMARD_SEED_K, device=torch_device
            )
            self.k_hadamard_lo = HadamardTransform(
                head_dim - k_split, seed=_HADAMARD_SEED_K_LO, device=torch_device
            )
            self.v_hadamard_hi = HadamardTransform(
                v_split, seed=_HADAMARD_SEED_V, device=torch_device
            )
            self.v_hadamard_lo = HadamardTransform(
                effective_v - v_split, seed=_HADAMARD_SEED_V_LO, device=torch_device
            )
            self._k_split_dim = k_split
            self._v_split_dim = v_split
            self.k_hadamard = self.k_hadamard_hi
            self.v_hadamard = self.v_hadamard_hi
        else:
            self.k_hadamard = HadamardTransform(
                head_dim, seed=_HADAMARD_SEED_K, device=torch_device
            )
            self.v_hadamard = HadamardTransform(
                effective_v, seed=_HADAMARD_SEED_V, device=torch_device
            )

        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )

        initialize_centroids_cache(torch_device)

    def _create_buffers(self):
        """Allocate compressed storage + per-layer write-through buffers."""
        self.store_dtype = torch.uint8

        m = self.size + self.page_size
        k_packed_dim = compute_packed_dim_mixed(self.head_dim, self.bits)
        v_packed_dim = compute_packed_dim_mixed(self.v_head_dim, self.bits)

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # Bit-packed centroid indices — per layer
                self.k_buffer = [
                    torch.zeros(
                        (m, self.head_num, k_packed_dim),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, self.head_num, v_packed_dim),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                # L2 norms — per layer
                norm_shape = (
                    (m, self.head_num, 2) if self.is_mixed else (m, self.head_num)
                )
                self.k_norms_buffer = [
                    torch.zeros(norm_shape, dtype=torch.float32, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_norms_buffer = [
                    torch.zeros(norm_shape, dtype=torch.float32, device=self.device)
                    for _ in range(self.layer_num)
                ]

                # QJL — only for "prod" mode
                if self.mode == "prod":
                    k_qjl_dim = compute_packed_dim(self.padded_head_dim, 1)
                    v_qjl_dim = compute_packed_dim(self.v_padded_head_dim, 1)
                    self.k_qjl_buffer = [
                        torch.zeros(
                            (m, self.head_num, k_qjl_dim),
                            dtype=torch.uint8, device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_qjl_buffer = [
                        torch.zeros(
                            (m, self.head_num, v_qjl_dim),
                            dtype=torch.uint8, device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.k_residual_norms_buffer = [
                        torch.zeros(
                            (m, self.head_num),
                            dtype=torch.float32, device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_residual_norms_buffer = [
                        torch.zeros(
                            (m, self.head_num),
                            dtype=torch.float32, device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

                # Per-layer write-through buffers (dequantized K/V in working dtype).
                # Updated on set_kv_buffer so reads are O(1).
                self._k_wt = [
                    torch.zeros(
                        (m, self.head_num, self.head_dim),
                        dtype=self.dtype, device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self._v_wt = [
                    torch.zeros(
                        (m, self.head_num, self.v_head_dim),
                        dtype=self.dtype, device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.k_buffer, self.v_buffer
        del self.k_norms_buffer, self.v_norms_buffer
        del self._k_wt, self._v_wt
        if self.mode == "prod":
            del self.k_qjl_buffer, self.v_qjl_buffer
            del self.k_residual_norms_buffer, self.v_residual_norms_buffer

    def get_kv_size_bytes(self):
        k_size = sum(get_tensor_size_bytes(b) for b in self.k_buffer)
        k_size += sum(get_tensor_size_bytes(b) for b in self.k_norms_buffer)
        k_size += sum(get_tensor_size_bytes(b) for b in self._k_wt)
        v_size = sum(get_tensor_size_bytes(b) for b in self.v_buffer)
        v_size += sum(get_tensor_size_bytes(b) for b in self.v_norms_buffer)
        v_size += sum(get_tensor_size_bytes(b) for b in self._v_wt)
        if self.mode == "prod":
            k_size += sum(get_tensor_size_bytes(b) for b in self.k_qjl_buffer)
            k_size += sum(
                get_tensor_size_bytes(b) for b in self.k_residual_norms_buffer
            )
            v_size += sum(get_tensor_size_bytes(b) for b in self.v_qjl_buffer)
            v_size += sum(
                get_tensor_size_bytes(b) for b in self.v_residual_norms_buffer
            )
        return k_size, v_size

    # ── Read path: O(1) pointer return ──

    def _get_key_buffer(self, layer_id: int):
        return self._k_wt[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        return self._v_wt[layer_id - self.start_layer]

    # ── Write path: quantize + store compressed + write-through dequant ──

    def _dequant_to_wt(self, k_q, v_q, loc, idx, num_tokens):
        """Dequantize and write to per-layer write-through buffers."""
        if self.is_mixed:
            k_recon = turboquant_dequantize_mixed(
                k_q, self.k_hadamard_hi, self.k_hadamard_lo, self.dtype
            )
            v_recon = turboquant_dequantize_mixed(
                v_q, self.v_hadamard_hi, self.v_hadamard_lo, self.dtype
            )
        else:
            k_recon = turboquant_dequantize(
                k_q, self.k_hadamard, int(self.bits), self.mode, self.dtype
            )
            v_recon = turboquant_dequantize(
                v_q, self.v_hadamard, int(self.bits), self.mode, self.dtype
            )
        self._k_wt[idx][loc] = k_recon[:, : self.head_dim].reshape(
            num_tokens, self.head_num, self.head_dim
        )
        self._v_wt[idx][loc] = v_recon[:, : self.v_head_dim].reshape(
            num_tokens, self.head_num, self.v_head_dim
        )

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """Quantize, store compressed, and write-through dequantized data."""
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        idx = layer_id - self.start_layer
        num_tokens = cache_k.shape[0]

        k_flat = cache_k.reshape(-1, self.head_dim)
        v_flat = cache_v.reshape(-1, self.v_head_dim)

        # Quantize
        if self.is_mixed:
            k_q = turboquant_quantize_mixed(
                k_flat, self.k_hadamard_hi, self.k_hadamard_lo,
                self.bits_hi, self.bits_lo, self._k_split_dim,
            )
            v_q = turboquant_quantize_mixed(
                v_flat, self.v_hadamard_hi, self.v_hadamard_lo,
                self.bits_hi, self.bits_lo, self._v_split_dim,
            )
        else:
            k_q = turboquant_quantize(k_flat, self.k_hadamard, int(self.bits), self.mode)
            v_q = turboquant_quantize(v_flat, self.v_hadamard, int(self.bits), self.mode)

        # Store compressed
        if self.is_mixed:
            packed_k = torch.cat([k_q["packed_hi"], k_q["packed_lo"]], dim=-1)
            packed_v = torch.cat([v_q["packed_hi"], v_q["packed_lo"]], dim=-1)
            self.k_buffer[idx][loc] = packed_k.reshape(num_tokens, self.head_num, -1)
            self.v_buffer[idx][loc] = packed_v.reshape(num_tokens, self.head_num, -1)
            self.k_norms_buffer[idx][loc] = torch.stack(
                [k_q["norms_hi"], k_q["norms_lo"]], dim=-1
            ).reshape(num_tokens, self.head_num, 2)
            self.v_norms_buffer[idx][loc] = torch.stack(
                [v_q["norms_hi"], v_q["norms_lo"]], dim=-1
            ).reshape(num_tokens, self.head_num, 2)
        else:
            self.k_buffer[idx][loc] = k_q["packed_indices"].reshape(num_tokens, self.head_num, -1)
            self.v_buffer[idx][loc] = v_q["packed_indices"].reshape(num_tokens, self.head_num, -1)
            self.k_norms_buffer[idx][loc] = k_q["norms"].reshape(num_tokens, self.head_num)
            self.v_norms_buffer[idx][loc] = v_q["norms"].reshape(num_tokens, self.head_num)

        if not self.is_mixed and self.mode == "prod":
            self.k_qjl_buffer[idx][loc] = k_q["qjl_signs"].reshape(num_tokens, self.head_num, -1)
            self.v_qjl_buffer[idx][loc] = v_q["qjl_signs"].reshape(num_tokens, self.head_num, -1)
            self.k_residual_norms_buffer[idx][loc] = k_q["residual_norms"].reshape(num_tokens, self.head_num)
            self.v_residual_norms_buffer[idx][loc] = v_q["residual_norms"].reshape(num_tokens, self.head_num)

        # Write-through: dequant into per-layer buffers
        self._dequant_to_wt(k_q, v_q, loc, idx, num_tokens)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if tgt_loc.numel() == 0:
            return
        for i in range(self.layer_num):
            self.k_buffer[i][tgt_loc] = self.k_buffer[i][src_loc]
            self.v_buffer[i][tgt_loc] = self.v_buffer[i][src_loc]
            self.k_norms_buffer[i][tgt_loc] = self.k_norms_buffer[i][src_loc]
            self.v_norms_buffer[i][tgt_loc] = self.v_norms_buffer[i][src_loc]
            self._k_wt[i][tgt_loc] = self._k_wt[i][src_loc]
            self._v_wt[i][tgt_loc] = self._v_wt[i][src_loc]
            if self.mode == "prod":
                self.k_qjl_buffer[i][tgt_loc] = self.k_qjl_buffer[i][src_loc]
                self.v_qjl_buffer[i][tgt_loc] = self.v_qjl_buffer[i][src_loc]
                self.k_residual_norms_buffer[i][tgt_loc] = self.k_residual_norms_buffer[i][src_loc]
                self.v_residual_norms_buffer[i][tgt_loc] = self.v_residual_norms_buffer[i][src_loc]
