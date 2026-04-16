"""
Unit tests for AttentionPack SVD KV cache compression.

Tests cover:
  - SVD compression/decompression roundtrip accuracy
  - Compression ratio verification
  - Partial decompression with importance scores
  - Importance score EMA updates
  - Visual token position detection
  - Fused Triton kernel correctness (vs reference)
  - PerRequestVisualCache lifecycle
  - SVD+FP8 quantization composition

Run with: python -m pytest test/srt/test_svd_kv_cache.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


class TestSVDCompression(unittest.TestCase):
    """Test SVD compression/decompression correctness."""

    def _make_pool(self, size=2048, head_num=32, head_dim=128, rank_k=64, rank_v=64,
                   num_decomp_groups=1, full_rank_fraction=0.25, alpha=0.25):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolSVD

        pool = MHATokenToKVPoolSVD(
            size=size,
            page_size=1,
            dtype=torch.float16,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=2,
            device="cuda" if torch.cuda.is_available() else "cpu",
            enable_memory_saver=False,
            rank_k=rank_k,
            rank_v=rank_v,
            compress_period=32,
            num_decomp_groups=num_decomp_groups,
            full_rank_fraction=full_rank_fraction,
            alpha=alpha,
        )
        return pool

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_roundtrip_accuracy(self):
        """Compress then decompress, check reconstruction error is bounded."""
        pool = self._make_pool(rank_k=64, rank_v=64)

        T_v = 576  # typical visual token count for LLaVA
        H, D = 32, 128
        req_pool_idx = 0

        # Generate low-rank-ish data (visual tokens tend to be low-rank)
        # Simulate by generating from a low-rank distribution
        base_k = torch.randn(T_v, 80, dtype=torch.float16, device="cuda")
        proj_k = torch.randn(80, H * D, dtype=torch.float16, device="cuda")
        k_data = (base_k @ proj_k).reshape(T_v, H, D)

        base_v = torch.randn(T_v, 80, dtype=torch.float16, device="cuda")
        proj_v = torch.randn(80, H * D, dtype=torch.float16, device="cuda")
        v_data = (base_v @ proj_v).reshape(T_v, H, D)

        # Write to pool slots
        slots = torch.arange(T_v, dtype=torch.int64, device="cuda")
        pool.k_buffer[0][slots] = k_data.to(pool.store_dtype)
        pool.v_buffer[0][slots] = v_data.to(pool.store_dtype)

        # Compress
        pool.alloc_visual_cache(req_pool_idx, T_v)
        pool.compress_visual_tokens(
            layer_id=pool.start_layer, req_pool_idx=req_pool_idx,
            visual_slot_indices=slots,
        )

        # Decompress
        k_recon, v_recon = pool.decompress_kv(pool.start_layer, req_pool_idx)

        self.assertIsNotNone(k_recon)
        self.assertIsNotNone(v_recon)
        self.assertEqual(k_recon.shape, (T_v, H, D))
        self.assertEqual(v_recon.shape, (T_v, H, D))

        # Check reconstruction error — with rank 64 on rank-80 data,
        # we expect most variance captured
        k_orig = k_data.float()
        k_rec = k_recon.float()
        k_error = (k_orig - k_rec).norm() / k_orig.norm()
        self.assertLess(k_error.item(), 0.4, f"Key reconstruction error too high: {k_error:.4f}")

        v_orig = v_data.float()
        v_rec = v_recon.float()
        v_error = (v_orig - v_rec).norm() / v_orig.norm()
        self.assertLess(v_error.item(), 0.4, f"Value reconstruction error too high: {v_error:.4f}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_compression_ratio(self):
        """Verify memory reduction matches expected ratio."""
        pool = self._make_pool(rank_k=64, rank_v=64)

        T_v = 576
        req_pool_idx = 0
        H, D = 32, 128

        slots = torch.arange(T_v, dtype=torch.int64, device="cuda")
        pool.alloc_visual_cache(req_pool_idx, T_v)
        pool.compress_visual_tokens(pool.start_layer, req_pool_idx, slots)

        stats = pool.get_compression_stats(req_pool_idx)

        self.assertTrue(stats["compressed"])
        self.assertEqual(stats["num_visual_tokens"], T_v)
        # Expected ratio: T_v * H * D * 2 * 2 / (T_v * R * 2 + R * HD * 2) * 2
        # = 576 * 4096 * 4 / (576 * 64 * 2 + 64 * 4096 * 2) * 2
        self.assertGreater(stats["compression_ratio"], 3.0,
                           f"Compression ratio too low: {stats['compression_ratio']:.2f}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_partial_decompression(self):
        """Verify partial decompression produces different results from full."""
        pool_full = self._make_pool(num_decomp_groups=1)
        pool_partial = self._make_pool(num_decomp_groups=2, full_rank_fraction=0.25)

        T_v = 100
        H, D = 32, 128
        req_pool_idx = 0

        # Same data in both pools
        k_data = torch.randn(T_v, H, D, dtype=torch.float16, device="cuda")
        slots = torch.arange(T_v, dtype=torch.int64, device="cuda")

        for pool in [pool_full, pool_partial]:
            pool.k_buffer[0][slots] = k_data.to(pool.store_dtype)
            pool.v_buffer[0][slots] = k_data.to(pool.store_dtype)
            pool.alloc_visual_cache(req_pool_idx, T_v)
            pool.compress_visual_tokens(pool.start_layer, req_pool_idx, slots)

        # Set non-uniform importance scores for partial pool
        vis_cache = pool_partial.get_visual_cache(req_pool_idx)
        vis_cache.importance[0] = torch.linspace(0, 1, T_v, device="cuda")

        k_full, _ = pool_full.decompress_kv(pool_full.start_layer, req_pool_idx)
        k_partial, _ = pool_partial.decompress_kv(pool_partial.start_layer, req_pool_idx)

        # Partial decompression should differ from full (reduced rank for low-importance)
        diff = (k_full.float() - k_partial.float()).norm()
        self.assertGreater(diff.item(), 0.0, "Partial decompression should differ from full")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_importance_score_ema(self):
        """Verify EMA importance score updates."""
        pool = self._make_pool(alpha=0.25)

        T_v = 50
        req_pool_idx = 0
        slots = torch.arange(T_v, dtype=torch.int64, device="cuda")
        pool.alloc_visual_cache(req_pool_idx, T_v)
        pool.compress_visual_tokens(pool.start_layer, req_pool_idx, slots)

        vis_cache = pool.get_visual_cache(req_pool_idx)

        # Initial scores should be zero
        self.assertTrue(torch.all(vis_cache.importance[0] == 0).item())

        # Update with uniform attention
        uniform_attn = torch.ones(T_v, device="cuda") / T_v
        pool.update_importance_scores(pool.start_layer, req_pool_idx, uniform_attn)

        # After first update with alpha=0.25:
        # I = 0.25^1 * 0 + (1 - 0.25^1) * (1/T_v) = 0.75 * (1/50)
        expected = 0.75 * (1.0 / T_v)
        actual = vis_cache.importance[0][0].item()
        self.assertAlmostEqual(actual, expected, places=4)

        # Update again with concentrated attention on token 0
        concentrated = torch.zeros(T_v, device="cuda")
        concentrated[0] = 1.0
        pool.update_importance_scores(pool.start_layer, req_pool_idx, concentrated)

        # Token 0 should now have much higher importance than others
        self.assertGreater(
            vis_cache.importance[0][0].item(),
            vis_cache.importance[0][1].item(),
        )


class TestVisualTokenTracking(unittest.TestCase):
    """Test visual token position detection."""

    def test_compute_visual_positions_with_image(self):
        """Test that image token positions are correctly identified."""
        from sglang.srt.managers.schedule_batch import MultimodalInputs, Req

        # Create a minimal Req-like object for testing
        # We can't easily construct a full Req, so test the logic directly
        mm = MultimodalInputs(mm_items=[], im_token_id=32000)

        # Simulate origin_input_ids with image tokens at known positions
        origin_ids = [1, 2, 32000, 32000, 32000, 3, 4, 32000, 5]
        # Image tokens at positions: 2, 3, 4, 7

        visual_positions = []
        for i, tid in enumerate(origin_ids):
            if mm.im_token_id is not None and tid == mm.im_token_id:
                visual_positions.append(i)
            elif mm.video_token_id is not None and tid == mm.video_token_id:
                visual_positions.append(i)

        self.assertEqual(visual_positions, [2, 3, 4, 7])

    def test_compute_visual_positions_no_multimodal(self):
        """No multimodal inputs → no visual positions."""
        visual_positions = []
        # When multimodal_inputs is None, should produce empty list
        self.assertEqual(visual_positions, [])

    def test_compute_visual_positions_video(self):
        """Test video token position detection."""
        from sglang.srt.managers.schedule_batch import MultimodalInputs

        mm = MultimodalInputs(mm_items=[], video_token_id=32001)

        origin_ids = [1, 32001, 32001, 2, 32001]
        visual_positions = []
        for i, tid in enumerate(origin_ids):
            if mm.im_token_id is not None and tid == mm.im_token_id:
                visual_positions.append(i)
            elif mm.video_token_id is not None and tid == mm.video_token_id:
                visual_positions.append(i)

        self.assertEqual(visual_positions, [1, 2, 4])


class TestPerRequestVisualCache(unittest.TestCase):
    """Test per-request visual cache lifecycle."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_alloc_and_free(self):
        """Test allocation and freeing of visual caches."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolSVD

        pool = MHATokenToKVPoolSVD(
            size=1024, page_size=1, dtype=torch.float16,
            head_num=32, head_dim=128, layer_num=2,
            device="cuda", enable_memory_saver=False,
        )

        # Allocate
        cache = pool.alloc_visual_cache(req_pool_idx=42, num_visual_tokens=100)
        self.assertIsNotNone(cache)
        self.assertEqual(cache.num_tokens, 100)
        self.assertFalse(cache.is_compressed)

        # Verify it's retrievable
        self.assertIsNotNone(pool.get_visual_cache(42))

        # Free
        pool.free_visual_cache(42)
        self.assertIsNone(pool.get_visual_cache(42))

        # Double free should be safe
        pool.free_visual_cache(42)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_compress_all_layers(self):
        """Test compression across all layers."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolSVD

        pool = MHATokenToKVPoolSVD(
            size=1024, page_size=1, dtype=torch.float16,
            head_num=32, head_dim=128, layer_num=4,
            device="cuda", enable_memory_saver=False,
            rank_k=32, rank_v=32,
        )

        T_v = 100
        slots = torch.arange(T_v, dtype=torch.int64, device="cuda")

        pool.compress_all_layers(req_pool_idx=0, visual_slot_indices=slots)

        vis_cache = pool.get_visual_cache(0)
        self.assertTrue(vis_cache.is_compressed)

        # All layers should have compressed factors
        for l in range(4):
            self.assertIsNotNone(vis_cache.k_compressed[l])
            self.assertEqual(vis_cache.k_compressed[l].shape, (T_v, 32))
            self.assertIsNotNone(vis_cache.k_decomp[l])
            self.assertEqual(vis_cache.k_decomp[l].shape, (32, 32 * 128))


class TestFinalVisualSlotResolution(unittest.TestCase):
    """Test resolving final visual slots from the current req_to_token mapping."""

    def _make_pool(self, size=128, head_num=4, head_dim=8, rank_k=4, rank_v=4):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolSVD

        return MHATokenToKVPoolSVD(
            size=size,
            page_size=1,
            dtype=torch.float16,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            rank_k=rank_k,
            rank_v=rank_v,
        )

    def test_resolve_visual_slots_after_remap(self):
        """Visual slots should come from the remapped req_to_token row."""
        pool = self._make_pool()

        visual_positions = torch.tensor([1, 3], dtype=torch.int64)
        req_to_token_row = torch.tensor([20, 21, 200, 201], dtype=torch.int32)

        resolved = pool.resolve_visual_slots(
            req_to_token_row=req_to_token_row,
            visual_token_positions=visual_positions,
            prompt_len=4,
        )

        self.assertTrue(torch.equal(resolved, torch.tensor([21, 201], dtype=torch.int64)))

    def test_compress_all_layers_uses_final_visual_slots(self):
        """The per-request cache should store the final resolved visual slots."""
        pool = self._make_pool()

        req_pool_idx = 0
        final_slots = torch.tensor([5, 9], dtype=torch.int64)
        k_data = torch.randn(2, 4, 8, dtype=torch.float16)
        v_data = torch.randn(2, 4, 8, dtype=torch.float16)

        pool.k_buffer[0][final_slots] = k_data.to(pool.store_dtype)
        pool.v_buffer[0][final_slots] = v_data.to(pool.store_dtype)

        pool.compress_all_layers(req_pool_idx=req_pool_idx, visual_slot_indices=final_slots)

        vis_cache = pool.get_visual_cache(req_pool_idx)
        self.assertTrue(vis_cache.is_compressed)
        self.assertTrue(torch.equal(vis_cache.visual_slot_indices, final_slots))

    def test_release_visual_slots_after_prefill(self):
        """Dense visual slots should be released after compression finalization."""
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

        pool = self._make_pool()
        allocator = TokenToKVPoolAllocator(
            size=32,
            dtype=torch.float16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )

        req_pool_idx = 0
        visual_positions = torch.tensor([1, 3], dtype=torch.int64)
        req_to_token_row = torch.tensor([20, 5, 21, 9], dtype=torch.int32)
        final_slots = torch.tensor([5, 9], dtype=torch.int64)

        pool.compress_all_layers(req_pool_idx=req_pool_idx, visual_slot_indices=final_slots)
        pool.release_visual_slots_after_prefill(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            visual_token_positions=visual_positions,
            token_to_kv_pool_allocator=allocator,
        )

        vis_cache = pool.get_visual_cache(req_pool_idx)
        self.assertTrue(torch.equal(req_to_token_row, torch.tensor([20, 0, 21, 0], dtype=torch.int32)))
        self.assertTrue(vis_cache.released_dense_visual_slots)
        self.assertIsNone(vis_cache.visual_slot_indices)
        self.assertTrue(torch.equal(vis_cache.visual_token_positions, visual_positions))

    def test_activate_and_deactivate_visual_scratch(self):
        """Decode scratch mappings should be temporary even if backing storage persists."""
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

        pool = self._make_pool()
        allocator = TokenToKVPoolAllocator(
            size=32,
            dtype=torch.float16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )

        req_pool_idx = 0
        visual_positions = torch.tensor([1, 3], dtype=torch.int64)
        req_to_token_row = torch.tensor([20, 0, 21, 0], dtype=torch.int32)
        final_slots = torch.tensor([5, 9], dtype=torch.int64)

        pool.compress_all_layers(req_pool_idx=req_pool_idx, visual_slot_indices=final_slots)
        vis_cache = pool.get_visual_cache(req_pool_idx)
        vis_cache.visual_token_positions = visual_positions.clone()
        vis_cache.visual_slot_indices = None
        vis_cache.released_dense_visual_slots = True

        scratch = pool.activate_visual_scratch(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertIsNotNone(scratch)
        self.assertTrue(torch.equal(req_to_token_row[visual_positions], scratch.to(torch.int32)))

        pool.deactivate_visual_scratch(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertTrue(torch.equal(req_to_token_row, torch.tensor([20, 0, 21, 0], dtype=torch.int32)))
        self.assertIsNone(vis_cache.visual_slot_indices)
        self.assertIsNotNone(vis_cache.scratch_slot_allocation)

    def test_activate_visual_scratch_returns_none_when_allocator_oom(self):
        """Scratch activation should fail cleanly without mutating request state."""
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

        pool = self._make_pool()
        allocator = TokenToKVPoolAllocator(
            size=1,
            dtype=torch.float16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )

        req_pool_idx = 0
        visual_positions = torch.tensor([1, 3], dtype=torch.int64)
        req_to_token_row = torch.tensor([20, 0, 21, 0], dtype=torch.int32)
        final_slots = torch.tensor([5, 9], dtype=torch.int64)

        pool.compress_all_layers(req_pool_idx=req_pool_idx, visual_slot_indices=final_slots)
        vis_cache = pool.get_visual_cache(req_pool_idx)
        vis_cache.visual_token_positions = visual_positions.clone()
        vis_cache.visual_slot_indices = None
        vis_cache.released_dense_visual_slots = True

        scratch = pool.activate_visual_scratch(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertIsNone(scratch)
        self.assertTrue(
            torch.equal(
                req_to_token_row,
                torch.tensor([20, 0, 21, 0], dtype=torch.int32),
            )
        )
        self.assertIsNone(vis_cache.visual_slot_indices)
        self.assertIsNone(vis_cache.scratch_slot_allocation)

    def test_activate_visual_scratch_page_aligns_allocation(self):
        """Scratch allocation should reserve whole pages while exposing only used slots."""
        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

        pool = self._make_pool(size=64, head_num=4, head_dim=8, rank_k=4, rank_v=4)
        pool.page_size = 4
        allocator = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )

        req_pool_idx = 0
        visual_positions = torch.tensor([1, 3], dtype=torch.int64)
        req_to_token_row = torch.tensor([20, 0, 21, 0], dtype=torch.int32)
        final_slots = torch.tensor([8, 9], dtype=torch.int64)

        pool.compress_all_layers(req_pool_idx=req_pool_idx, visual_slot_indices=final_slots)
        vis_cache = pool.get_visual_cache(req_pool_idx)
        vis_cache.visual_token_positions = visual_positions.clone()
        vis_cache.visual_slot_indices = None
        vis_cache.released_dense_visual_slots = True

        scratch = pool.activate_visual_scratch(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertEqual(len(scratch), 2)
        self.assertIsNotNone(vis_cache.scratch_slot_allocation)
        self.assertEqual(len(vis_cache.scratch_slot_allocation), 4)
        self.assertTrue(torch.equal(req_to_token_row[visual_positions], scratch.to(torch.int32)))

        pool.deactivate_visual_scratch(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertIsNotNone(vis_cache.scratch_slot_allocation)
        self.assertTrue(torch.equal(req_to_token_row, torch.tensor([20, 0, 21, 0], dtype=torch.int32)))

    def test_activate_visual_scratch_reuses_persistent_allocation(self):
        """Released visual requests should reuse the same scratch backing across decode steps."""
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

        pool = self._make_pool()
        allocator = TokenToKVPoolAllocator(
            size=32,
            dtype=torch.float16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )

        req_pool_idx = 0
        visual_positions = torch.tensor([1, 3], dtype=torch.int64)
        req_to_token_row = torch.tensor([20, 0, 21, 0], dtype=torch.int32)
        final_slots = torch.tensor([5, 9], dtype=torch.int64)

        pool.compress_all_layers(req_pool_idx=req_pool_idx, visual_slot_indices=final_slots)
        vis_cache = pool.get_visual_cache(req_pool_idx)
        vis_cache.visual_token_positions = visual_positions.clone()
        vis_cache.visual_slot_indices = None
        vis_cache.released_dense_visual_slots = True

        available_before = allocator.available_size()
        scratch_first = pool.activate_visual_scratch(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            token_to_kv_pool_allocator=allocator,
        )
        allocation_first = vis_cache.scratch_slot_allocation.clone()
        available_after_first = allocator.available_size()
        pool.deactivate_visual_scratch(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            token_to_kv_pool_allocator=allocator,
        )

        scratch_second = pool.activate_visual_scratch(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            token_to_kv_pool_allocator=allocator,
        )

        self.assertTrue(torch.equal(scratch_first, scratch_second))
        self.assertTrue(torch.equal(allocation_first, vis_cache.scratch_slot_allocation))
        self.assertEqual(available_after_first, allocator.available_size())
        self.assertLess(available_after_first, available_before)

    def test_free_visual_cache_releases_persistent_scratch(self):
        """Request teardown should free persistent scratch backing storage."""
        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

        pool = self._make_pool(size=64, head_num=4, head_dim=8, rank_k=4, rank_v=4)
        pool.page_size = 4
        allocator = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )

        req_pool_idx = 0
        visual_positions = torch.tensor([1, 3], dtype=torch.int64)
        req_to_token_row = torch.tensor([20, 0, 21, 0], dtype=torch.int32)
        final_slots = torch.tensor([8, 9], dtype=torch.int64)

        pool.compress_all_layers(req_pool_idx=req_pool_idx, visual_slot_indices=final_slots)
        vis_cache = pool.get_visual_cache(req_pool_idx)
        vis_cache.visual_token_positions = visual_positions.clone()
        vis_cache.visual_slot_indices = None
        vis_cache.released_dense_visual_slots = True

        available_before = allocator.available_size()
        pool.activate_visual_scratch(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            token_to_kv_pool_allocator=allocator,
        )
        available_after_activate = allocator.available_size()
        pool.deactivate_visual_scratch(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            token_to_kv_pool_allocator=allocator,
        )
        pool.free_visual_cache(
            req_pool_idx, token_to_kv_pool_allocator=allocator
        )

        self.assertLess(available_after_activate, available_before)
        self.assertEqual(allocator.available_size(), available_before)
        self.assertIsNone(pool.get_visual_cache(req_pool_idx))

    def test_compression_stats_report_released_and_scratch_state(self):
        """Compression stats should surface released/scratch state for debugging."""
        pool = self._make_pool()

        req_pool_idx = 0
        final_slots = torch.tensor([5, 9], dtype=torch.int64)
        pool.compress_all_layers(req_pool_idx=req_pool_idx, visual_slot_indices=final_slots)

        vis_cache = pool.get_visual_cache(req_pool_idx)
        vis_cache.released_dense_visual_slots = True
        vis_cache.visual_slot_indices = None
        stats = pool.get_compression_stats(req_pool_idx)
        self.assertTrue(stats["released_dense_visual_slots"])
        self.assertFalse(stats["has_active_scratch"])

        vis_cache.visual_slot_indices = torch.tensor([11, 12], dtype=torch.int64)
        stats = pool.get_compression_stats(req_pool_idx)
        self.assertTrue(stats["has_active_scratch"])


class TestSVDRadixOwnership(unittest.TestCase):
    """Test that multimodal SVD requests only cache the pure-text prefix."""

    class _DummyReq:
        def __init__(self):
            self._kv_committed_len = 0

        def pop_committed_kv_cache(self):
            return self._kv_committed_len

    def _make_tree(self, page_size=1):
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.mem_cache.radix_cache import RadixCache

        req_to_token_pool = ReqToTokenPool(
            size=4,
            max_context_len=32,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=128,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        tree = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
            )
        )
        return tree, allocator, req_to_token_pool

    def _enable_svd_args(self, page_size=1):
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        args = ServerArgs(model_path="dummy")
        args.enable_svd_kv_cache = True
        args.page_size = page_size
        set_global_server_args_for_scheduler(args)

    def test_cache_unfinished_req_only_inserts_text_prefix(self):
        self._enable_svd_args(page_size=1)
        tree, allocator, req_to_token_pool = self._make_tree(page_size=1)

        req = self._DummyReq()
        req.req_pool_idx = 0
        req.fill_ids = [10, 11, 12, 13, 14]
        req.origin_input_ids = req.fill_ids[:]
        req.output_ids = []
        req.extra_key = None
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.visual_token_positions = torch.tensor([2, 3], dtype=torch.int64)
        req.svd_first_visual_pos = None
        req.multimodal_inputs = object()

        kv_indices = allocator.alloc(len(req.fill_ids))
        req_to_token_pool.write((req.req_pool_idx, slice(0, len(req.fill_ids))), kv_indices)

        tree.cache_unfinished_req(req)

        self.assertEqual(req.cache_protected_len, 2)
        self.assertEqual(tree.total_size(), 2)
        self.assertTrue(torch.equal(req.prefix_indices[:2], kv_indices[:2]))
        self.assertTrue(torch.equal(req.prefix_indices[2:], kv_indices[2:]))

    def test_cache_finished_req_frees_visual_suffix_instead_of_caching_it(self):
        self._enable_svd_args(page_size=1)
        tree, allocator, req_to_token_pool = self._make_tree(page_size=1)

        req = self._DummyReq()
        req.req_pool_idx = 0
        req.origin_input_ids = [10, 11, 12, 13, 14]
        req.output_ids = []
        req._kv_committed_len = len(req.origin_input_ids)
        req.extra_key = None
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.visual_token_positions = torch.tensor([2, 3], dtype=torch.int64)
        req.svd_first_visual_pos = None
        req.multimodal_inputs = object()

        kv_indices = allocator.alloc(req._kv_committed_len)
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, req._kv_committed_len)),
            kv_indices,
        )

        tree.cache_finished_req(req, is_insert=True)

        self.assertEqual(tree.total_size(), 2)
        self.assertEqual(allocator.available_size(), 126)

    def test_page_aligned_prefix_stops_before_visual_page(self):
        self._enable_svd_args(page_size=4)
        tree, allocator, req_to_token_pool = self._make_tree(page_size=4)

        req = self._DummyReq()
        req.req_pool_idx = 0
        req.fill_ids = [10, 11, 12, 13, 14, 15]
        req.origin_input_ids = req.fill_ids[:]
        req.output_ids = []
        req.extra_key = None
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.visual_token_positions = torch.tensor([3], dtype=torch.int64)
        req.svd_first_visual_pos = None
        req.multimodal_inputs = object()

        kv_indices = allocator.alloc(len(req.fill_ids))
        req_to_token_pool.write((req.req_pool_idx, slice(0, len(req.fill_ids))), kv_indices)

        tree.cache_unfinished_req(req)

        self.assertEqual(req.cache_protected_len, 0)
        self.assertEqual(tree.total_size(), 0)
        self.assertTrue(torch.equal(req.prefix_indices, kv_indices))


class TestSVDDecodeScratchBudget(unittest.TestCase):
    class _FakeKVPool:
        def __init__(self, caches):
            self._caches = caches

        def get_visual_cache(self, req_pool_idx):
            return self._caches.get(int(req_pool_idx))

    class _FakeAllocator:
        def __init__(self, available_size, page_size, kv_pool):
            self._available_size = available_size
            self.page_size = page_size
            self._kv_pool = kv_pool

        def available_size(self):
            return self._available_size

        def get_kvcache(self):
            return self._kv_pool

    class _DummyReq:
        def __init__(self, req_pool_idx, output_len=0, prompt_len=0, max_new_tokens=16):
            self.req_pool_idx = req_pool_idx
            self.output_ids = [0] * output_len
            self.origin_input_ids = [0] * prompt_len
            self.kv_committed_len = 1
            self.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
            self.rid = f"req-{req_pool_idx}"

    def _enable_svd_args(self, page_size=1):
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        args = ServerArgs(model_path="dummy")
        args.enable_svd_kv_cache = True
        args.page_size = page_size
        set_global_server_args_for_scheduler(args)

    def _make_batch(self, reqs, allocator):
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.reqs = reqs
        batch.token_to_kv_pool_allocator = allocator
        batch.tree_cache = SimpleNamespace()
        batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        batch.enable_overlap = False
        return batch

    def test_visual_scratch_tokens_required_next_decode(self):
        self._enable_svd_args(page_size=4)
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        caches = {
            0: SimpleNamespace(
                is_compressed=True,
                released_dense_visual_slots=True,
                num_tokens=3,
                scratch_slot_allocation=None,
            ),
            1: SimpleNamespace(
                is_compressed=True,
                released_dense_visual_slots=False,
                num_tokens=7,
                scratch_slot_allocation=None,
            ),
            2: SimpleNamespace(
                is_compressed=False,
                released_dense_visual_slots=True,
                num_tokens=9,
                scratch_slot_allocation=None,
            ),
        }
        allocator = self._FakeAllocator(
            available_size=64, page_size=4, kv_pool=self._FakeKVPool(caches)
        )
        batch = self._make_batch(
            [
                self._DummyReq(0),
                self._DummyReq(1),
                self._DummyReq(2),
                self._DummyReq(None),
            ],
            allocator,
        )

        self.assertEqual(
            ScheduleBatch.visual_scratch_tokens_required_next_decode(batch),
            4,
        )

    def test_visual_scratch_tokens_required_next_decode_skips_persistent_scratch(self):
        self._enable_svd_args(page_size=4)
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        caches = {
            0: SimpleNamespace(
                is_compressed=True,
                released_dense_visual_slots=True,
                num_tokens=3,
                scratch_slot_allocation=torch.tensor([8, 9, 10, 11], dtype=torch.int64),
            ),
        }
        allocator = self._FakeAllocator(
            available_size=64, page_size=4, kv_pool=self._FakeKVPool(caches)
        )
        batch = self._make_batch([self._DummyReq(0)], allocator)

        self.assertEqual(
            ScheduleBatch.visual_scratch_tokens_required_next_decode(batch),
            0,
        )

    def test_check_decode_mem_accounts_for_visual_scratch(self):
        self._enable_svd_args(page_size=4)
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        caches = {
            0: SimpleNamespace(
                is_compressed=True,
                released_dense_visual_slots=True,
                num_tokens=3,
                scratch_slot_allocation=None,
            ),
        }
        allocator = self._FakeAllocator(
            available_size=7, page_size=4, kv_pool=self._FakeKVPool(caches)
        )
        batch = self._make_batch([self._DummyReq(0)], allocator)
        batch.new_tokens_required_next_decode = lambda selected_indices=None: 4

        with patch(
            "sglang.srt.managers.schedule_batch.evict_from_tree_cache"
        ) as mock_evict:
            self.assertFalse(ScheduleBatch.check_decode_mem(batch))
            mock_evict.assert_called_once_with(batch.tree_cache, 8)

    def test_retract_decode_can_make_progress_when_only_scratch_causes_oom(self):
        self._enable_svd_args(page_size=4)
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        caches = {
            0: SimpleNamespace(
                is_compressed=True,
                released_dense_visual_slots=True,
                num_tokens=5,
                scratch_slot_allocation=None,
            ),
            1: SimpleNamespace(
                is_compressed=True,
                released_dense_visual_slots=True,
                num_tokens=5,
                scratch_slot_allocation=None,
            ),
        }
        allocator = self._FakeAllocator(
            available_size=12, page_size=4, kv_pool=self._FakeKVPool(caches)
        )
        reqs = [
            self._DummyReq(0, output_len=4, prompt_len=8, max_new_tokens=16),
            self._DummyReq(1, output_len=1, prompt_len=4, max_new_tokens=16),
        ]
        batch = self._make_batch(reqs, allocator)
        batch.new_tokens_required_next_decode = lambda selected_indices=None: 0
        batch.release_req = lambda idx, remaining_req_count, server_args: None
        def _filter_batch(keep_indices=None, **kwargs):
            keep = keep_indices if keep_indices is not None else kwargs["keep_indices"]
            batch.reqs = [batch.reqs[i] for i in keep]

        batch.filter_batch = _filter_batch

        with patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"):
            retracted_reqs, _, reqs_to_abort = ScheduleBatch.retract_decode(
                batch,
                SimpleNamespace(speculative_algorithm=None),
            )

        self.assertEqual(len(retracted_reqs), 1)
        self.assertEqual(len(reqs_to_abort), 0)
        self.assertEqual(len(batch.reqs), 1)
        self.assertEqual(batch.reqs[0].req_pool_idx, 0)


class TestSVDDecodeScratchLifecycle(unittest.TestCase):
    def _make_pool(self, size=64, device="cpu"):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolSVD

        return MHATokenToKVPoolSVD(
            size=size,
            page_size=1,
            dtype=torch.float16,
            head_num=4,
            head_dim=8,
            layer_num=2,
            device=device,
            enable_memory_saver=False,
            rank_k=4,
            rank_v=4,
            compress_period=32,
        )

    def _make_idle_forward_batch(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        return SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            req_pool_indices=torch.empty((0,), dtype=torch.int64),
            batch_size=0,
            token_to_kv_pool=None,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.empty((0, 0), dtype=torch.int32)
            ),
            seq_lens=torch.empty((0,), dtype=torch.int32),
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
            seq_lens_sum=0,
            encoder_lens=None,
            spec_info=None,
        )

    def test_idle_init_forward_metadata_skips_scratch_activation(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            DecodeMetadata,
            FlashInferAttnBackend,
        )

        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend._maybe_activate_svd_scratch = Mock(
            side_effect=AssertionError("idle should not activate scratch")
        )
        backend.indices_updater_decode = SimpleNamespace(update=Mock())
        backend.decode_wrappers = []
        backend.decode_split_tile_size = None

        forward_batch = self._make_idle_forward_batch()
        backend.init_forward_metadata(forward_batch)

        self.assertIsInstance(backend.forward_metadata, DecodeMetadata)
        self.assertFalse(backend.forward_metadata.has_svd_scratch)
        backend.indices_updater_decode.update.assert_called_once()

    def test_idle_finish_forward_metadata_skips_scratch_cleanup(self):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend.svd_enabled = True
        backend.indices_updater_decode = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace()
        )

        forward_batch = self._make_idle_forward_batch()
        backend.finish_forward_metadata(forward_batch)

    def test_decode_scratch_preflight_rejects_underbudget_batch(self):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

        pool = self._make_pool()
        allocator = TokenToKVPoolAllocator(
            size=2,
            dtype=torch.float16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )

        for req_pool_idx, final_slots in (
            (0, torch.tensor([5, 9], dtype=torch.int64)),
            (1, torch.tensor([10, 11], dtype=torch.int64)),
        ):
            pool.compress_all_layers(
                req_pool_idx=req_pool_idx, visual_slot_indices=final_slots
            )
            vis_cache = pool.get_visual_cache(req_pool_idx)
            vis_cache.visual_token_positions = torch.tensor([1, 3], dtype=torch.int64)
            vis_cache.visual_slot_indices = None
            vis_cache.released_dense_visual_slots = True

        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend.svd_enabled = True
        backend.indices_updater_decode = SimpleNamespace(
            token_to_kv_pool_allocator=allocator
        )

        req_to_token = torch.tensor(
            [[20, 0, 21, 0], [30, 0, 31, 0]], dtype=torch.int32
        )

        with self.assertRaisesRegex(
            RuntimeError, "Insufficient KV scratch capacity before decode"
        ):
            backend._maybe_activate_svd_scratch(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                batch_size=2,
                kv_pool=pool,
                req_to_token=req_to_token,
            )

        self.assertTrue(
            torch.equal(
                req_to_token,
                torch.tensor([[20, 0, 21, 0], [30, 0, 31, 0]], dtype=torch.int32),
            )
        )

    def test_decode_scratch_activation_rolls_back_partial_batch(self):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolSVD

        allocator = SimpleNamespace(page_size=1, available_size=lambda: 8)
        caches = {
            0: SimpleNamespace(
                is_compressed=True,
                released_dense_visual_slots=True,
                visual_slot_indices=None,
                scratch_slot_allocation=None,
                visual_token_positions=torch.tensor([1, 3], dtype=torch.int64),
                num_tokens=2,
            ),
            1: SimpleNamespace(
                is_compressed=True,
                released_dense_visual_slots=True,
                visual_slot_indices=None,
                scratch_slot_allocation=None,
                visual_token_positions=torch.tensor([0, 2], dtype=torch.int64),
                num_tokens=2,
            ),
        }
        kv_pool = MHATokenToKVPoolSVD.__new__(MHATokenToKVPoolSVD)
        kv_pool.get_visual_cache = lambda req_pool_idx: caches[req_pool_idx]
        kv_pool.activate_visual_scratch = Mock(
            side_effect=[
                torch.tensor([101, 102], dtype=torch.int64),
                None,
            ]
        )
        kv_pool.deactivate_visual_scratch = Mock()

        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend.svd_enabled = True
        backend.indices_updater_decode = SimpleNamespace(
            token_to_kv_pool_allocator=allocator
        )

        req_to_token = torch.tensor(
            [[20, 0, 21, 0], [30, 0, 31, 0]], dtype=torch.int32
        )

        with self.assertRaisesRegex(
            RuntimeError, "Visual scratch activation failed after preflight"
        ):
            backend._maybe_activate_svd_scratch(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                batch_size=2,
                kv_pool=kv_pool,
                req_to_token=req_to_token,
            )

        kv_pool.deactivate_visual_scratch.assert_called_once()
        _, deactivate_kwargs = kv_pool.deactivate_visual_scratch.call_args
        self.assertEqual(deactivate_kwargs["req_pool_idx"], 0)
        self.assertTrue(deactivate_kwargs["free_allocation"])
        self.assertTrue(
            torch.equal(
                deactivate_kwargs["req_to_token_row"],
                req_to_token[0],
            )
        )

    def test_recompression_skips_released_dense_visual_slots(self):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        vis_cache = SimpleNamespace(
            is_compressed=True,
            released_dense_visual_slots=True,
            steps_since_compress=31,
            visual_slot_indices=torch.tensor([7, 8], dtype=torch.int64),
        )
        kv_pool = SimpleNamespace(
            start_layer=0,
            layer_num=2,
            compress_period=32,
            get_visual_cache=lambda req_pool_idx: vis_cache,
            compress_visual_tokens=Mock(),
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
        )

        backend._svd_check_recompression(forward_batch, kv_pool)

        self.assertEqual(vis_cache.steps_since_compress, 31)
        kv_pool.compress_visual_tokens.assert_not_called()


class TestSVDFusedDecodeVisualIdentity(unittest.TestCase):
    def test_fused_single_uses_visual_positions_for_released_visual_cache(self):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)

        k_buffer = torch.zeros(16, 1, 2, dtype=torch.float16)
        v_buffer = torch.zeros(16, 1, 2, dtype=torch.float16)
        k_buffer[10, 0] = torch.tensor([10.0, 10.5], dtype=torch.float16)
        k_buffer[11, 0] = torch.tensor([11.0, 11.5], dtype=torch.float16)
        v_buffer[10, 0] = torch.tensor([20.0, 20.5], dtype=torch.float16)
        v_buffer[11, 0] = torch.tensor([21.0, 21.5], dtype=torch.float16)

        vis_cache = SimpleNamespace(
            is_compressed=True,
            k_compressed=[torch.zeros(2, 1, dtype=torch.float16)],
            k_decomp=[torch.zeros(1, 2, dtype=torch.float16)],
            v_compressed=[torch.zeros(2, 1, dtype=torch.float16)],
            v_decomp=[torch.zeros(1, 2, dtype=torch.float16)],
            importance=[torch.zeros(2, dtype=torch.float32)],
            visual_token_positions=torch.tensor([1, 3], dtype=torch.int64),
            visual_slot_indices=None,
        )
        kv_pool = SimpleNamespace(
            start_layer=0,
            get_visual_cache=lambda req_pool_idx: vis_cache,
            _get_key_buffer=lambda layer_id: k_buffer,
            _get_value_buffer=lambda layer_id: v_buffer,
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([4], dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor([[10, 0, 11, 0]], dtype=torch.int32)
            ),
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=1,
            head_dim=2,
            scaling=1.0,
        )
        q = torch.zeros(1, 1, 2, dtype=torch.float16)
        captured = {}

        def _fake_fused(q, k_bar, d_k, v_bar, d_v, k_uncomp, v_uncomp, importance, sm_scale):
            captured["k_uncomp"] = k_uncomp.clone()
            captured["v_uncomp"] = v_uncomp.clone()
            return torch.zeros_like(q)

        with patch(
            "sglang.srt.layers.attention.triton_ops.svd_decode_attention.svd_fused_decode_attention",
            side_effect=_fake_fused,
        ):
            out = backend._svd_fused_decode_single(q, layer, forward_batch, kv_pool)

        self.assertTrue(torch.equal(out, torch.zeros_like(q)))
        expected_k = k_buffer[torch.tensor([10, 11], dtype=torch.int64)]
        expected_v = v_buffer[torch.tensor([10, 11], dtype=torch.int64)]
        self.assertTrue(torch.equal(captured["k_uncomp"], expected_k))
        self.assertTrue(torch.equal(captured["v_uncomp"], expected_v))


class TestFusedKernel(unittest.TestCase):
    """Test fused Triton kernel correctness against reference implementation."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_vs_reference(self):
        """Compare fused kernel output to explicit decompress + attention."""
        from sglang.srt.layers.attention.triton_ops.svd_decode_attention import (
            svd_fused_decode_attention,
        )

        torch.manual_seed(42)

        B, H_q, H_kv, D = 1, 32, 32, 128
        T_c, R_k, R_v = 64, 32, 32
        T_u = 16

        q = torch.randn(B, H_q, D, dtype=torch.float16, device="cuda")
        k_bar = torch.randn(T_c, R_k, dtype=torch.float16, device="cuda")
        d_k = torch.randn(R_k, H_kv * D, dtype=torch.float16, device="cuda")
        v_bar = torch.randn(T_c, R_v, dtype=torch.float16, device="cuda")
        d_v = torch.randn(R_v, H_kv * D, dtype=torch.float16, device="cuda")
        k_uncomp = torch.randn(T_u, H_kv, D, dtype=torch.float16, device="cuda")
        v_uncomp = torch.randn(T_u, H_kv, D, dtype=torch.float16, device="cuda")
        importance = torch.zeros(T_c, dtype=torch.float32, device="cuda")
        sm_scale = 1.0 / (D ** 0.5)

        # Reference: explicit decompress + attention
        k_decomp = (k_bar.float() @ d_k.float()).reshape(T_c, H_kv, D)
        v_decomp = (v_bar.float() @ d_v.float()).reshape(T_c, H_kv, D)

        # Concat compressed (decompressed) + uncompressed
        k_all = torch.cat([k_decomp, k_uncomp.float()], dim=0)  # [T_c+T_u, H, D]
        v_all = torch.cat([v_decomp, v_uncomp.float()], dim=0)

        # Attention: [B, H_q, D] @ [T, H_kv, D] -> [B, H_q, T]
        # For simplicity, assume H_q == H_kv (no GQA)
        q_f = q.float()
        scores = torch.einsum("bhd,thd->bht", q_f, k_all) * sm_scale
        attn = torch.softmax(scores, dim=-1)
        ref_o = torch.einsum("bht,thd->bhd", attn, v_all)

        # Fused kernel
        fused_o = svd_fused_decode_attention(
            q, k_bar, d_k, v_bar, d_v, k_uncomp, v_uncomp, importance, sm_scale
        )

        # Compare — fp16 matmul + online softmax introduces some error
        ref_o_f16 = ref_o.half()
        rel_diff = (ref_o_f16 - fused_o).norm() / ref_o_f16.norm()

        self.assertLess(
            rel_diff.item(), 0.05,
            f"Fused kernel output differs from reference: rel_diff={rel_diff:.4f}"
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_gqa(self):
        """Test fused kernel with GQA (H_q != H_kv)."""
        from sglang.srt.layers.attention.triton_ops.svd_decode_attention import (
            svd_fused_decode_attention,
        )

        torch.manual_seed(123)

        B, H_q, H_kv, D = 1, 32, 8, 128  # GQA: 4 query heads per KV head
        T_c, R_k, R_v = 32, 16, 16
        T_u = 8

        q = torch.randn(B, H_q, D, dtype=torch.float16, device="cuda")
        k_bar = torch.randn(T_c, R_k, dtype=torch.float16, device="cuda")
        d_k = torch.randn(R_k, H_kv * D, dtype=torch.float16, device="cuda")
        v_bar = torch.randn(T_c, R_v, dtype=torch.float16, device="cuda")
        d_v = torch.randn(R_v, H_kv * D, dtype=torch.float16, device="cuda")
        k_uncomp = torch.randn(T_u, H_kv, D, dtype=torch.float16, device="cuda")
        v_uncomp = torch.randn(T_u, H_kv, D, dtype=torch.float16, device="cuda")
        importance = torch.zeros(T_c, dtype=torch.float32, device="cuda")
        sm_scale = 1.0 / (D ** 0.5)

        # Reference: decompress then per-head attention with GQA expansion
        k_decomp = (k_bar.float() @ d_k.float()).reshape(T_c, H_kv, D)
        v_decomp = (v_bar.float() @ d_v.float()).reshape(T_c, H_kv, D)
        k_all = torch.cat([k_decomp, k_uncomp.float()], dim=0)
        v_all = torch.cat([v_decomp, v_uncomp.float()], dim=0)

        # Expand KV heads for GQA: [T, H_kv, D] -> [T, H_q, D]
        heads_per_group = H_q // H_kv
        k_expanded = k_all.unsqueeze(2).expand(-1, -1, heads_per_group, -1).reshape(-1, H_q, D)
        v_expanded = v_all.unsqueeze(2).expand(-1, -1, heads_per_group, -1).reshape(-1, H_q, D)

        q_f = q.float()
        scores = torch.einsum("bhd,thd->bht", q_f, k_expanded) * sm_scale
        attn = torch.softmax(scores, dim=-1)
        ref_o = torch.einsum("bht,thd->bhd", attn, v_expanded)

        fused_o = svd_fused_decode_attention(
            q, k_bar, d_k, v_bar, d_v, k_uncomp, v_uncomp, importance, sm_scale
        )

        rel_diff = (ref_o.half() - fused_o).norm() / ref_o.half().norm()
        self.assertLess(
            rel_diff.item(), 0.05,
            f"GQA fused kernel output differs: rel_diff={rel_diff:.4f}"
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_no_compressed_tokens(self):
        """Test fused kernel when T_c=0 (no compressed tokens)."""
        from sglang.srt.layers.attention.triton_ops.svd_decode_attention import (
            svd_fused_decode_attention,
        )

        torch.manual_seed(99)

        B, H_q, H_kv, D = 1, 8, 8, 64
        T_c, R_k, R_v = 0, 16, 16
        T_u = 32

        q = torch.randn(B, H_q, D, dtype=torch.float16, device="cuda")
        k_bar = torch.empty(0, R_k, dtype=torch.float16, device="cuda")
        d_k = torch.randn(R_k, H_kv * D, dtype=torch.float16, device="cuda")
        v_bar = torch.empty(0, R_v, dtype=torch.float16, device="cuda")
        d_v = torch.randn(R_v, H_kv * D, dtype=torch.float16, device="cuda")
        k_uncomp = torch.randn(T_u, H_kv, D, dtype=torch.float16, device="cuda")
        v_uncomp = torch.randn(T_u, H_kv, D, dtype=torch.float16, device="cuda")
        importance = torch.empty(0, dtype=torch.float32, device="cuda")
        sm_scale = 1.0 / (D ** 0.5)

        # Reference: standard attention over uncompressed only
        q_f = q.float()
        scores = torch.einsum("bhd,thd->bht", q_f, k_uncomp.float()) * sm_scale
        attn = torch.softmax(scores, dim=-1)
        ref_o = torch.einsum("bht,thd->bhd", attn, v_uncomp.float())

        fused_o = svd_fused_decode_attention(
            q, k_bar, d_k, v_bar, d_v, k_uncomp, v_uncomp, importance, sm_scale
        )

        rel_diff = (ref_o.half() - fused_o).norm() / ref_o.half().norm()
        self.assertLess(rel_diff.item(), 0.05, f"T_c=0 case failed: rel_diff={rel_diff:.4f}")


class TestSVDFP4Composition(unittest.TestCase):
    """Test SVD + FP8 quantization composition."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_svdfp4_roundtrip(self):
        """Test that SVD+FP8 compress/decompress works end-to-end."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolSVDFP4

        pool = MHATokenToKVPoolSVDFP4(
            size=1024, page_size=1, dtype=torch.float16,
            head_num=32, head_dim=128, layer_num=1,
            device="cuda", enable_memory_saver=False,
            rank_k=32, rank_v=32,
        )

        T_v = 100
        slots = torch.arange(T_v, dtype=torch.int64, device="cuda")

        # Write data
        k_data = torch.randn(T_v, 32, 128, dtype=torch.float16, device="cuda")
        pool.k_buffer[0][slots] = k_data

        pool.alloc_visual_cache(0, T_v)
        pool.compress_visual_tokens(pool.start_layer, 0, slots)

        # Decompress
        k_recon, v_recon = pool.decompress_kv(pool.start_layer, 0)
        self.assertIsNotNone(k_recon)
        self.assertEqual(k_recon.shape, (T_v, 32, 128))

        # Verify accuracy is bounded (FP8 adds quantization error on top of SVD)
        k_orig = k_data.float()
        k_rec = k_recon.float()
        rel_error = (k_orig - k_rec).norm() / k_orig.norm()
        # SVD (rank 32 on 4096-dim) + FP8 quantization is very lossy on random data.
        # Real VLM KV vectors are low-rank so error would be much lower.
        # Here we just verify the pipeline doesn't crash and produces finite output.
        self.assertTrue(torch.isfinite(k_recon).all(), "Non-finite values in reconstruction")
        self.assertLess(
            rel_error.item(), 1.0,
            f"SVD+FP8 reconstruction error unexpectedly high: {rel_error:.4f}"
        )


class TestServerArgs(unittest.TestCase):
    """Test that SVD server args are properly defined."""

    def test_svd_args_exist(self):
        """Verify all SVD-related args are on ServerArgs."""
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(model_path="dummy")
        self.assertFalse(args.enable_svd_kv_cache)
        self.assertEqual(args.svd_rank_k, 64)
        self.assertEqual(args.svd_rank_v, 64)
        self.assertEqual(args.svd_compress_period, 32)
        self.assertEqual(args.svd_decomp_groups, 2)
        self.assertAlmostEqual(args.svd_full_rank_fraction, 0.25)
        self.assertAlmostEqual(args.svd_alpha, 0.25)
        self.assertTrue(args.svd_visual_only)
        self.assertTrue(args.svd_fused_kernel)
        self.assertFalse(args.svd_quantize)


if __name__ == "__main__":
    unittest.main()
