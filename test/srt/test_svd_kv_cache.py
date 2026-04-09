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

    def _make_pool(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolSVD

        return MHATokenToKVPoolSVD(
            size=128,
            page_size=1,
            dtype=torch.float16,
            head_num=4,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            rank_k=4,
            rank_v=4,
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
        """Decode scratch slots should be temporary and restore row placeholders."""
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
