"""Unit tests for SVD-specific FlashInfer backend behavior."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolSVD
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestFlashInferBackendSVD(CustomTestCase):
    def _make_pool(self, size=64, device="cpu"):
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

    def test_decode_init_cleans_up_scratch_on_update_failure(self):
        caches = {
            0: SimpleNamespace(
                is_compressed=True,
                released_dense_visual_slots=True,
                visual_slot_indices=None,
                visual_token_positions=torch.tensor([1, 3], dtype=torch.int64),
                num_tokens=2,
                scratch_slot_allocation=None,
            )
        }
        kv_pool = MHATokenToKVPoolSVD.__new__(MHATokenToKVPoolSVD)
        kv_pool.get_visual_cache = lambda req_pool_idx: caches[req_pool_idx]

        def _activate(req_pool_idx, req_to_token_row, token_to_kv_pool_allocator):
            scratch = torch.tensor([101, 102], dtype=torch.int64)
            vis_cache = caches[req_pool_idx]
            req_to_token_row[vis_cache.visual_token_positions] = scratch.to(
                req_to_token_row.dtype
            )
            vis_cache.visual_slot_indices = scratch.clone()
            vis_cache.scratch_slot_allocation = scratch.clone()
            return scratch

        def _deactivate(req_pool_idx, req_to_token_row, token_to_kv_pool_allocator):
            vis_cache = caches[req_pool_idx]
            req_to_token_row[vis_cache.visual_token_positions] = 0
            vis_cache.visual_slot_indices = None
            vis_cache.scratch_slot_allocation = None

        kv_pool.activate_visual_scratch = Mock(side_effect=_activate)
        kv_pool.deactivate_visual_scratch = Mock(side_effect=_deactivate)

        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend.svd_enabled = True
        backend.decode_wrappers = []
        backend.decode_split_tile_size = None
        backend.indices_updater_decode = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=1,
                available_size=lambda: 8,
            ),
            update=Mock(side_effect=RuntimeError("decode metadata update failed")),
        )

        req_to_token = torch.tensor([[20, 0, 21, 0]], dtype=torch.int32)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            batch_size=1,
            token_to_kv_pool=kv_pool,
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
            seq_lens_sum=4,
            encoder_lens=None,
            spec_info=None,
        )

        with self.assertRaisesRegex(RuntimeError, "decode metadata update failed"):
            backend.init_forward_metadata(forward_batch)

        kv_pool.deactivate_visual_scratch.assert_called_once()
        self.assertTrue(
            torch.equal(
                req_to_token,
                torch.tensor([[20, 0, 21, 0]], dtype=torch.int32),
            )
        )
        self.assertIsNone(caches[0].visual_slot_indices)
        self.assertIsNone(caches[0].scratch_slot_allocation)

    def test_unreleased_compressed_visual_slots_keep_fused_decode_eligible(self):
        pool = self._make_pool()
        req_pool_idx = 0
        final_slots = torch.tensor([5, 9], dtype=torch.int64)
        k_data = torch.randn(2, 4, 8, dtype=torch.float16)
        v_data = torch.randn(2, 4, 8, dtype=torch.float16)

        pool.k_buffer[0][final_slots] = k_data.to(pool.store_dtype)
        pool.v_buffer[0][final_slots] = v_data.to(pool.store_dtype)
        pool.compress_all_layers(
            req_pool_idx=req_pool_idx, visual_slot_indices=final_slots
        )

        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend.svd_enabled = True
        backend.indices_updater_decode = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=1,
                available_size=lambda: 32,
            )
        )

        req_to_token = torch.tensor([[20, 5, 21, 9]], dtype=torch.int32)

        has_svd_scratch = backend._maybe_activate_svd_scratch(
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            batch_size=1,
            kv_pool=pool,
            req_to_token=req_to_token,
        )

        self.assertFalse(has_svd_scratch)
        self.assertTrue(
            torch.equal(
                req_to_token, torch.tensor([[20, 5, 21, 9]], dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(pool.get_visual_cache(req_pool_idx).visual_slot_indices, final_slots)
        )


if __name__ == "__main__":
    unittest.main()
