"""Unit tests for SVD visual slot release."""

import unittest

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolSVD
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestSVDVisualSlotRelease(CustomTestCase):
    def _make_pool(self, size=64, page_size=4):
        return MHATokenToKVPoolSVD(
            size=size,
            page_size=page_size,
            dtype=torch.float16,
            head_num=4,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            rank_k=4,
            rank_v=4,
        )

    def _write_visual_kv(self, pool, final_slots):
        k_data = torch.randn(len(final_slots), 4, 8, dtype=torch.float16)
        v_data = torch.randn(len(final_slots), 4, 8, dtype=torch.float16)
        pool.k_buffer[0][final_slots] = k_data.to(pool.store_dtype)
        pool.v_buffer[0][final_slots] = v_data.to(pool.store_dtype)

    def test_release_visual_slots_skips_shared_pages(self):
        pool = self._make_pool()
        allocator = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )

        req_pool_idx = 0
        shared_page = allocator.alloc(4)
        other_page = allocator.alloc(4)
        visual_positions = torch.tensor([2, 3], dtype=torch.int64)
        req_to_token_row = torch.tensor(
            [
                shared_page[0].item(),
                shared_page[1].item(),
                shared_page[2].item(),
                shared_page[3].item(),
                other_page[0].item(),
            ],
            dtype=torch.int32,
        )
        final_slots = shared_page[2:].clone()

        self._write_visual_kv(pool, final_slots)
        pool.compress_all_layers(
            req_pool_idx=req_pool_idx, visual_slot_indices=final_slots
        )

        released = pool.release_visual_slots_after_prefill(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            visual_token_positions=visual_positions,
            token_to_kv_pool_allocator=allocator,
        )

        vis_cache = pool.get_visual_cache(req_pool_idx)
        self.assertFalse(released)
        self.assertTrue(
            torch.equal(
                req_to_token_row,
                torch.tensor(
                    [
                        shared_page[0].item(),
                        shared_page[1].item(),
                        shared_page[2].item(),
                        shared_page[3].item(),
                        other_page[0].item(),
                    ],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertFalse(vis_cache.released_dense_visual_slots)
        self.assertTrue(torch.equal(vis_cache.visual_slot_indices, final_slots))

    def test_release_visual_slots_frees_page_exclusive_visual_kv(self):
        pool = self._make_pool()
        allocator = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )

        req_pool_idx = 0
        text_page = allocator.alloc(4)
        visual_page = allocator.alloc(4)
        visual_positions = torch.tensor([3, 4], dtype=torch.int64)
        req_to_token_row = torch.tensor(
            [
                text_page[0].item(),
                text_page[1].item(),
                text_page[2].item(),
                visual_page[0].item(),
                visual_page[1].item(),
            ],
            dtype=torch.int32,
        )
        final_slots = visual_page[:2].clone()

        self._write_visual_kv(pool, final_slots)
        pool.compress_all_layers(
            req_pool_idx=req_pool_idx, visual_slot_indices=final_slots
        )

        released = pool.release_visual_slots_after_prefill(
            req_pool_idx=req_pool_idx,
            req_to_token_row=req_to_token_row,
            visual_token_positions=visual_positions,
            token_to_kv_pool_allocator=allocator,
        )

        vis_cache = pool.get_visual_cache(req_pool_idx)
        self.assertTrue(released)
        self.assertTrue(
            torch.equal(
                req_to_token_row,
                torch.tensor(
                    [
                        text_page[0].item(),
                        text_page[1].item(),
                        text_page[2].item(),
                        0,
                        0,
                    ],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertTrue(vis_cache.released_dense_visual_slots)
        self.assertIsNone(vis_cache.visual_slot_indices)
        self.assertTrue(torch.equal(vis_cache.visual_token_positions, visual_positions))


if __name__ == "__main__":
    unittest.main()
