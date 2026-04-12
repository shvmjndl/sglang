"""Unit tests for SVD handling in scheduler output processing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestSchedulerOutputProcessorSVD(CustomTestCase):
    class _DummyReq:
        def __init__(self, rid):
            self.rid = rid
            self._finished = False
            self.is_retracted = False
            self.is_chunked = 0
            self.output_ids = []
            self.return_logprob = False
            self.return_hidden_states = False
            self.grammar = None
            self.origin_input_ids = [1, 2, 3]
            self.time_stats = SimpleNamespace(
                set_prefill_finished_time=Mock(),
                set_completion_time=Mock(),
                set_last_chunked_prefill_finish_time=Mock(),
            )

        def finished(self):
            return self._finished

        def check_finished(self):
            return None

    def test_mixed_batch_decode_reqs_skip_prefill_finalization(self):
        prefill_req = self._DummyReq("prefill")
        decode_req = self._DummyReq("decode")
        scheduler = SimpleNamespace(
            is_generation=True,
            tree_cache=SimpleNamespace(cache_unfinished_req=Mock()),
            enable_hisparse=False,
            maybe_collect_routed_experts=Mock(),
            maybe_collect_customized_info=Mock(),
            _svd_compress_after_prefill=Mock(),
            stream_output=Mock(),
            report_prefill_stats=Mock(),
            abort_request=Mock(),
        )
        batch = SimpleNamespace(
            reqs=[prefill_req, decode_req],
            decoding_reqs=[decode_req],
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            logits_output=SimpleNamespace(
                next_token_logprobs=None,
                input_token_logprobs=None,
                hidden_states=None,
                customized_info=None,
            ),
            next_token_ids=torch.tensor([17, 23], dtype=torch.int64),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            can_run_cuda_graph=False,
        )

        SchedulerOutputProcessorMixin.process_batch_result_prefill(
            scheduler, batch, result
        )

        scheduler.tree_cache.cache_unfinished_req.assert_called_once_with(prefill_req)
        scheduler._svd_compress_after_prefill.assert_called_once_with(prefill_req)
        self.assertEqual(prefill_req.output_ids, [17])
        self.assertEqual(decode_req.output_ids, [23])


if __name__ == "__main__":
    unittest.main()
