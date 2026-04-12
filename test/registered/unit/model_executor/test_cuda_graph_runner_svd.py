"""Unit tests for SVD cleanup in graph replay paths."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestCudaGraphRunnerSVD(CustomTestCase):
    def test_replay_cleans_up_when_prepare_fails(self):
        finish_forward_metadata = Mock()
        runner = CudaGraphRunner.__new__(CudaGraphRunner)
        runner.deepep_adapter = SimpleNamespace(replay=Mock())
        runner.enable_pdmux = False
        runner.model_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(
                finish_forward_metadata=finish_forward_metadata
            )
        )
        runner.replay_prepare = Mock(side_effect=RuntimeError("replay prepare failed"))

        with self.assertRaisesRegex(RuntimeError, "replay prepare failed"):
            runner.replay(SimpleNamespace())

        finish_forward_metadata.assert_called_once()

    def test_piecewise_replay_cleans_up_when_prepare_fails(self):
        init_forward_metadata = Mock()
        finish_forward_metadata = Mock()
        runner = PiecewiseCudaGraphRunner.__new__(PiecewiseCudaGraphRunner)
        runner.model_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(
                init_forward_metadata=init_forward_metadata,
                finish_forward_metadata=finish_forward_metadata,
            )
        )
        runner.replay_prepare = Mock(
            side_effect=RuntimeError("piecewise replay prepare failed")
        )

        with patch(
            "sglang.srt.model_executor.piecewise_cuda_graph_runner.enable_piecewise_cuda_graph",
            return_value=contextlib.nullcontext(),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "piecewise replay prepare failed"
            ):
                runner.replay(SimpleNamespace())

        init_forward_metadata.assert_called_once()
        finish_forward_metadata.assert_called_once()


if __name__ == "__main__":
    unittest.main()
