"""
Root conftest for FoxML test suite.

Mocks torch if not installed so LIVE_TRADING imports don't fail.
"""

import sys
from unittest.mock import MagicMock

# Mock torch if not installed (needed for TRAINING.common.live.seq_ring_buffer)
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_mock = MagicMock()
        torch_mock.__version__ = "0.0.0-mock"
        # Tensor operations needed by SeqRingBuffer
        torch_mock.zeros = MagicMock(side_effect=lambda *a, **kw: MagicMock())
        torch_mock.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        sys.modules["torch"] = torch_mock
