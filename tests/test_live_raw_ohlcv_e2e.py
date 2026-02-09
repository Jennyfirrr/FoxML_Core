"""
Integration tests for raw OHLCV inference path (Phase 2).

Tests end-to-end flow from DataFrame -> normalization -> buffer -> prediction.
"""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from TRAINING.training_strategies.utils import _normalize_ohlcv_sequence


# ---------------------------------------------------------------------------
# Test: Normalization matches training exactly
# ---------------------------------------------------------------------------

class TestRawOHLCVNormalization:

    def test_normalization_matches_training(self):
        """Normalization in live path uses the exact same function as training."""
        ohlcv = np.array([
            [100.0, 101.0, 99.0, 100.5, 1000.0],
            [100.5, 102.0, 100.0, 101.0, 1100.0],
            [101.0, 103.0, 100.5, 102.0, 900.0],
            [102.0, 102.5, 101.0, 101.5, 1200.0],
        ], dtype=np.float32)

        result_returns = _normalize_ohlcv_sequence(ohlcv, method="returns")
        result_log = _normalize_ohlcv_sequence(ohlcv, method="log_returns")
        result_minmax = _normalize_ohlcv_sequence(ohlcv, method="minmax")

        # Verify shapes preserved
        assert result_returns.shape == (4, 5)
        assert result_log.shape == (4, 5)
        assert result_minmax.shape == (4, 5)

        # Verify returns normalization: first bar close should be near zero
        assert abs(result_returns[0, 3]) < 0.01

        # Verify minmax: all values in [0, 1]
        assert result_minmax.min() >= -1e-6
        assert result_minmax.max() <= 1.0 + 1e-6

    def test_normalization_none_method(self):
        """'none' normalization returns copy of input."""
        ohlcv = np.array([
            [100.0, 101.0, 99.0, 100.5, 1000.0],
            [100.5, 102.0, 100.0, 101.0, 1100.0],
        ], dtype=np.float32)

        result = _normalize_ohlcv_sequence(ohlcv, method="none")
        np.testing.assert_array_equal(result, ohlcv)
        assert result is not ohlcv


# ---------------------------------------------------------------------------
# Test: Column case insensitivity
# ---------------------------------------------------------------------------

class TestColumnCaseInsensitive:

    def test_prepare_raw_sequence_capitalized_columns(self, tmp_path):
        """Column matching works for capitalized columns (broker data)."""
        from tests.test_live_inference_input_mode import _make_run_root
        from LIVE_TRADING.prediction.predictor import MultiHorizonPredictor

        metadata = {
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "input_mode": "raw_sequence",
            "sequence_length": 3,
            "sequence_channels": ["open", "high", "low", "close", "volume"],
            "sequence_normalization": "returns",
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        }
        run_root = _make_run_root(tmp_path, family="Ridge", metadata=metadata)
        predictor = MultiHorizonPredictor(run_root=str(run_root))

        prices = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000.0, 1100.0, 900.0, 1200.0, 800.0],
        })

        result = predictor._prepare_raw_sequence(prices, "ret_5m", "Ridge")
        assert result is not None
        assert result.shape == (5,)

    def test_prepare_raw_sequence_lowercase_columns(self, tmp_path):
        """Column matching works for lowercase columns (parquet data)."""
        from tests.test_live_inference_input_mode import _make_run_root
        from LIVE_TRADING.prediction.predictor import MultiHorizonPredictor

        metadata = {
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "input_mode": "raw_sequence",
            "sequence_length": 3,
            "sequence_channels": ["open", "high", "low", "close", "volume"],
            "sequence_normalization": "returns",
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        }
        run_root = _make_run_root(tmp_path, family="Ridge", metadata=metadata)
        predictor = MultiHorizonPredictor(run_root=str(run_root))

        prices = pd.DataFrame({
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000.0, 1100.0, 900.0, 1200.0, 800.0],
        })

        result = predictor._prepare_raw_sequence(prices, "ret_5m", "Ridge")
        assert result is not None
        assert result.shape == (5,)


# ---------------------------------------------------------------------------
# Test: Insufficient data handling
# ---------------------------------------------------------------------------

class TestInsufficientData:

    def test_prepare_raw_sequence_insufficient_data(self, tmp_path):
        """Returns None when insufficient bars available."""
        from tests.test_live_inference_input_mode import _make_run_root
        from LIVE_TRADING.prediction.predictor import MultiHorizonPredictor

        metadata = {
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "input_mode": "raw_sequence",
            "sequence_length": 64,
            "sequence_channels": ["open", "high", "low", "close", "volume"],
            "sequence_normalization": "returns",
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        }
        run_root = _make_run_root(tmp_path, family="Ridge", metadata=metadata)
        predictor = MultiHorizonPredictor(run_root=str(run_root))

        # Only 1 bar — insufficient
        prices = pd.DataFrame({
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1000.0],
        })

        result = predictor._prepare_raw_sequence(prices, "ret_5m", "Ridge")
        assert result is None

    def test_prepare_raw_sequence_missing_columns(self, tmp_path):
        """Returns None when OHLCV columns missing."""
        from tests.test_live_inference_input_mode import _make_run_root
        from LIVE_TRADING.prediction.predictor import MultiHorizonPredictor

        metadata = {
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "input_mode": "raw_sequence",
            "sequence_length": 3,
            "sequence_channels": ["open", "high", "low", "close", "volume"],
            "sequence_normalization": "returns",
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        }
        run_root = _make_run_root(tmp_path, family="Ridge", metadata=metadata)
        predictor = MultiHorizonPredictor(run_root=str(run_root))

        # Missing volume column
        prices = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
        })

        result = predictor._prepare_raw_sequence(prices, "ret_5m", "Ridge")
        assert result is None


# ---------------------------------------------------------------------------
# Test: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_no_input_mode_defaults_to_features(self, tmp_path):
        """Models without input_mode field work as feature-based (backward compat)."""
        from tests.test_live_inference_input_mode import _make_run_root
        from LIVE_TRADING.models.loader import ModelLoader

        metadata = {
            "family": "Ridge", "target": "ret_5m",
            "feature_list": ["rsi_14", "sma_20", "vol_10"],
            "n_features": 3,
            "metrics": {"mean_IC": 0.05},
            "model_checksum": None,
            "interval_minutes": 1.0,
        }
        run_root = _make_run_root(tmp_path, family="Ridge", metadata=metadata)
        loader = ModelLoader(run_root, verify_checksums=False)

        assert loader.get_input_mode("ret_5m", "Ridge") == "features"
        assert loader.get_feature_list("ret_5m", "Ridge") == ["rsi_14", "sma_20", "vol_10"]
        assert loader.get_sequence_config("ret_5m", "Ridge") == {}


# ---------------------------------------------------------------------------
# Test: Contract fields consumed
# ---------------------------------------------------------------------------

class TestContractFieldsConsumed:

    def test_all_raw_sequence_fields_consumed(self, tmp_path):
        """All INTEGRATION_CONTRACTS.md v1.3 raw_sequence fields are consumed."""
        from tests.test_live_inference_input_mode import _make_run_root
        from LIVE_TRADING.models.loader import ModelLoader
        from LIVE_TRADING.models.inference import InferenceEngine

        metadata = {
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "input_mode": "raw_sequence",            # CONTRACT field
            "sequence_length": 32,                    # CONTRACT field
            "sequence_channels": ["open", "high", "low", "close", "volume"],  # CONTRACT field
            "sequence_normalization": "log_returns",  # CONTRACT field
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        }
        run_root = _make_run_root(tmp_path, family="Ridge", metadata=metadata)
        loader = ModelLoader(run_root, verify_checksums=False)

        # 1. input_mode consumed by get_input_mode()
        assert loader.get_input_mode("ret_5m", "Ridge") == "raw_sequence"

        # 2. sequence_length, sequence_channels, sequence_normalization consumed
        config = loader.get_sequence_config("ret_5m", "Ridge")
        assert config["sequence_length"] == 32
        assert config["sequence_channels"] == ["open", "high", "low", "close", "volume"]
        assert config["sequence_normalization"] == "log_returns"

        # 3. Buffer uses sequence_channels for F dimension
        engine = InferenceEngine(loader, device="cpu")
        engine._init_sequential_buffer("ret_5m", "Ridge", metadata)
        buffer_mgr = engine._seq_buffers.get("ret_5m:Ridge")
        assert buffer_mgr is not None
        assert buffer_mgr.F == 5  # 5 OHLCV channels
        assert buffer_mgr.T == 32  # sequence_length

    def test_feature_list_empty_is_valid_for_raw(self, tmp_path):
        """feature_list=[] is valid for raw_sequence models (CONTRACT rule 6)."""
        from tests.test_live_inference_input_mode import _make_run_root
        from LIVE_TRADING.models.loader import ModelLoader

        metadata = {
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [],
            "n_features": 0,
            "input_mode": "raw_sequence",
            "sequence_length": 64,
            "sequence_channels": ["open", "high", "low", "close", "volume"],
            "sequence_normalization": "returns",
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        }
        run_root = _make_run_root(tmp_path, family="Ridge", metadata=metadata)
        loader = ModelLoader(run_root, verify_checksums=False)

        features = loader.get_feature_list("ret_5m", "Ridge")
        assert features == []
