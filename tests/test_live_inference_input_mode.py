"""
Unit tests for LIVE_TRADING input_mode awareness (Phases 0-2).

Tests the following changes:
- ModelLoader: get_input_mode(), get_sequence_config(), get_feature_list() for raw models
- InferenceEngine: buffer init for raw models, predict() routing
- MultiHorizonPredictor: predict_single_target(), _predict_single() branching
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from LIVE_TRADING.common.exceptions import InferenceError
from LIVE_TRADING.models.loader import ModelLoader
from LIVE_TRADING.models.inference import InferenceEngine


# ---------------------------------------------------------------------------
# Picklable dummy model (MagicMock is not picklable with torch mock)
# ---------------------------------------------------------------------------

class _DummyModel:
    """Picklable dummy model for tests."""
    def predict(self, X):
        return np.array([0.5])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_run_root(
    tmp_path: Path,
    target: str = "ret_5m",
    family: str = "Ridge",
    view: str = "CROSS_SECTIONAL",
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """
    Create a minimal run root with manifest and model_meta.json.

    Default family is Ridge (pickle-loadable without TF/torch).
    """
    run_root = tmp_path / "run_root"
    family_dir = run_root / "targets" / target / "models" / f"view={view}" / f"family={family}"
    family_dir.mkdir(parents=True)

    # Write manifest
    manifest = {"target_index": {target: {"status": "complete", "families_trained": [family]}}}
    (run_root / "manifest.json").write_text(json.dumps(manifest))

    # Write model_meta.json
    meta = metadata or {
        "family": family,
        "target": target,
        "feature_list": [f"feat_{i}" for i in range(10)],
        "n_features": 10,
        "metrics": {"mean_IC": 0.05},
        "model_checksum": None,
        "interval_minutes": 1.0,
    }
    (family_dir / "model_meta.json").write_text(json.dumps(meta))

    # Write a picklable dummy model
    (family_dir / "model.pkl").write_bytes(pickle.dumps(_DummyModel()))

    return run_root


# ---------------------------------------------------------------------------
# ModelLoader tests
# ---------------------------------------------------------------------------

class TestModelLoaderInputMode:

    def test_get_input_mode_default(self, tmp_path):
        """Models without input_mode field return 'features'."""
        run_root = _make_run_root(tmp_path, family="Ridge", metadata={
            "family": "Ridge", "target": "ret_5m",
            "feature_list": ["a", "b"], "n_features": 2,
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        })
        loader = ModelLoader(run_root, verify_checksums=False)
        assert loader.get_input_mode("ret_5m", "Ridge") == "features"

    def test_get_input_mode_raw_sequence(self, tmp_path):
        """Models with input_mode='raw_sequence' detected correctly."""
        run_root = _make_run_root(tmp_path, family="Ridge", metadata={
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "input_mode": "raw_sequence",
            "sequence_length": 64,
            "sequence_channels": ["open", "high", "low", "close", "volume"],
            "sequence_normalization": "returns",
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        })
        loader = ModelLoader(run_root, verify_checksums=False)
        assert loader.get_input_mode("ret_5m", "Ridge") == "raw_sequence"

    def test_get_feature_list_no_warning_for_raw(self, tmp_path, caplog):
        """get_feature_list() returns [] without warning for raw_sequence models."""
        run_root = _make_run_root(tmp_path, family="Ridge", metadata={
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "input_mode": "raw_sequence",
            "sequence_length": 64,
            "sequence_channels": ["open", "high", "low", "close", "volume"],
            "sequence_normalization": "returns",
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        })
        loader = ModelLoader(run_root, verify_checksums=False)

        with caplog.at_level(logging.WARNING):
            result = loader.get_feature_list("ret_5m", "Ridge")

        assert result == []
        assert "No feature list found" not in caplog.text

    def test_get_feature_list_warns_for_missing_features(self, tmp_path, caplog):
        """get_feature_list() warns when no features found for feature-based model."""
        run_root = _make_run_root(tmp_path, family="Ridge", metadata={
            "family": "Ridge", "target": "ret_5m",
            "n_features": 0,
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        })
        loader = ModelLoader(run_root, verify_checksums=False)

        with caplog.at_level(logging.WARNING):
            result = loader.get_feature_list("ret_5m", "Ridge")

        assert result == []
        assert "No feature list found" in caplog.text

    def test_get_sequence_config(self, tmp_path):
        """Sequence config extracted correctly from metadata."""
        run_root = _make_run_root(tmp_path, family="Ridge", metadata={
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "input_mode": "raw_sequence",
            "sequence_length": 128,
            "sequence_channels": ["open", "high", "low", "close", "volume"],
            "sequence_normalization": "log_returns",
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        })
        loader = ModelLoader(run_root, verify_checksums=False)
        config = loader.get_sequence_config("ret_5m", "Ridge")

        assert config["sequence_length"] == 128
        assert config["sequence_channels"] == ["open", "high", "low", "close", "volume"]
        assert config["sequence_normalization"] == "log_returns"

    def test_get_sequence_config_defaults(self, tmp_path):
        """Sequence config uses safe defaults for missing fields."""
        run_root = _make_run_root(tmp_path, family="Ridge", metadata={
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "input_mode": "raw_sequence",
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        })
        loader = ModelLoader(run_root, verify_checksums=False)
        config = loader.get_sequence_config("ret_5m", "Ridge")

        assert config["sequence_length"] == 64
        assert config["sequence_channels"] == ["open", "high", "low", "close", "volume"]
        assert config["sequence_normalization"] == "returns"

    def test_get_sequence_config_features_mode(self, tmp_path):
        """Sequence config returns empty dict for feature-based models."""
        run_root = _make_run_root(tmp_path, family="LightGBM", metadata={
            "family": "LightGBM", "target": "ret_5m",
            "feature_list": ["a", "b"], "n_features": 2,
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        })
        loader = ModelLoader(run_root, verify_checksums=False)
        config = loader.get_sequence_config("ret_5m", "LightGBM")
        assert config == {}


# ---------------------------------------------------------------------------
# InferenceEngine tests
# ---------------------------------------------------------------------------

class TestInferenceEngineInputMode:

    def _make_engine_with_model(
        self,
        tmp_path: Path,
        metadata: Dict[str, Any],
        family: str = "Ridge",
    ) -> tuple[InferenceEngine, ModelLoader]:
        """Helper to create an InferenceEngine with a model loaded."""
        run_root = _make_run_root(tmp_path, family=family, metadata=metadata)
        loader = ModelLoader(run_root, verify_checksums=False)
        engine = InferenceEngine(loader, device="cpu")
        return engine, loader

    def test_buffer_init_raw_sequence(self, tmp_path):
        """Buffer initialized with F=5 for raw_sequence models."""
        metadata = {
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "input_mode": "raw_sequence",
            "sequence_length": 32,
            "sequence_channels": ["open", "high", "low", "close", "volume"],
            "sequence_normalization": "returns",
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        }
        engine, loader = self._make_engine_with_model(tmp_path, metadata)
        engine._init_sequential_buffer("ret_5m", "Ridge", metadata)

        buffer_mgr = engine._seq_buffers.get("ret_5m:Ridge")
        assert buffer_mgr is not None
        assert buffer_mgr.F == 5
        assert buffer_mgr.T == 32

    def test_buffer_init_features(self, tmp_path):
        """Buffer initialized with F=len(feature_list) for feature models."""
        feature_list = [f"feat_{i}" for i in range(100)]
        metadata = {
            "family": "Ridge", "target": "ret_5m",
            "feature_list": feature_list, "n_features": 100,
            "sequence_length": 20,
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        }
        engine, loader = self._make_engine_with_model(tmp_path, metadata)
        engine._init_sequential_buffer("ret_5m", "Ridge", metadata)

        buffer_mgr = engine._seq_buffers.get("ret_5m:Ridge")
        assert buffer_mgr is not None
        assert buffer_mgr.F == 100
        assert buffer_mgr.T == 20

    def test_buffer_init_skips_zero_features(self, tmp_path):
        """Buffer not initialized when no features or channels available."""
        metadata = {
            "family": "Ridge", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        }
        engine, loader = self._make_engine_with_model(tmp_path, metadata)
        engine._init_sequential_buffer("ret_5m", "Ridge", metadata)

        assert "ret_5m:Ridge" not in engine._seq_buffers

    def test_raw_sequence_non_sequential_family_raises(self, tmp_path):
        """raw_sequence mode with tree family raises InferenceError."""
        metadata = {
            "family": "LightGBM", "target": "ret_5m",
            "feature_list": [], "n_features": 0,
            "input_mode": "raw_sequence",
            "sequence_length": 32,
            "sequence_channels": ["open", "high", "low", "close", "volume"],
            "sequence_normalization": "returns",
            "metrics": {}, "model_checksum": None, "interval_minutes": 1.0,
        }
        run_root = _make_run_root(tmp_path, family="LightGBM", metadata=metadata)
        loader = ModelLoader(run_root, verify_checksums=False)
        engine = InferenceEngine(loader, device="cpu")

        features = np.random.randn(5).astype(np.float32)
        with pytest.raises(InferenceError, match="raw_sequence mode only supported"):
            engine.predict("ret_5m", "LightGBM", features, "SPY")


# ---------------------------------------------------------------------------
# MultiHorizonPredictor tests
# ---------------------------------------------------------------------------

class TestMultiHorizonPredictor:

    def test_predict_single_target_exists(self, tmp_path):
        """predict_single_target() method exists and callable."""
        from LIVE_TRADING.prediction.predictor import MultiHorizonPredictor

        run_root = _make_run_root(tmp_path)
        predictor = MultiHorizonPredictor(run_root=str(run_root))

        # Method exists
        assert hasattr(predictor, "predict_single_target")
        assert callable(predictor.predict_single_target)

    def test_predict_single_target_returns_none_no_families(self, tmp_path):
        """predict_single_target returns None when no families available."""
        import pandas as pd
        from LIVE_TRADING.prediction.predictor import MultiHorizonPredictor

        # Create run root with no models for "will_peak_5m"
        run_root = _make_run_root(tmp_path, target="ret_5m")
        predictor = MultiHorizonPredictor(run_root=str(run_root))

        prices = pd.DataFrame({
            "Open": [100.0] * 5,
            "High": [101.0] * 5,
            "Low": [99.0] * 5,
            "Close": [100.5] * 5,
            "Volume": [1000.0] * 5,
        })

        result = predictor.predict_single_target(
            target="will_peak_5m",  # Not in run
            prices=prices,
            symbol="SPY",
        )
        assert result is None

    def test_model_prediction_has_alpha(self):
        """ModelPrediction has .alpha property (used by barrier gate)."""
        from LIVE_TRADING.prediction.predictor import ModelPrediction
        from LIVE_TRADING.prediction.confidence import ConfidenceComponents

        pred = ModelPrediction(
            family="Ridge",
            horizon="5m",
            raw=0.01,
            standardized=0.5,
            confidence=ConfidenceComponents(
                ic=0.8, freshness=0.9, capacity=1.0, stability=0.85, overall=0.72,
            ),
            calibrated=0.36,
        )
        # .alpha is alias for .calibrated
        assert pred.alpha == 0.36
        assert hasattr(pred, "alpha")
