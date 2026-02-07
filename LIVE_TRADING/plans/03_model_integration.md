# Plan 03: Model Integration

## Overview

Model integration layer that loads trained models from TRAINING artifacts and provides unified inference. Handles routing between different model families (LightGBM, Keras, etc.) and manages sequential model buffers.

## Files to Create

### 1. `LIVE_TRADING/models/__init__.py`

```python
from .loader import ModelLoader, load_model_from_run
from .inference import InferenceEngine, predict

__all__ = ["ModelLoader", "load_model_from_run", "InferenceEngine", "predict"]
```

### 2. `LIVE_TRADING/models/loader.py`
**Purpose:** Load models from TRAINING artifact directories

**Integration:** Uses `TRAINING/orchestration/utils/target_first_paths.py` for SST-compliant paths

```python
"""
Model Loader
============

Loads trained models from TRAINING run artifacts.
Uses SST path helpers for artifact location.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import iterdir_sorted
from TRAINING.models.registry import FAMILY_CAPABILITIES, get_trainer_info

from LIVE_TRADING.common.constants import (
    FAMILIES,
    SEQUENTIAL_FAMILIES,
    TF_FAMILIES,
    TREE_FAMILIES,
)
from LIVE_TRADING.common.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads models from a TRAINING run's artifact directory.

    Uses SST-compliant paths to locate model files.
    """

    def __init__(self, run_root: Path | str):
        """
        Initialize model loader.

        Args:
            run_root: Path to run directory (e.g., RESULTS/runs/<run_id>/<timestamp>)
        """
        self.run_root = Path(run_root)

        if not self.run_root.exists():
            raise ModelLoadError("N/A", "N/A", f"Run root does not exist: {run_root}")

        # Cache loaded models
        self._model_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

        # Target index from manifest
        self._target_index = self._load_target_index()

        logger.info(f"ModelLoader initialized from {run_root}")

    def _load_target_index(self) -> Dict[str, Any]:
        """Load target index from run manifest."""
        manifest_path = self.run_root / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
                return manifest.get("target_index", {})
        return {}

    def get_target_models_dir(
        self,
        target: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Path:
        """
        Get the models directory for a target.

        Args:
            target: Target name
            view: View type (CROSS_SECTIONAL or SEQUENTIAL)

        Returns:
            Path to models directory
        """
        # SST path structure: targets/<target>/models/view=<view>/
        return self.run_root / "targets" / target / "models" / f"view={view}"

    def get_family_dir(
        self,
        target: str,
        family: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Path:
        """
        Get the directory for a specific family's model.

        Args:
            target: Target name
            family: Model family name
            view: View type

        Returns:
            Path to family model directory
        """
        models_dir = self.get_target_models_dir(target, view)
        return models_dir / f"family={family}"

    def load_model(
        self,
        target: str,
        family: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model and its metadata.

        Args:
            target: Target name
            family: Model family name
            view: View type

        Returns:
            Tuple of (model, metadata_dict)

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        cache_key = f"{target}:{family}:{view}"

        if cache_key in self._model_cache:
            logger.debug(f"Using cached model: {cache_key}")
            return self._model_cache[cache_key]

        family_dir = self.get_family_dir(target, family, view)

        if not family_dir.exists():
            raise ModelLoadError(family, target, f"Family directory not found: {family_dir}")

        # Load metadata first
        meta_path = family_dir / "model_meta.json"
        if not meta_path.exists():
            raise ModelLoadError(family, target, f"Model metadata not found: {meta_path}")

        with open(meta_path) as f:
            metadata = json.load(f)

        # Load model based on family type
        model = self._load_model_by_family(family, family_dir, metadata)

        # Cache and return
        self._model_cache[cache_key] = (model, metadata)
        logger.info(f"Loaded model: {cache_key}")

        return model, metadata

    def _load_model_by_family(
        self,
        family: str,
        family_dir: Path,
        metadata: Dict[str, Any],
    ) -> Any:
        """Load model using family-specific loader."""

        if family in TREE_FAMILIES:
            return self._load_tree_model(family_dir)
        elif family in TF_FAMILIES:
            return self._load_keras_model(family_dir, metadata)
        else:
            # Fallback to pickle
            return self._load_pickle_model(family_dir)

    def _load_tree_model(self, family_dir: Path) -> Any:
        """Load tree-based model (LightGBM, XGBoost)."""
        model_path = family_dir / "model.pkl"

        if not model_path.exists():
            # Try joblib format
            model_path = family_dir / "model.joblib"

        if not model_path.exists():
            raise ModelLoadError("tree", "N/A", f"No model file found in {family_dir}")

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _load_keras_model(
        self,
        family_dir: Path,
        metadata: Dict[str, Any],
    ) -> Any:
        """Load Keras/TensorFlow model."""
        import tensorflow as tf

        model_path = family_dir / "model.h5"

        if not model_path.exists():
            # Try SavedModel format
            model_path = family_dir / "model"

        if not model_path.exists():
            raise ModelLoadError("keras", "N/A", f"No model file found in {family_dir}")

        model = tf.keras.models.load_model(str(model_path), compile=False)
        return model

    def _load_pickle_model(self, family_dir: Path) -> Any:
        """Load generic pickle model."""
        model_path = family_dir / "model.pkl"

        if not model_path.exists():
            raise ModelLoadError("pickle", "N/A", f"No model file found in {family_dir}")

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def get_feature_list(
        self,
        target: str,
        family: str,
        view: str = "CROSS_SECTIONAL",
    ) -> List[str]:
        """
        Get the feature list for a model.

        Args:
            target: Target name
            family: Model family name
            view: View type

        Returns:
            List of feature names in order
        """
        _, metadata = self.load_model(target, family, view)
        return metadata.get("feature_list", [])

    def get_routing_decision(
        self,
        target: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Dict[str, Any]:
        """
        Get the routing decision for a target.

        Args:
            target: Target name
            view: View type

        Returns:
            Routing decision dict with selected family, metrics, etc.
        """
        models_dir = self.get_target_models_dir(target, view)
        routing_path = models_dir / "routing_decision.json"

        if not routing_path.exists():
            return {}

        with open(routing_path) as f:
            return json.load(f)

    def list_available_targets(self) -> List[str]:
        """List all targets in the run."""
        targets_dir = self.run_root / "targets"
        if not targets_dir.exists():
            return []

        return sorted([
            d.name for d in iterdir_sorted(targets_dir)
            if d.is_dir()
        ])

    def list_available_families(
        self,
        target: str,
        view: str = "CROSS_SECTIONAL",
    ) -> List[str]:
        """List available model families for a target."""
        models_dir = self.get_target_models_dir(target, view)
        if not models_dir.exists():
            return []

        families = []
        for d in iterdir_sorted(models_dir):
            if d.is_dir() and d.name.startswith("family="):
                family_name = d.name.replace("family=", "")
                families.append(family_name)

        return families

    def get_model_metrics(
        self,
        target: str,
        family: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Dict[str, float]:
        """Get model metrics (AUC, etc.) from metadata."""
        _, metadata = self.load_model(target, family, view)
        return metadata.get("metrics", {})


def load_model_from_run(
    run_root: Path | str,
    target: str,
    family: str,
    view: str = "CROSS_SECTIONAL",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Convenience function to load a single model.

    Args:
        run_root: Path to run directory
        target: Target name
        family: Model family
        view: View type

    Returns:
        Tuple of (model, metadata)
    """
    loader = ModelLoader(run_root)
    return loader.load_model(target, family, view)
```

### 3. `LIVE_TRADING/models/inference.py`
**Purpose:** Unified inference routing for different model families

```python
"""
Inference Engine
================

Unified inference interface that routes to family-specific prediction methods.
Handles sequential models with buffer management.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from TRAINING.common.live.seq_ring_buffer import SeqBufferManager, LiveSeqInference
from TRAINING.models.registry import FAMILY_CAPABILITIES

from LIVE_TRADING.common.constants import (
    SEQUENTIAL_FAMILIES,
    TF_FAMILIES,
    TREE_FAMILIES,
)
from LIVE_TRADING.common.exceptions import InferenceError
from .loader import ModelLoader

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Unified inference engine for all model families.

    Routes predictions to appropriate family-specific methods.
    Manages sequential buffers for time-series models.
    """

    def __init__(
        self,
        loader: ModelLoader,
        device: str = "cpu",
    ):
        """
        Initialize inference engine.

        Args:
            loader: ModelLoader instance
            device: Device for inference ("cpu" or "cuda")
        """
        self.loader = loader
        self.device = device

        # Loaded models cache
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # Sequential buffer managers per target
        self._seq_buffers: Dict[str, SeqBufferManager] = {}

        logger.info(f"InferenceEngine initialized on device: {device}")

    def load_models_for_target(
        self,
        target: str,
        families: List[str] | None = None,
        view: str = "CROSS_SECTIONAL",
    ) -> None:
        """
        Pre-load models for a target.

        Args:
            target: Target name
            families: List of families to load (None = all available)
            view: View type
        """
        if families is None:
            families = self.loader.list_available_families(target, view)

        for family in families:
            try:
                model, metadata = self.loader.load_model(target, family, view)
                cache_key = f"{target}:{family}:{view}"
                self._models[cache_key] = model
                self._metadata[cache_key] = metadata

                # Initialize buffer for sequential models
                if family in SEQUENTIAL_FAMILIES:
                    self._init_sequential_buffer(target, family, metadata)

                logger.info(f"Loaded {family} model for {target}")

            except Exception as e:
                logger.warning(f"Failed to load {family} for {target}: {e}")

    def _init_sequential_buffer(
        self,
        target: str,
        family: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Initialize sequential buffer for a model."""
        seq_length = metadata.get("sequence_length", 20)
        n_features = len(metadata.get("feature_list", []))

        buffer_key = f"{target}:{family}"
        self._seq_buffers[buffer_key] = SeqBufferManager(
            T=seq_length,
            F=n_features,
            ttl_seconds=300.0,
        )
        logger.debug(f"Initialized buffer for {buffer_key}: T={seq_length}, F={n_features}")

    def predict(
        self,
        target: str,
        family: str,
        features: np.ndarray,
        symbol: str = "default",
        view: str = "CROSS_SECTIONAL",
    ) -> float:
        """
        Make prediction for a single sample.

        Args:
            target: Target name
            family: Model family
            features: Feature array (1D for cross-sectional, 2D for sequential)
            symbol: Symbol for sequential buffer tracking
            view: View type

        Returns:
            Prediction value

        Raises:
            InferenceError: If prediction fails
        """
        cache_key = f"{target}:{family}:{view}"

        if cache_key not in self._models:
            # Try to load on-demand
            model, metadata = self.loader.load_model(target, family, view)
            self._models[cache_key] = model
            self._metadata[cache_key] = metadata

            if family in SEQUENTIAL_FAMILIES:
                self._init_sequential_buffer(target, family, metadata)

        model = self._models[cache_key]
        metadata = self._metadata[cache_key]

        try:
            if family in TREE_FAMILIES:
                return self._predict_tree(model, features)
            elif family in SEQUENTIAL_FAMILIES:
                return self._predict_sequential(
                    model, features, target, family, symbol
                )
            elif family in TF_FAMILIES:
                return self._predict_keras(model, features)
            else:
                return self._predict_generic(model, features)

        except Exception as e:
            raise InferenceError(family, symbol, str(e))

    def _predict_tree(self, model: Any, features: np.ndarray) -> float:
        """Predict with tree-based model (LightGBM, XGBoost)."""
        X = np.atleast_2d(features)
        pred = model.predict(X)
        return float(pred[0])

    def _predict_keras(self, model: Any, features: np.ndarray) -> float:
        """Predict with Keras model (non-sequential)."""
        X = np.atleast_2d(features).astype(np.float32)
        pred = model.predict(X, verbose=0)
        return float(pred.squeeze())

    def _predict_sequential(
        self,
        model: Any,
        features: np.ndarray,
        target: str,
        family: str,
        symbol: str,
    ) -> float:
        """Predict with sequential model using buffer."""
        buffer_key = f"{target}:{family}"
        buffer_manager = self._seq_buffers.get(buffer_key)

        if buffer_manager is None:
            raise InferenceError(family, symbol, "Buffer not initialized")

        # Push features to buffer
        features_1d = np.atleast_1d(features).astype(np.float32)
        buffer_manager.push_features(symbol, features_1d)

        # Check if buffer is ready
        if not buffer_manager.is_ready(symbol):
            # Return NaN while warming up
            return float("nan")

        # Get sequence and predict
        sequence = buffer_manager.get_sequence(symbol)
        if sequence is None:
            return float("nan")

        import torch
        with torch.no_grad():
            pred = model(sequence.to(self.device))
            return float(pred.cpu().numpy().squeeze())

    def _predict_generic(self, model: Any, features: np.ndarray) -> float:
        """Generic predict for unknown model types."""
        X = np.atleast_2d(features)
        if hasattr(model, "predict"):
            pred = model.predict(X)
            return float(np.atleast_1d(pred)[0])
        else:
            raise InferenceError("generic", "N/A", "Model has no predict method")

    def predict_all_families(
        self,
        target: str,
        features: np.ndarray,
        symbol: str = "default",
        view: str = "CROSS_SECTIONAL",
        families: List[str] | None = None,
    ) -> Dict[str, float]:
        """
        Make predictions from all loaded families for a target.

        Args:
            target: Target name
            features: Feature array
            symbol: Symbol for tracking
            view: View type
            families: Specific families (None = all loaded)

        Returns:
            Dict mapping family name to prediction
        """
        if families is None:
            families = self.loader.list_available_families(target, view)

        results = {}
        for family in families:
            try:
                pred = self.predict(target, family, features, symbol, view)
                if not np.isnan(pred):
                    results[family] = pred
            except InferenceError as e:
                logger.warning(f"Inference failed for {family}: {e}")

        return results

    def warmup_sequential(
        self,
        target: str,
        family: str,
        historical_features: np.ndarray,
        symbol: str = "default",
    ) -> int:
        """
        Warm up sequential buffer with historical data.

        Args:
            target: Target name
            family: Model family
            historical_features: Historical feature array (T x F)
            symbol: Symbol for buffer

        Returns:
            Number of samples pushed
        """
        buffer_key = f"{target}:{family}"
        buffer_manager = self._seq_buffers.get(buffer_key)

        if buffer_manager is None:
            raise InferenceError(family, symbol, "Buffer not initialized")

        count = 0
        for i in range(len(historical_features)):
            buffer_manager.push_features(symbol, historical_features[i])
            count += 1

        logger.info(f"Warmed up {buffer_key} with {count} samples")
        return count

    def reset_buffers(self, target: str | None = None) -> None:
        """Reset sequential buffers."""
        if target is None:
            for manager in self._seq_buffers.values():
                manager.reset_all()
        else:
            for key, manager in self._seq_buffers.items():
                if key.startswith(f"{target}:"):
                    manager.reset_all()


def predict(
    loader: ModelLoader,
    target: str,
    family: str,
    features: np.ndarray,
    view: str = "CROSS_SECTIONAL",
) -> float:
    """
    Convenience function for single prediction.

    Args:
        loader: ModelLoader instance
        target: Target name
        family: Model family
        features: Feature array
        view: View type

    Returns:
        Prediction value
    """
    engine = InferenceEngine(loader)
    return engine.predict(target, family, features, view=view)
```

### 4. `LIVE_TRADING/models/feature_builder.py`
**Purpose:** Build features from market data for inference

```python
"""
Feature Builder
===============

Builds features from market data matching the training feature set.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from CONFIG.config_loader import get_cfg

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Builds features from market data for model inference.

    Must produce features in the same order as training.
    """

    def __init__(self, feature_list: List[str]):
        """
        Initialize feature builder.

        Args:
            feature_list: Ordered list of feature names (from model metadata)
        """
        self.feature_list = feature_list
        self._feature_funcs = self._register_feature_funcs()

    def _register_feature_funcs(self) -> Dict[str, callable]:
        """Register feature computation functions."""
        return {
            "ret_1d": self._calc_ret_1d,
            "ret_5d": self._calc_ret_5d,
            "ret_10d": self._calc_ret_10d,
            "ret_20d": self._calc_ret_20d,
            "vol_10d": self._calc_vol_10d,
            "vol_20d": self._calc_vol_20d,
            "rsi_14": self._calc_rsi,
            "ma_ratio_20": self._calc_ma_ratio_20,
            "bb_position": self._calc_bb_position,
        }

    def build_features(
        self,
        prices: pd.DataFrame,
        symbol: str | None = None,
    ) -> np.ndarray:
        """
        Build feature array from price data.

        Args:
            prices: DataFrame with OHLCV columns
            symbol: Optional symbol for logging

        Returns:
            Feature array matching feature_list order
        """
        features = []

        for feat_name in self.feature_list:
            if feat_name in self._feature_funcs:
                value = self._feature_funcs[feat_name](prices)
            else:
                # Unknown feature - use NaN
                logger.warning(f"Unknown feature: {feat_name}")
                value = np.nan

            features.append(value)

        return np.array(features, dtype=np.float32)

    def _calc_ret_1d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change().iloc[-1])

    def _calc_ret_5d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change(5).iloc[-1])

    def _calc_ret_10d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change(10).iloc[-1])

    def _calc_ret_20d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change(20).iloc[-1])

    def _calc_vol_10d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change().rolling(10).std().iloc[-1])

    def _calc_vol_20d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change().rolling(20).std().iloc[-1])

    def _calc_rsi(self, prices: pd.DataFrame, period: int = 14) -> float:
        close = prices["Close"]
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def _calc_ma_ratio_20(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        ma = close.rolling(20).mean()
        return float((close.iloc[-1] / ma.iloc[-1]) - 1)

    def _calc_bb_position(self, prices: pd.DataFrame, period: int = 20) -> float:
        close = prices["Close"]
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        position = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        return float(position)


def build_features_from_prices(
    prices: pd.DataFrame,
    feature_list: List[str],
) -> np.ndarray:
    """
    Convenience function to build features.

    Args:
        prices: OHLCV DataFrame
        feature_list: Ordered feature names

    Returns:
        Feature array
    """
    builder = FeatureBuilder(feature_list)
    return builder.build_features(prices)
```

## Tests

### `LIVE_TRADING/tests/test_model_loader.py`

```python
"""Tests for model loader."""

import pytest
from pathlib import Path
import json
import pickle
import tempfile
import numpy as np

from LIVE_TRADING.models.loader import ModelLoader, load_model_from_run
from LIVE_TRADING.common.exceptions import ModelLoadError


@pytest.fixture
def mock_run_dir():
    """Create a mock run directory with model artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)

        # Create directory structure
        target_dir = run_root / "targets" / "test_target" / "models" / "view=CROSS_SECTIONAL" / "family=LightGBM"
        target_dir.mkdir(parents=True)

        # Create mock model
        class MockModel:
            def predict(self, X):
                return np.array([0.5])

        with open(target_dir / "model.pkl", "wb") as f:
            pickle.dump(MockModel(), f)

        # Create metadata
        metadata = {
            "feature_list": ["ret_1d", "vol_10d"],
            "metrics": {"auc": 0.65},
        }
        with open(target_dir / "model_meta.json", "w") as f:
            json.dump(metadata, f)

        yield run_root


class TestModelLoader:
    def test_init_invalid_path(self):
        with pytest.raises(ModelLoadError):
            ModelLoader("/nonexistent/path")

    def test_load_model(self, mock_run_dir):
        loader = ModelLoader(mock_run_dir)
        model, metadata = loader.load_model("test_target", "LightGBM")

        assert hasattr(model, "predict")
        assert metadata["feature_list"] == ["ret_1d", "vol_10d"]

    def test_get_feature_list(self, mock_run_dir):
        loader = ModelLoader(mock_run_dir)
        features = loader.get_feature_list("test_target", "LightGBM")

        assert features == ["ret_1d", "vol_10d"]

    def test_list_available_targets(self, mock_run_dir):
        loader = ModelLoader(mock_run_dir)
        targets = loader.list_available_targets()

        assert "test_target" in targets

    def test_list_available_families(self, mock_run_dir):
        loader = ModelLoader(mock_run_dir)
        families = loader.list_available_families("test_target")

        assert "LightGBM" in families

    def test_model_caching(self, mock_run_dir):
        loader = ModelLoader(mock_run_dir)

        model1, _ = loader.load_model("test_target", "LightGBM")
        model2, _ = loader.load_model("test_target", "LightGBM")

        assert model1 is model2  # Same cached instance

    def test_convenience_function(self, mock_run_dir):
        model, metadata = load_model_from_run(
            mock_run_dir, "test_target", "LightGBM"
        )
        assert hasattr(model, "predict")
```

### `LIVE_TRADING/tests/test_inference.py`

```python
"""Tests for inference engine."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from LIVE_TRADING.models.inference import InferenceEngine, predict
from LIVE_TRADING.common.exceptions import InferenceError


@pytest.fixture
def mock_loader():
    """Create a mock model loader."""
    loader = Mock()

    class MockModel:
        def predict(self, X):
            return np.array([0.5 * X.mean()])

    loader.load_model.return_value = (MockModel(), {"feature_list": ["f1", "f2"]})
    loader.list_available_families.return_value = ["LightGBM"]

    return loader


class TestInferenceEngine:
    def test_predict_tree_model(self, mock_loader):
        engine = InferenceEngine(mock_loader)

        features = np.array([1.0, 2.0])
        result = engine.predict("target", "LightGBM", features)

        assert isinstance(result, float)

    def test_predict_all_families(self, mock_loader):
        engine = InferenceEngine(mock_loader)

        features = np.array([1.0, 2.0])
        results = engine.predict_all_families("target", features)

        assert "LightGBM" in results
        assert isinstance(results["LightGBM"], float)

    def test_inference_error_handling(self, mock_loader):
        mock_loader.load_model.side_effect = Exception("Load failed")
        engine = InferenceEngine(mock_loader)

        with pytest.raises(InferenceError):
            engine.predict("target", "LightGBM", np.array([1.0]))
```

## SST Compliance Checklist

- [ ] Uses `iterdir_sorted()` for filesystem enumeration
- [ ] Uses SST path structure: `targets/<target>/models/view=<view>/family=<family>/`
- [ ] Imports from `TRAINING.models.registry` for family capabilities
- [ ] Imports from `TRAINING.common.live.seq_ring_buffer` for sequential models
- [ ] Model caching to avoid repeated loads
- [ ] Proper exception handling with `ModelLoadError`, `InferenceError`

## Dependencies

- `TRAINING.models.registry.FAMILY_CAPABILITIES`
- `TRAINING.common.live.seq_ring_buffer.SeqBufferManager`
- `TRAINING.common.utils.determinism_ordering.iterdir_sorted`
- `CONFIG.config_loader.get_cfg`

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `__init__.py` | 10 |
| `loader.py` | 300 |
| `inference.py` | 280 |
| `feature_builder.py` | 150 |
| `tests/test_model_loader.py` | 100 |
| `tests/test_inference.py` | 80 |
| **Total** | ~920 |
