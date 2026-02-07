# Plan 04: Prediction Pipeline

## Overview

Multi-horizon prediction pipeline that generates standardized predictions across all horizons and model families. Includes Z-score standardization and confidence scoring.

## Mathematical Foundation

### Z-Score Standardization
```
s_{m,h} = clip((r̂_{m,h} - μ_{m,h}) / σ_{m,h}, -3, 3)
```
- `r̂_{m,h}` = raw prediction from model m, horizon h
- `μ_{m,h}` = rolling mean of predictions
- `σ_{m,h}` = rolling standard deviation
- Clipping prevents extreme outliers

### Confidence Calculation
```
c_{m,h} = IC_{m,h} × freshness × capacity × stability
```
- `IC` = Information Coefficient (Spearman correlation)
- `freshness = e^{-Δt/τ_h}` decays with data age
- `capacity = min(1, κ × ADV / planned_dollars)`
- `stability = 1 / rolling_RMSE`

## Files to Create

### 1. `LIVE_TRADING/prediction/__init__.py`

```python
from .predictor import MultiHorizonPredictor, HorizonPredictions
from .standardization import ZScoreStandardizer
from .confidence import ConfidenceScorer

__all__ = [
    "MultiHorizonPredictor",
    "HorizonPredictions",
    "ZScoreStandardizer",
    "ConfidenceScorer",
]
```

### 2. `LIVE_TRADING/prediction/standardization.py`
**Purpose:** Z-score standardization for model predictions

```python
"""
Z-Score Standardization
=======================

Standardizes model predictions to comparable scales using rolling z-scores.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.constants import (
    DEFAULT_CONFIG,
    ZSCORE_CLIP_MIN,
    ZSCORE_CLIP_MAX,
    STANDARDIZATION_WINDOW,
)

logger = logging.getLogger(__name__)


@dataclass
class StandardizationStats:
    """Statistics for standardization."""
    mean: float
    std: float
    count: int
    last_raw: float
    last_standardized: float


class ZScoreStandardizer:
    """
    Rolling Z-score standardizer for model predictions.

    Maintains separate statistics for each (model, horizon) pair.
    """

    def __init__(
        self,
        window_size: int | None = None,
        clip_min: float = ZSCORE_CLIP_MIN,
        clip_max: float = ZSCORE_CLIP_MAX,
    ):
        """
        Initialize standardizer.

        Args:
            window_size: Rolling window size in observations
            clip_min: Minimum clipped z-score
            clip_max: Maximum clipped z-score
        """
        self.window_size = window_size or get_cfg(
            "live_trading.standardization.window_size",
            default=STANDARDIZATION_WINDOW,
        )
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Rolling buffers per (model, horizon) key
        self._buffers: Dict[str, deque] = {}

        logger.info(f"ZScoreStandardizer initialized: window={self.window_size}")

    def _get_key(self, model: str, horizon: str) -> str:
        """Get buffer key for model/horizon pair."""
        return f"{model}:{horizon}"

    def standardize(
        self,
        raw_prediction: float,
        model: str,
        horizon: str,
    ) -> float:
        """
        Standardize a prediction using rolling z-score.

        Args:
            raw_prediction: Raw model prediction
            model: Model family name
            horizon: Horizon (e.g., "5m")

        Returns:
            Standardized prediction clipped to [-3, 3]
        """
        key = self._get_key(model, horizon)

        # Initialize buffer if needed
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.window_size)

        buffer = self._buffers[key]

        # Add to buffer
        buffer.append(raw_prediction)

        # Need minimum samples for meaningful standardization
        if len(buffer) < 3:
            # Return raw prediction during warmup
            return np.clip(raw_prediction, self.clip_min, self.clip_max)

        # Calculate rolling statistics
        arr = np.array(buffer)
        mean = np.mean(arr)
        std = np.std(arr)

        # Avoid division by zero
        if std < 1e-9:
            return 0.0

        # Z-score
        z = (raw_prediction - mean) / std

        # Clip
        z_clipped = np.clip(z, self.clip_min, self.clip_max)

        return float(z_clipped)

    def standardize_batch(
        self,
        predictions: Dict[str, float],
        horizon: str,
    ) -> Dict[str, float]:
        """
        Standardize predictions from multiple models.

        Args:
            predictions: Dict mapping model name to raw prediction
            horizon: Horizon for all predictions

        Returns:
            Dict mapping model name to standardized prediction
        """
        return {
            model: self.standardize(pred, model, horizon)
            for model, pred in predictions.items()
        }

    def get_stats(self, model: str, horizon: str) -> Optional[StandardizationStats]:
        """
        Get current statistics for a model/horizon.

        Args:
            model: Model family name
            horizon: Horizon

        Returns:
            Statistics or None if not enough data
        """
        key = self._get_key(model, horizon)
        if key not in self._buffers or len(self._buffers[key]) < 2:
            return None

        buffer = self._buffers[key]
        arr = np.array(buffer)

        return StandardizationStats(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            count=len(buffer),
            last_raw=float(arr[-1]),
            last_standardized=self.standardize(arr[-1], model, horizon),
        )

    def reset(self, model: str | None = None, horizon: str | None = None) -> None:
        """Reset buffers."""
        if model is None and horizon is None:
            self._buffers.clear()
        elif model is not None and horizon is not None:
            key = self._get_key(model, horizon)
            if key in self._buffers:
                self._buffers[key].clear()
```

### 3. `LIVE_TRADING/prediction/confidence.py`
**Purpose:** Confidence scoring based on IC, freshness, and capacity

```python
"""
Confidence Scoring
==================

Calculates confidence scores for model predictions based on
IC, freshness, capacity, and stability.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from scipy import stats

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.constants import (
    DEFAULT_CONFIG,
    FRESHNESS_TAU,
    CAPACITY_KAPPA,
    MIN_IC_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceComponents:
    """Components of confidence score."""
    ic: float
    freshness: float
    capacity: float
    stability: float
    overall: float


class ConfidenceScorer:
    """
    Calculates confidence scores for model predictions.

    Confidence = IC × freshness × capacity × stability
    """

    def __init__(
        self,
        ic_window: int = 20,
        stability_window: int = 10,
    ):
        """
        Initialize confidence scorer.

        Args:
            ic_window: Window for IC calculation
            stability_window: Window for stability calculation
        """
        self.ic_window = ic_window
        self.stability_window = stability_window

        # Rolling buffers for IC calculation: (prediction, actual_return)
        self._prediction_buffers: Dict[str, deque] = {}
        self._return_buffers: Dict[str, deque] = {}

        # Calibration RMSE buffers
        self._rmse_buffers: Dict[str, deque] = {}

        logger.info(f"ConfidenceScorer initialized: ic_window={ic_window}")

    def _get_key(self, model: str, horizon: str) -> str:
        return f"{model}:{horizon}"

    def update_with_actual(
        self,
        model: str,
        horizon: str,
        prediction: float,
        actual_return: float,
    ) -> None:
        """
        Update with actual return for IC calculation.

        Args:
            model: Model family name
            horizon: Horizon
            prediction: Previous prediction
            actual_return: Actual realized return
        """
        key = self._get_key(model, horizon)

        # Initialize buffers
        if key not in self._prediction_buffers:
            self._prediction_buffers[key] = deque(maxlen=self.ic_window)
            self._return_buffers[key] = deque(maxlen=self.ic_window)
            self._rmse_buffers[key] = deque(maxlen=self.stability_window)

        # Add to buffers
        self._prediction_buffers[key].append(prediction)
        self._return_buffers[key].append(actual_return)

        # Calculate squared error for stability
        error_sq = (prediction - actual_return) ** 2
        self._rmse_buffers[key].append(error_sq)

    def calculate_ic(self, model: str, horizon: str) -> float:
        """
        Calculate Information Coefficient (Spearman correlation).

        Args:
            model: Model family name
            horizon: Horizon

        Returns:
            IC value (0 if insufficient data)
        """
        key = self._get_key(model, horizon)

        if key not in self._prediction_buffers:
            return 0.0

        preds = np.array(self._prediction_buffers[key])
        rets = np.array(self._return_buffers[key])

        if len(preds) < 5:
            return 0.0

        # Spearman correlation
        try:
            corr, _ = stats.spearmanr(preds, rets)
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def calculate_freshness(
        self,
        horizon: str,
        data_timestamp: datetime,
        current_time: datetime | None = None,
    ) -> float:
        """
        Calculate freshness factor based on data age.

        freshness = e^{-Δt/τ_h}

        Args:
            horizon: Horizon
            data_timestamp: Timestamp of data
            current_time: Current time (default: now)

        Returns:
            Freshness factor in [0, 1]
        """
        if current_time is None:
            current_time = datetime.now()

        # Time difference in seconds
        delta_t = (current_time - data_timestamp).total_seconds()

        # Get tau for horizon
        tau = FRESHNESS_TAU.get(horizon, 300.0)

        # Exponential decay
        freshness = math.exp(-delta_t / tau)

        return max(0.0, min(1.0, freshness))

    def calculate_capacity(
        self,
        adv: float,
        planned_dollars: float,
        kappa: float = CAPACITY_KAPPA,
    ) -> float:
        """
        Calculate capacity factor.

        capacity = min(1, κ × ADV / planned_dollars)

        Args:
            adv: Average daily volume in dollars
            planned_dollars: Planned trade size in dollars
            kappa: Participation rate

        Returns:
            Capacity factor in [0, 1]
        """
        if planned_dollars <= 0:
            return 1.0

        capacity = kappa * adv / planned_dollars
        return min(1.0, capacity)

    def calculate_stability(self, model: str, horizon: str) -> float:
        """
        Calculate stability factor based on calibration RMSE.

        stability = 1 / (1 + rolling_RMSE)

        Args:
            model: Model family name
            horizon: Horizon

        Returns:
            Stability factor in [0, 1]
        """
        key = self._get_key(model, horizon)

        if key not in self._rmse_buffers or len(self._rmse_buffers[key]) < 3:
            return 0.5  # Neutral during warmup

        rmse = math.sqrt(np.mean(self._rmse_buffers[key]))
        stability = 1.0 / (1.0 + rmse)

        return stability

    def calculate_confidence(
        self,
        model: str,
        horizon: str,
        data_timestamp: datetime,
        adv: float = float("inf"),
        planned_dollars: float = 0.0,
    ) -> ConfidenceComponents:
        """
        Calculate overall confidence score.

        Args:
            model: Model family name
            horizon: Horizon
            data_timestamp: Data timestamp
            adv: Average daily volume (optional)
            planned_dollars: Planned trade size (optional)

        Returns:
            ConfidenceComponents with all factors
        """
        ic = self.calculate_ic(model, horizon)
        freshness = self.calculate_freshness(horizon, data_timestamp)
        capacity = self.calculate_capacity(adv, planned_dollars)
        stability = self.calculate_stability(model, horizon)

        # Combined confidence
        # Use max(IC, threshold) to avoid negative multiplication
        ic_adjusted = max(ic, MIN_IC_THRESHOLD)
        overall = ic_adjusted * freshness * capacity * stability

        return ConfidenceComponents(
            ic=ic,
            freshness=freshness,
            capacity=capacity,
            stability=stability,
            overall=overall,
        )

    def apply_confidence(
        self,
        standardized_prediction: float,
        confidence: float,
    ) -> float:
        """
        Apply confidence weighting to prediction.

        calibrated = standardized × confidence

        Args:
            standardized_prediction: Z-score standardized prediction
            confidence: Confidence score

        Returns:
            Calibrated prediction
        """
        return standardized_prediction * confidence
```

### 4. `LIVE_TRADING/prediction/predictor.py`
**Purpose:** Multi-horizon prediction coordinator

```python
"""
Multi-Horizon Predictor
=======================

Coordinates predictions across all horizons and model families.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import HORIZONS, FAMILIES
from LIVE_TRADING.models.loader import ModelLoader
from LIVE_TRADING.models.inference import InferenceEngine
from LIVE_TRADING.models.feature_builder import FeatureBuilder
from .standardization import ZScoreStandardizer
from .confidence import ConfidenceScorer, ConfidenceComponents

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Single model prediction with metadata."""
    family: str
    horizon: str
    raw: float
    standardized: float
    confidence: ConfidenceComponents
    calibrated: float  # standardized × confidence


@dataclass
class HorizonPredictions:
    """Predictions for a single horizon."""
    horizon: str
    timestamp: datetime
    predictions: Dict[str, ModelPrediction] = field(default_factory=dict)

    @property
    def families(self) -> List[str]:
        return list(self.predictions.keys())

    @property
    def mean_calibrated(self) -> float:
        if not self.predictions:
            return 0.0
        return np.mean([p.calibrated for p in self.predictions.values()])

    def get_calibrated_dict(self) -> Dict[str, float]:
        return {f: p.calibrated for f, p in sorted_items(self.predictions)}

    def get_standardized_dict(self) -> Dict[str, float]:
        return {f: p.standardized for f, p in sorted_items(self.predictions)}


@dataclass
class AllPredictions:
    """Predictions across all horizons."""
    symbol: str
    timestamp: datetime
    horizons: Dict[str, HorizonPredictions] = field(default_factory=dict)

    def get_horizon(self, horizon: str) -> Optional[HorizonPredictions]:
        return self.horizons.get(horizon)

    @property
    def available_horizons(self) -> List[str]:
        return list(self.horizons.keys())


class MultiHorizonPredictor:
    """
    Generates predictions across all horizons and model families.

    Pipeline:
    1. Build features from market data
    2. Run inference for each (horizon, family) pair
    3. Standardize predictions
    4. Calculate confidence
    5. Apply calibration
    """

    def __init__(
        self,
        run_root: str,
        horizons: List[str] | None = None,
        families: List[str] | None = None,
        device: str = "cpu",
    ):
        """
        Initialize multi-horizon predictor.

        Args:
            run_root: Path to TRAINING run artifacts
            horizons: Horizons to predict (default: all)
            families: Model families to use (default: all available)
            device: Device for inference
        """
        self.horizons = horizons or get_cfg(
            "live_trading.horizons", default=HORIZONS
        )
        self.families = families

        # Initialize components
        self.loader = ModelLoader(run_root)
        self.engine = InferenceEngine(self.loader, device=device)
        self.standardizer = ZScoreStandardizer()
        self.confidence_scorer = ConfidenceScorer()

        # Feature builders per target (cached)
        self._feature_builders: Dict[str, FeatureBuilder] = {}

        logger.info(f"MultiHorizonPredictor initialized: horizons={self.horizons}")

    def _get_feature_builder(
        self,
        target: str,
        family: str,
    ) -> FeatureBuilder:
        """Get or create feature builder for target/family."""
        key = f"{target}:{family}"
        if key not in self._feature_builders:
            feature_list = self.loader.get_feature_list(target, family)
            self._feature_builders[key] = FeatureBuilder(feature_list)
        return self._feature_builders[key]

    def predict_all_horizons(
        self,
        target: str,
        prices: Any,  # pd.DataFrame
        symbol: str,
        data_timestamp: datetime | None = None,
        adv: float = float("inf"),
        planned_dollars: float = 0.0,
    ) -> AllPredictions:
        """
        Generate predictions for all horizons.

        Args:
            target: Target name (e.g., "ret_5m")
            prices: OHLCV DataFrame
            symbol: Trading symbol
            data_timestamp: Data timestamp for freshness
            adv: Average daily volume
            planned_dollars: Planned trade size

        Returns:
            AllPredictions with all horizons
        """
        if data_timestamp is None:
            data_timestamp = datetime.now()

        # Get available families for this target
        available_families = self.loader.list_available_families(target)
        families_to_use = self.families or available_families

        all_preds = AllPredictions(
            symbol=symbol,
            timestamp=data_timestamp,
        )

        for horizon in self.horizons:
            horizon_preds = self._predict_horizon(
                target=target,
                horizon=horizon,
                prices=prices,
                symbol=symbol,
                families=families_to_use,
                data_timestamp=data_timestamp,
                adv=adv,
                planned_dollars=planned_dollars,
            )
            all_preds.horizons[horizon] = horizon_preds

        return all_preds

    def _predict_horizon(
        self,
        target: str,
        horizon: str,
        prices: Any,
        symbol: str,
        families: List[str],
        data_timestamp: datetime,
        adv: float,
        planned_dollars: float,
    ) -> HorizonPredictions:
        """Generate predictions for a single horizon."""
        horizon_preds = HorizonPredictions(
            horizon=horizon,
            timestamp=data_timestamp,
        )

        for family in families:
            try:
                pred = self._predict_single(
                    target=target,
                    horizon=horizon,
                    family=family,
                    prices=prices,
                    symbol=symbol,
                    data_timestamp=data_timestamp,
                    adv=adv,
                    planned_dollars=planned_dollars,
                )
                if pred is not None:
                    horizon_preds.predictions[family] = pred

            except Exception as e:
                logger.warning(f"Prediction failed for {family}/{horizon}: {e}")

        return horizon_preds

    def _predict_single(
        self,
        target: str,
        horizon: str,
        family: str,
        prices: Any,
        symbol: str,
        data_timestamp: datetime,
        adv: float,
        planned_dollars: float,
    ) -> Optional[ModelPrediction]:
        """Generate single model prediction."""

        # Build features
        builder = self._get_feature_builder(target, family)
        features = builder.build_features(prices, symbol)

        if np.any(np.isnan(features)):
            logger.warning(f"NaN in features for {family}/{symbol}")
            return None

        # Run inference
        raw_pred = self.engine.predict(target, family, features, symbol)

        if np.isnan(raw_pred):
            return None

        # Standardize
        std_pred = self.standardizer.standardize(raw_pred, family, horizon)

        # Calculate confidence
        confidence = self.confidence_scorer.calculate_confidence(
            model=family,
            horizon=horizon,
            data_timestamp=data_timestamp,
            adv=adv,
            planned_dollars=planned_dollars,
        )

        # Calibrated prediction
        calibrated = self.confidence_scorer.apply_confidence(
            std_pred, confidence.overall
        )

        return ModelPrediction(
            family=family,
            horizon=horizon,
            raw=raw_pred,
            standardized=std_pred,
            confidence=confidence,
            calibrated=calibrated,
        )

    def update_actuals(
        self,
        model: str,
        horizon: str,
        prediction: float,
        actual_return: float,
    ) -> None:
        """
        Update with actual returns for IC tracking.

        Args:
            model: Model family
            horizon: Horizon
            prediction: Previous prediction
            actual_return: Realized return
        """
        self.confidence_scorer.update_with_actual(
            model, horizon, prediction, actual_return
        )
```

## Tests

### `LIVE_TRADING/tests/test_standardization.py`

```python
"""Tests for Z-score standardization."""

import pytest
import numpy as np

from LIVE_TRADING.prediction.standardization import ZScoreStandardizer


class TestZScoreStandardizer:
    def test_init(self):
        std = ZScoreStandardizer(window_size=10)
        assert std.window_size == 10

    def test_warmup_returns_clipped(self):
        std = ZScoreStandardizer(window_size=10)

        # First few predictions should return clipped values
        result = std.standardize(5.0, "LightGBM", "5m")
        assert -3.0 <= result <= 3.0

    def test_standardization(self):
        std = ZScoreStandardizer(window_size=10)

        # Push consistent values
        for _ in range(10):
            std.standardize(0.0, "LightGBM", "5m")

        # Extreme value should be clipped
        result = std.standardize(10.0, "LightGBM", "5m")
        assert result == 3.0  # Clipped to max

    def test_separate_buffers_per_model(self):
        std = ZScoreStandardizer(window_size=5)

        # Different models should have separate buffers
        for i in range(5):
            std.standardize(1.0, "LightGBM", "5m")
            std.standardize(100.0, "XGBoost", "5m")

        # Stats should be different
        stats_lgb = std.get_stats("LightGBM", "5m")
        stats_xgb = std.get_stats("XGBoost", "5m")

        assert stats_lgb.mean != stats_xgb.mean

    def test_batch_standardization(self):
        std = ZScoreStandardizer(window_size=10)

        preds = {"LightGBM": 0.5, "XGBoost": 0.6}
        results = std.standardize_batch(preds, "5m")

        assert "LightGBM" in results
        assert "XGBoost" in results


class TestConfidenceScorer:
    # Tests in separate file for clarity
    pass
```

### `LIVE_TRADING/tests/test_predictor.py`

```python
"""Tests for multi-horizon predictor."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import numpy as np

from LIVE_TRADING.prediction.predictor import (
    MultiHorizonPredictor,
    ModelPrediction,
    HorizonPredictions,
    AllPredictions,
)


@pytest.fixture
def mock_loader():
    loader = Mock()
    loader.list_available_families.return_value = ["LightGBM"]
    loader.get_feature_list.return_value = ["ret_1d", "vol_10d"]
    return loader


class TestHorizonPredictions:
    def test_mean_calibrated(self):
        hp = HorizonPredictions(horizon="5m", timestamp=datetime.now())

        # Add mock predictions
        hp.predictions["LightGBM"] = Mock(calibrated=0.5)
        hp.predictions["XGBoost"] = Mock(calibrated=0.3)

        assert hp.mean_calibrated == pytest.approx(0.4)

    def test_empty_mean(self):
        hp = HorizonPredictions(horizon="5m", timestamp=datetime.now())
        assert hp.mean_calibrated == 0.0


class TestAllPredictions:
    def test_available_horizons(self):
        ap = AllPredictions(symbol="AAPL", timestamp=datetime.now())
        ap.horizons["5m"] = Mock()
        ap.horizons["10m"] = Mock()

        assert "5m" in ap.available_horizons
        assert "10m" in ap.available_horizons
```

## SST Compliance Checklist

- [ ] Uses `get_cfg()` for all configuration
- [ ] Uses `sorted_items()` for dict iteration in `get_calibrated_dict()`
- [ ] Proper dataclass definitions for type safety
- [ ] Rolling buffers use `deque` with `maxlen` for memory efficiency
- [ ] All datetime handling uses UTC

## Dependencies

- `CONFIG.config_loader.get_cfg`
- `TRAINING.common.utils.determinism_ordering.sorted_items`
- `LIVE_TRADING.models.loader.ModelLoader`
- `LIVE_TRADING.models.inference.InferenceEngine`
- External: `scipy.stats` for Spearman correlation

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `__init__.py` | 15 |
| `standardization.py` | 180 |
| `confidence.py` | 220 |
| `predictor.py` | 280 |
| `tests/test_standardization.py` | 80 |
| `tests/test_predictor.py` | 80 |
| **Total** | ~855 |
