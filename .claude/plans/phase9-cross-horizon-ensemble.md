# Phase 9: Cross-Horizon Ensemble

**Status**: Not Started
**Parent Plan**: [multi_horizon_training_master.md](./multi_horizon_training_master.md)
**Estimated Effort**: 2-3 days
**Depends On**: Phase 8 (Multi-Horizon Training)

---

## Overview

Cross-horizon ensemble combines predictions from models trained on different horizons to produce better final predictions. Shorter horizons (5m, 15m) capture recent momentum while longer horizons (60m, 1d) provide trend context.

This is similar to what the live trading system does with `ridge_risk_parity` blending, but applied during training validation.

---

## Subphases

| Subphase | Description | Effort | Files |
|----------|-------------|--------|-------|
| 9.1 | CrossHorizonStacker | 6h | `cross_horizon_ensemble.py` |
| 9.2 | Decay functions | 3h | `horizon_decay.py` |
| 9.3 | Ensemble integration | 4h | `ensemble_trainer.py`, config |
| 9.4 | Tests | 3h | `test_cross_horizon_ensemble.py` |

---

## Subphase 9.1: CrossHorizonStacker

### Goal
Create a stacking ensemble that blends predictions across horizons with learned weights.

### New File: `TRAINING/model_fun/cross_horizon_ensemble.py`

```python
"""
Cross-Horizon Ensemble

Blends predictions from models trained on different horizons.
Uses ridge regression with optional decay priors.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

logger = logging.getLogger(__name__)


class CrossHorizonStacker:
    """
    Stacking ensemble that combines predictions across horizons.

    The key insight: For a 15-minute prediction, both 5m and 60m models
    provide useful information:
    - 5m model: Recent momentum, short-term mean reversion
    - 60m model: Longer-term trend, structural factors

    Architecture:
        pred_5m  ──┐
        pred_15m ──┼──► Ridge Meta-Learner ──► final_pred
        pred_60m ──┘

    With optional horizon decay that weights nearer horizons higher.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.meta_model = None
        self.horizons = []
        self.target_horizon = None
        self.weights = None
        self._decay_function = None

    def fit(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        horizons: Dict[str, int],
        target_horizon: int,
        decay_function: Optional[str] = None,
        decay_params: Optional[Dict] = None
    ) -> 'CrossHorizonStacker':
        """
        Fit the stacking meta-learner.

        Args:
            predictions: Dict of model_name → predictions (N,)
            y_true: Ground truth labels (N,)
            horizons: Dict of model_name → horizon_minutes
            target_horizon: Target prediction horizon in minutes
            decay_function: 'exponential', 'linear', 'inverse', or None
            decay_params: Parameters for decay function

        Returns:
            self
        """
        self.horizons = horizons
        self.target_horizon = target_horizon

        # Stack predictions
        model_names = sorted(predictions.keys())  # Deterministic order
        X_stack = np.column_stack([predictions[name] for name in model_names])

        # Compute decay weights if specified
        sample_weight = None
        if decay_function:
            from TRAINING.model_fun.horizon_decay import get_decay_function
            self._decay_function = get_decay_function(
                decay_function, decay_params or {}
            )

            # Compute horizon distances from target
            horizon_list = [horizons[name] for name in model_names]
            decay_weights = self._compute_decay_weights(horizon_list, target_horizon)

            # Create regularization prior (penalize distant horizons more)
            alpha = self.config.get('ridge_alpha', 1.0)
            # Adjust alpha inversely to decay weights
            effective_alpha = alpha / (decay_weights + 1e-6)

            # For sklearn Ridge, we use sample_weight as proxy
            # Better approach: custom regularization, but Ridge is simpler
            logger.debug(f"Decay weights: {dict(zip(model_names, decay_weights))}")

        # Fit ridge meta-learner
        ridge_alpha = self.config.get('ridge_alpha', 1.0)
        self.meta_model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        self.meta_model.fit(X_stack, y_true)

        # Store weights
        self.weights = dict(zip(model_names, self.meta_model.coef_))
        logger.info(f"Fitted cross-horizon stacker with weights: {self.weights}")

        return self

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict using the stacked ensemble.

        Args:
            predictions: Dict of model_name → predictions

        Returns:
            Blended predictions
        """
        if self.meta_model is None:
            raise RuntimeError("Stacker not fitted")

        model_names = sorted(predictions.keys())
        X_stack = np.column_stack([predictions[name] for name in model_names])

        return self.meta_model.predict(X_stack)

    def predict_with_confidence(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence based on horizon agreement.

        Args:
            predictions: Dict of model_name → predictions

        Returns:
            (predictions, confidence scores)
        """
        pred = self.predict(predictions)

        # Confidence = 1 - normalized std across horizons
        model_names = sorted(predictions.keys())
        X_stack = np.column_stack([predictions[name] for name in model_names])

        # Per-sample std
        std = np.std(X_stack, axis=1)
        max_std = np.percentile(std, 95) + 1e-6

        # Normalize to [0, 1], high agreement = high confidence
        confidence = 1.0 - np.clip(std / max_std, 0, 1)

        return pred, confidence

    def _compute_decay_weights(
        self,
        horizons: List[int],
        target_horizon: int
    ) -> np.ndarray:
        """
        Compute decay weights based on horizon distance.

        Args:
            horizons: List of horizon values
            target_horizon: Target horizon

        Returns:
            Array of weights (higher = more relevant)
        """
        if self._decay_function is None:
            return np.ones(len(horizons))

        distances = np.abs(np.array(horizons) - target_horizon)
        weights = self._decay_function(distances)

        # Normalize
        weights = weights / weights.sum()

        return weights

    def get_weights(self) -> Dict[str, float]:
        """Get learned blend weights."""
        return self.weights.copy() if self.weights else {}

    def get_effective_weights(
        self,
        include_decay: bool = True
    ) -> Dict[str, float]:
        """
        Get effective weights including decay adjustment.

        Args:
            include_decay: Whether to include decay adjustment

        Returns:
            Dict of model_name → effective_weight
        """
        if not self.weights:
            return {}

        weights = self.weights.copy()

        if include_decay and self._decay_function:
            model_names = sorted(weights.keys())
            horizon_list = [self.horizons[name] for name in model_names]
            decay_weights = self._compute_decay_weights(
                horizon_list, self.target_horizon
            )

            for name, decay in zip(model_names, decay_weights):
                weights[name] *= decay

            # Re-normalize
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

        return weights


class CrossHorizonEnsembleTrainer:
    """
    Full ensemble trainer that trains individual models then stacks.

    Flow:
        1. Train per-horizon models (LightGBM, XGBoost, etc.)
        2. Generate OOF predictions for each model
        3. Fit CrossHorizonStacker on OOF predictions
        4. Final model = stacker(individual models)
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.horizon_models = {}  # horizon → model
        self.stacker = None
        self.target_horizon = None

    def train(
        self,
        X_tr: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        X_va: Optional[np.ndarray] = None,
        y_va_dict: Optional[Dict[str, np.ndarray]] = None,
        target_column: str = None,
        base_family: str = 'lightgbm',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train cross-horizon ensemble.

        Args:
            X_tr: Training features
            y_dict: Dict of horizon targets (fwd_ret_5m, fwd_ret_15m, etc.)
            X_va: Validation features
            y_va_dict: Validation targets
            target_column: Which horizon to optimize for
            base_family: Base model family

        Returns:
            Training results
        """
        from TRAINING.common.horizon_bundle import parse_horizon_from_target

        # Determine target horizon
        if target_column:
            _, self.target_horizon = parse_horizon_from_target(target_column)
        else:
            # Default to middle horizon
            horizons = []
            for target in y_dict.keys():
                _, h = parse_horizon_from_target(target)
                if h:
                    horizons.append(h)
            self.target_horizon = sorted(horizons)[len(horizons) // 2] if horizons else 15

        # Step 1: Train individual models per horizon
        oof_predictions = {}
        horizons_map = {}

        for target, y in y_dict.items():
            _, horizon = parse_horizon_from_target(target)
            if horizon is None:
                continue

            logger.info(f"Training {base_family} for horizon {horizon}m")

            model = self._train_single_model(
                X_tr, y, base_family, **kwargs
            )

            self.horizon_models[target] = model
            horizons_map[target] = horizon

            # Generate OOF predictions
            oof_predictions[target] = self._generate_oof_predictions(
                model, X_tr, y, **kwargs
            )

        # Step 2: Fit stacker
        target_y = y_dict.get(target_column) or list(y_dict.values())[0]

        self.stacker = CrossHorizonStacker(self.config.get('stacker', {}))
        self.stacker.fit(
            predictions=oof_predictions,
            y_true=target_y,
            horizons=horizons_map,
            target_horizon=self.target_horizon,
            decay_function=self.config.get('decay_function'),
            decay_params=self.config.get('decay_params')
        )

        # Compute metrics
        stacked_pred = self.stacker.predict(oof_predictions)
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(target_y, stacked_pred)
        rmse = np.sqrt(mean_squared_error(target_y, stacked_pred))

        return {
            'r2': r2,
            'rmse': rmse,
            'horizons': list(horizons_map.keys()),
            'weights': self.stacker.get_weights(),
            'target_horizon': self.target_horizon
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate blended prediction."""
        predictions = {}
        for target, model in self.horizon_models.items():
            predictions[target] = model.predict(X)

        return self.stacker.predict(predictions)

    def _train_single_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        family: str,
        **kwargs
    ):
        """Train a single model."""
        if family.lower() == 'lightgbm':
            import lightgbm as lgb
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'num_leaves': kwargs.get('num_leaves', 31),
                'learning_rate': kwargs.get('learning_rate', 0.05),
                'n_estimators': kwargs.get('n_estimators', 100),
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            return model

        elif family.lower() == 'ridge':
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=kwargs.get('alpha', 1.0))
            model.fit(X, y)
            return model

        else:
            raise ValueError(f"Unknown family: {family}")

    def _generate_oof_predictions(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        **kwargs
    ) -> np.ndarray:
        """Generate out-of-fold predictions for stacking."""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(y))

        for train_idx, val_idx in kf.split(X):
            X_fold_tr, X_fold_va = X[train_idx], X[val_idx]
            y_fold_tr = y[train_idx]

            # Clone and retrain
            model_clone = self._train_single_model(
                X_fold_tr, y_fold_tr,
                family='lightgbm',  # TODO: Detect from model
                **kwargs
            )

            oof[val_idx] = model_clone.predict(X_fold_va)

        return oof
```

---

## Subphase 9.2: Decay Functions

### Goal
Implement horizon decay functions that weight nearer horizons higher.

### New File: `TRAINING/model_fun/horizon_decay.py`

```python
"""
Horizon Decay Functions

Weight nearer horizons higher when blending predictions.
"""

import numpy as np
from typing import Dict, Callable, Any


def exponential_decay(
    distances: np.ndarray,
    half_life: float = 30.0
) -> np.ndarray:
    """
    Exponential decay based on horizon distance.

    Args:
        distances: Array of |horizon - target_horizon| values
        half_life: Distance at which weight = 0.5

    Returns:
        Decay weights (higher = more relevant)
    """
    # w = exp(-ln(2) * distance / half_life)
    return np.exp(-np.log(2) * distances / half_life)


def linear_decay(
    distances: np.ndarray,
    max_distance: float = 120.0
) -> np.ndarray:
    """
    Linear decay based on horizon distance.

    Args:
        distances: Array of distances
        max_distance: Distance at which weight = 0

    Returns:
        Decay weights
    """
    weights = 1.0 - (distances / max_distance)
    return np.maximum(weights, 0.0)


def inverse_decay(
    distances: np.ndarray,
    epsilon: float = 1.0
) -> np.ndarray:
    """
    Inverse decay: w = 1 / (distance + epsilon)

    Args:
        distances: Array of distances
        epsilon: Smoothing parameter

    Returns:
        Decay weights
    """
    return 1.0 / (distances + epsilon)


def step_decay(
    distances: np.ndarray,
    threshold: float = 30.0,
    inside_weight: float = 1.0,
    outside_weight: float = 0.1
) -> np.ndarray:
    """
    Step function decay.

    Args:
        distances: Array of distances
        threshold: Distance threshold
        inside_weight: Weight for horizons within threshold
        outside_weight: Weight for horizons outside threshold

    Returns:
        Decay weights
    """
    weights = np.where(
        distances <= threshold,
        inside_weight,
        outside_weight
    )
    return weights


DECAY_FUNCTIONS: Dict[str, Callable] = {
    'exponential': exponential_decay,
    'linear': linear_decay,
    'inverse': inverse_decay,
    'step': step_decay,
}


def get_decay_function(
    name: str,
    params: Dict[str, Any]
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get a decay function by name with bound parameters.

    Args:
        name: Function name
        params: Parameters to bind

    Returns:
        Callable that takes distances and returns weights
    """
    if name not in DECAY_FUNCTIONS:
        raise ValueError(f"Unknown decay function: {name}")

    base_fn = DECAY_FUNCTIONS[name]

    def bound_fn(distances: np.ndarray) -> np.ndarray:
        return base_fn(distances, **params)

    return bound_fn


def compute_optimal_decay(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    horizons: Dict[str, int],
    target_horizon: int,
    decay_types: list = None
) -> Dict[str, Any]:
    """
    Find optimal decay function via grid search.

    Args:
        predictions: Model predictions
        y_true: Ground truth
        horizons: Model horizons
        target_horizon: Target horizon
        decay_types: List of decay functions to try

    Returns:
        Best decay configuration
    """
    from sklearn.metrics import r2_score

    if decay_types is None:
        decay_types = ['exponential', 'linear', 'inverse']

    best_r2 = -np.inf
    best_config = None

    for decay_type in decay_types:
        # Grid search over parameters
        if decay_type == 'exponential':
            param_grid = [{'half_life': h} for h in [15, 30, 60, 120]]
        elif decay_type == 'linear':
            param_grid = [{'max_distance': d} for d in [60, 120, 240]]
        elif decay_type == 'inverse':
            param_grid = [{'epsilon': e} for e in [1, 5, 10]]
        else:
            param_grid = [{}]

        for params in param_grid:
            decay_fn = get_decay_function(decay_type, params)

            # Compute weighted blend
            model_names = sorted(predictions.keys())
            horizon_list = [horizons[name] for name in model_names]
            distances = np.abs(np.array(horizon_list) - target_horizon)
            weights = decay_fn(distances)
            weights = weights / weights.sum()

            # Blend predictions
            X_stack = np.column_stack([predictions[name] for name in model_names])
            blended = np.dot(X_stack, weights)

            # Evaluate
            r2 = r2_score(y_true, blended)

            if r2 > best_r2:
                best_r2 = r2
                best_config = {
                    'decay_type': decay_type,
                    'params': params,
                    'weights': dict(zip(model_names, weights)),
                    'r2': r2
                }

    return best_config
```

---

## Subphase 9.3: Ensemble Integration

### Modified: `TRAINING/model_fun/ensemble_trainer.py`

Add cross-horizon support to existing ensemble:

```python
# Add to EnsembleTrainer class

def train_cross_horizon(
    self,
    X_tr: np.ndarray,
    y_dict: Dict[str, np.ndarray],
    X_va: Optional[np.ndarray] = None,
    y_va_dict: Optional[Dict[str, np.ndarray]] = None,
    target_horizon_minutes: int = 15,
    **kwargs
) -> Dict[str, Any]:
    """
    Train ensemble with cross-horizon stacking.

    Args:
        X_tr: Training features
        y_dict: Dict of horizon targets
        X_va: Validation features
        y_va_dict: Validation targets
        target_horizon_minutes: Target prediction horizon

    Returns:
        Training results
    """
    from TRAINING.model_fun.cross_horizon_ensemble import CrossHorizonStacker
    from TRAINING.model_fun.horizon_decay import compute_optimal_decay

    # Train individual models per horizon
    horizon_models = {}
    oof_predictions = {}

    for target, y in y_dict.items():
        # Train standard ensemble for this horizon
        result = self.train(X_tr, y, X_va, y_va_dict.get(target), **kwargs)
        horizon_models[target] = self.model

        # Generate OOF predictions
        oof_predictions[target] = self._generate_oof(X_tr, y, **kwargs)

    # Find optimal decay
    target_key = f"fwd_ret_{target_horizon_minutes}m"
    target_y = y_dict.get(target_key, list(y_dict.values())[0])

    horizons_map = {}
    for target in y_dict.keys():
        from TRAINING.common.horizon_bundle import parse_horizon_from_target
        _, h = parse_horizon_from_target(target)
        if h:
            horizons_map[target] = h

    optimal_decay = compute_optimal_decay(
        oof_predictions, target_y, horizons_map, target_horizon_minutes
    )

    # Fit stacker with optimal decay
    stacker_config = {
        'ridge_alpha': kwargs.get('ridge_alpha', 1.0),
    }

    stacker = CrossHorizonStacker(stacker_config)
    stacker.fit(
        predictions=oof_predictions,
        y_true=target_y,
        horizons=horizons_map,
        target_horizon=target_horizon_minutes,
        decay_function=optimal_decay['decay_type'],
        decay_params=optimal_decay['params']
    )

    self._cross_horizon_stacker = stacker
    self._horizon_models = horizon_models

    return {
        'horizon_models': list(horizon_models.keys()),
        'optimal_decay': optimal_decay,
        'stacker_weights': stacker.get_weights(),
        'cross_horizon_r2': optimal_decay['r2']
    }
```

### Config: Add to `CONFIG/pipeline/training/intelligent.yaml`

```yaml
ensemble:
  # Existing options...

  # Cross-horizon stacking (Phase 9)
  cross_horizon:
    enabled: false
    base_horizons: [5, 15, 60]  # minutes

    # Decay function
    decay_function: exponential
    decay_params:
      half_life: 30

    # Stacker
    ridge_alpha: 1.0

    # Auto-tune decay
    auto_tune_decay: true
    decay_types_to_try:
      - exponential
      - linear
      - inverse
```

---

## Subphase 9.4: Tests

### New File: `tests/test_cross_horizon_ensemble.py`

```python
import pytest
import numpy as np
from TRAINING.model_fun.cross_horizon_ensemble import (
    CrossHorizonStacker,
    CrossHorizonEnsembleTrainer,
)
from TRAINING.model_fun.horizon_decay import (
    exponential_decay,
    linear_decay,
    get_decay_function,
    compute_optimal_decay,
)


class TestDecayFunctions:
    def test_exponential_decay(self):
        distances = np.array([0, 15, 30, 60])
        weights = exponential_decay(distances, half_life=30)

        # At distance=0, weight should be 1
        assert weights[0] == pytest.approx(1.0)
        # At distance=half_life, weight should be 0.5
        assert weights[2] == pytest.approx(0.5, rel=0.01)
        # Weights should decrease with distance
        assert all(weights[i] > weights[i+1] for i in range(len(weights)-1))

    def test_linear_decay(self):
        distances = np.array([0, 30, 60, 120])
        weights = linear_decay(distances, max_distance=120)

        assert weights[0] == pytest.approx(1.0)
        assert weights[3] == pytest.approx(0.0)

    def test_get_decay_function(self):
        fn = get_decay_function('exponential', {'half_life': 30})
        result = fn(np.array([30]))
        assert result[0] == pytest.approx(0.5, rel=0.01)


class TestCrossHorizonStacker:
    def test_fit_predict(self):
        np.random.seed(42)

        # Create synthetic predictions
        n = 100
        predictions = {
            'fwd_ret_5m': np.random.randn(n),
            'fwd_ret_15m': np.random.randn(n),
            'fwd_ret_60m': np.random.randn(n),
        }
        horizons = {'fwd_ret_5m': 5, 'fwd_ret_15m': 15, 'fwd_ret_60m': 60}

        # True is correlated with 15m prediction
        y_true = 0.5 * predictions['fwd_ret_15m'] + np.random.randn(n) * 0.5

        stacker = CrossHorizonStacker()
        stacker.fit(predictions, y_true, horizons, target_horizon=15)

        # Should learn higher weight for 15m
        weights = stacker.get_weights()
        assert weights['fwd_ret_15m'] > weights['fwd_ret_5m']
        assert weights['fwd_ret_15m'] > weights['fwd_ret_60m']

        # Predict
        preds = stacker.predict(predictions)
        assert len(preds) == n

    def test_with_decay(self):
        np.random.seed(42)
        n = 100

        predictions = {
            'fwd_ret_5m': np.random.randn(n),
            'fwd_ret_60m': np.random.randn(n),
        }
        horizons = {'fwd_ret_5m': 5, 'fwd_ret_60m': 60}
        y_true = np.random.randn(n)

        stacker = CrossHorizonStacker({'ridge_alpha': 1.0})
        stacker.fit(
            predictions, y_true, horizons,
            target_horizon=15,
            decay_function='exponential',
            decay_params={'half_life': 30}
        )

        effective_weights = stacker.get_effective_weights(include_decay=True)
        # 5m is closer to 15m, should have higher effective weight
        assert effective_weights['fwd_ret_5m'] > effective_weights['fwd_ret_60m']


class TestOptimalDecay:
    def test_finds_best_decay(self):
        np.random.seed(42)
        n = 100

        # 15m model is best for 15m target
        predictions = {
            'fwd_ret_5m': np.random.randn(n),
            'fwd_ret_15m': np.random.randn(n),
            'fwd_ret_60m': np.random.randn(n),
        }
        horizons = {'fwd_ret_5m': 5, 'fwd_ret_15m': 15, 'fwd_ret_60m': 60}
        y_true = predictions['fwd_ret_15m'] + np.random.randn(n) * 0.1

        result = compute_optimal_decay(
            predictions, y_true, horizons, target_horizon=15
        )

        assert result['r2'] > 0.5
        assert 'decay_type' in result
        assert 'weights' in result
```

---

## Config Summary

### New Config Keys

```yaml
# CONFIG/pipeline/training/intelligent.yaml

ensemble:
  cross_horizon:
    enabled: false
    base_horizons: [5, 15, 60]
    decay_function: exponential
    decay_params:
      half_life: 30
    ridge_alpha: 1.0
    auto_tune_decay: true
```

### Example Experiment

```yaml
# CONFIG/experiments/cross_horizon_example.yaml

experiment:
  name: cross_horizon_ensemble
  description: Ensemble with cross-horizon stacking

intelligent_training:
  auto_targets: true
  include_target_patterns: ["fwd_ret"]
  exclude_target_patterns: ["will_peak", "mdd"]

training:
  model_families:
    - lightgbm

  ensemble:
    cross_horizon:
      enabled: true
      base_horizons: [5, 15, 30, 60]
      decay_function: exponential
      decay_params:
        half_life: 30
      auto_tune_decay: true
```

---

## Validation Checklist

### Subphase 9.1
- [ ] `CrossHorizonStacker.fit()` learns sensible weights
- [ ] `CrossHorizonStacker.predict()` produces valid output
- [ ] Weights sum to reasonable value (close to 1)

### Subphase 9.2
- [ ] All decay functions produce valid weights
- [ ] `compute_optimal_decay()` finds best configuration
- [ ] Decay weights higher for nearer horizons

### Subphase 9.3
- [ ] `train_cross_horizon()` integrates with existing ensemble
- [ ] Config loads without error
- [ ] Deterministic: same seeds → same weights

### Subphase 9.4
- [ ] All unit tests pass
- [ ] Integration test with real data

---

## Test Commands

```bash
# Unit tests
pytest tests/test_cross_horizon_ensemble.py -v

# With real data (requires lightgbm)
pytest tests/test_cross_horizon_ensemble.py -v --runslow
```

---

## Success Metrics

- [ ] Cross-horizon ensemble improves R² by 5%+ over single-horizon
- [ ] Optimal decay function found automatically
- [ ] Inference latency < 2x single model
- [ ] Weights interpretable (nearer horizons weighted higher)

---

## Next Steps After Phase 9

1. **Phase 10**: Multi-interval experiments (train at 5m, validate at 1m)
2. **Metrics dashboard**: Visualize per-horizon and blended metrics
3. **Live trading integration**: Use same blending in inference
