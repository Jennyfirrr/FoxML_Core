# Phase 8: Multi-Horizon Training

**Status**: Not Started
**Parent Plan**: [multi_horizon_training_master.md](./multi_horizon_training_master.md)
**Estimated Effort**: 2-3 days

---

## Overview

Multi-horizon training allows training a single model with shared representations across multiple prediction horizons (e.g., 5m, 15m, 60m). This reduces training time and improves generalization by sharing learned features across related tasks.

---

## Subphases

| Subphase | Description | Effort | Files |
|----------|-------------|--------|-------|
| 8.1 | HorizonBundle type + grouping | 4h | `horizon_bundle.py`, `horizon_ranker.py` |
| 8.2 | MultiHorizonTrainer | 6h | `multi_horizon_trainer.py` |
| 8.3 | Training loop integration | 4h | `training.py`, `intelligent_trainer.py` |
| 8.4 | Config + tests | 4h | `intelligent.yaml`, `test_*.py` |

---

## Subphase 8.1: HorizonBundle Type + Grouping

### Goal
Create data structures for grouping related targets by horizon and ranking bundles by diversity.

### New File: `TRAINING/common/horizon_bundle.py`

```python
"""
Horizon Bundle Types

Groups related targets across prediction horizons for multi-horizon training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
import numpy as np


@dataclass
class HorizonBundle:
    """
    A bundle of related targets across multiple horizons.

    Example:
        bundle = HorizonBundle(
            base_name="fwd_ret",
            horizons=[5, 15, 60],
            targets=["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m"]
        )
    """
    base_name: str  # e.g., "fwd_ret", "will_peak"
    horizons: List[int]  # minutes: [5, 15, 60]
    targets: List[str]  # ["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m"]

    # Diversity metrics (populated during ranking)
    correlation_matrix: Optional[np.ndarray] = None
    diversity_score: float = 0.0

    # Training config
    loss_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.loss_weights:
            # Default: equal weights
            self.loss_weights = {t: 1.0 for t in self.targets}

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)

    @property
    def primary_horizon(self) -> int:
        """Middle horizon (or shortest if 2 horizons)."""
        if len(self.horizons) == 1:
            return self.horizons[0]
        return sorted(self.horizons)[len(self.horizons) // 2]

    @property
    def primary_target(self) -> str:
        """Target corresponding to primary horizon."""
        idx = self.horizons.index(self.primary_horizon)
        return self.targets[idx]


def parse_horizon_from_target(target: str) -> Tuple[str, Optional[int]]:
    """
    Extract base name and horizon from target string.

    Args:
        target: Target name (e.g., "fwd_ret_60m", "will_peak_5m_0.8")

    Returns:
        (base_name, horizon_minutes) or (target, None) if not parseable

    Examples:
        "fwd_ret_60m" → ("fwd_ret", 60)
        "will_peak_5m_0.8" → ("will_peak", 5)
        "custom_target" → ("custom_target", None)
    """
    # Pattern: base_name_Nm or base_name_Nm_threshold
    pattern = r'^(.+?)_(\d+)m(?:_[\d.]+)?$'
    match = re.match(pattern, target)

    if match:
        base_name = match.group(1)
        horizon = int(match.group(2))
        return base_name, horizon

    return target, None


def group_targets_by_base(targets: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    """
    Group targets by their base name.

    Args:
        targets: List of target names

    Returns:
        Dict mapping base_name → [(target, horizon), ...]
    """
    groups = {}

    for target in targets:
        base_name, horizon = parse_horizon_from_target(target)
        if horizon is not None:
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append((target, horizon))

    # Sort each group by horizon
    for base_name in groups:
        groups[base_name] = sorted(groups[base_name], key=lambda x: x[1])

    return groups


def create_bundles_from_targets(
    targets: List[str],
    min_horizons: int = 2,
    max_horizons: int = 5
) -> List[HorizonBundle]:
    """
    Auto-create horizon bundles from target list.

    Args:
        targets: All available targets
        min_horizons: Minimum horizons for a bundle
        max_horizons: Maximum horizons per bundle

    Returns:
        List of HorizonBundle objects
    """
    groups = group_targets_by_base(targets)
    bundles = []

    for base_name, target_horizons in groups.items():
        if len(target_horizons) < min_horizons:
            continue

        # Limit to max_horizons (take evenly spaced)
        if len(target_horizons) > max_horizons:
            indices = np.linspace(0, len(target_horizons) - 1, max_horizons, dtype=int)
            target_horizons = [target_horizons[i] for i in indices]

        bundle = HorizonBundle(
            base_name=base_name,
            horizons=[h for _, h in target_horizons],
            targets=[t for t, _ in target_horizons]
        )
        bundles.append(bundle)

    return bundles


def compute_bundle_diversity(
    bundle: HorizonBundle,
    y_dict: Dict[str, np.ndarray]
) -> float:
    """
    Compute diversity score for a bundle based on target correlations.

    Low correlation = high diversity = good for multi-task learning.

    Args:
        bundle: HorizonBundle to score
        y_dict: Dict of target_name → target_values

    Returns:
        Diversity score (0-1, higher = more diverse)
    """
    if bundle.n_horizons < 2:
        return 1.0  # Single horizon is trivially "diverse"

    # Compute correlation matrix
    targets_data = []
    for target in bundle.targets:
        if target in y_dict:
            targets_data.append(y_dict[target].flatten())

    if len(targets_data) < 2:
        return 0.0

    # Stack and compute correlation
    data_matrix = np.column_stack(targets_data)
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)

    bundle.correlation_matrix = corr_matrix

    # Diversity = 1 - mean absolute off-diagonal correlation
    n = corr_matrix.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    mean_abs_corr = np.mean(np.abs(corr_matrix[off_diag_mask]))

    bundle.diversity_score = 1.0 - mean_abs_corr
    return bundle.diversity_score
```

### New File: `TRAINING/orchestration/horizon_ranker.py`

```python
"""
Horizon Bundle Ranker

Ranks bundles by diversity and predictability for multi-horizon training.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from TRAINING.common.horizon_bundle import (
    HorizonBundle,
    create_bundles_from_targets,
    compute_bundle_diversity,
)

logger = logging.getLogger(__name__)


def rank_bundles(
    bundles: List[HorizonBundle],
    y_dict: Dict[str, np.ndarray],
    target_scores: Optional[Dict[str, float]] = None,
    diversity_weight: float = 0.3,
    predictability_weight: float = 0.7
) -> List[HorizonBundle]:
    """
    Rank horizon bundles by combined diversity and predictability.

    Args:
        bundles: List of bundles to rank
        y_dict: Target values for diversity calculation
        target_scores: Optional dict of target → predictability score
        diversity_weight: Weight for diversity in ranking
        predictability_weight: Weight for predictability in ranking

    Returns:
        Bundles sorted by combined score (best first)
    """
    for bundle in bundles:
        # Compute diversity
        diversity = compute_bundle_diversity(bundle, y_dict)

        # Compute average predictability (if scores provided)
        if target_scores:
            predictabilities = [
                target_scores.get(t, 0.0) for t in bundle.targets
            ]
            predictability = np.mean(predictabilities) if predictabilities else 0.0
        else:
            predictability = 0.5  # Neutral if no scores

        # Combined score
        bundle.combined_score = (
            diversity_weight * diversity +
            predictability_weight * predictability
        )

    # Sort by combined score (descending)
    return sorted(bundles, key=lambda b: b.combined_score, reverse=True)


def filter_bundles_by_diversity(
    bundles: List[HorizonBundle],
    min_diversity: float = 0.3
) -> List[HorizonBundle]:
    """
    Filter out bundles with low diversity (too correlated).

    Args:
        bundles: Bundles to filter
        min_diversity: Minimum diversity score

    Returns:
        Bundles with diversity >= min_diversity
    """
    return [b for b in bundles if b.diversity_score >= min_diversity]


def select_top_bundles(
    targets: List[str],
    y_dict: Dict[str, np.ndarray],
    target_scores: Optional[Dict[str, float]] = None,
    top_n: int = 3,
    min_diversity: float = 0.3,
    min_horizons: int = 2,
    max_horizons: int = 5
) -> List[HorizonBundle]:
    """
    Full bundle selection pipeline.

    Args:
        targets: All available targets
        y_dict: Target values
        target_scores: Predictability scores
        top_n: Number of bundles to return
        min_diversity: Minimum diversity threshold
        min_horizons: Minimum horizons per bundle
        max_horizons: Maximum horizons per bundle

    Returns:
        Top N bundles, ranked by quality
    """
    # Create bundles
    bundles = create_bundles_from_targets(
        targets, min_horizons=min_horizons, max_horizons=max_horizons
    )

    if not bundles:
        logger.warning("No horizon bundles found in targets")
        return []

    logger.info(f"Created {len(bundles)} horizon bundles from {len(targets)} targets")

    # Rank bundles
    ranked = rank_bundles(bundles, y_dict, target_scores)

    # Filter by diversity
    filtered = filter_bundles_by_diversity(ranked, min_diversity)

    if len(filtered) < len(ranked):
        logger.info(
            f"Filtered {len(ranked) - len(filtered)} bundles with diversity < {min_diversity}"
        )

    # Return top N
    result = filtered[:top_n]

    for i, bundle in enumerate(result):
        logger.info(
            f"  Bundle {i+1}: {bundle.base_name} ({bundle.n_horizons} horizons, "
            f"diversity={bundle.diversity_score:.3f})"
        )

    return result
```

### Tests: `tests/test_horizon_bundle.py`

```python
import pytest
import numpy as np
from TRAINING.common.horizon_bundle import (
    HorizonBundle,
    parse_horizon_from_target,
    group_targets_by_base,
    create_bundles_from_targets,
    compute_bundle_diversity,
)


class TestParseHorizon:
    def test_standard_format(self):
        assert parse_horizon_from_target("fwd_ret_5m") == ("fwd_ret", 5)
        assert parse_horizon_from_target("fwd_ret_60m") == ("fwd_ret", 60)

    def test_with_threshold(self):
        assert parse_horizon_from_target("will_peak_5m_0.8") == ("will_peak", 5)

    def test_no_horizon(self):
        assert parse_horizon_from_target("custom_target") == ("custom_target", None)


class TestGroupTargets:
    def test_groups_by_base(self):
        targets = ["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m", "will_peak_5m"]
        groups = group_targets_by_base(targets)

        assert "fwd_ret" in groups
        assert len(groups["fwd_ret"]) == 3
        assert "will_peak" in groups


class TestBundleCreation:
    def test_creates_bundles(self):
        targets = ["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m"]
        bundles = create_bundles_from_targets(targets, min_horizons=2)

        assert len(bundles) == 1
        assert bundles[0].base_name == "fwd_ret"
        assert bundles[0].n_horizons == 3


class TestDiversity:
    def test_high_diversity(self):
        """Uncorrelated targets should have high diversity."""
        bundle = HorizonBundle(
            base_name="test",
            horizons=[5, 15],
            targets=["t_5m", "t_15m"]
        )

        # Uncorrelated random data
        np.random.seed(42)
        y_dict = {
            "t_5m": np.random.randn(100),
            "t_15m": np.random.randn(100),
        }

        diversity = compute_bundle_diversity(bundle, y_dict)
        assert diversity > 0.7

    def test_low_diversity(self):
        """Highly correlated targets should have low diversity."""
        bundle = HorizonBundle(
            base_name="test",
            horizons=[5, 15],
            targets=["t_5m", "t_15m"]
        )

        # Highly correlated data
        np.random.seed(42)
        base = np.random.randn(100)
        y_dict = {
            "t_5m": base,
            "t_15m": base + np.random.randn(100) * 0.1,  # 95%+ correlation
        }

        diversity = compute_bundle_diversity(bundle, y_dict)
        assert diversity < 0.2
```

---

## Subphase 8.2: MultiHorizonTrainer

### Goal
Create a trainer that handles multiple horizons with shared encoder + per-horizon heads.

### New File: `TRAINING/model_fun/multi_horizon_trainer.py`

```python
"""
Multi-Horizon Trainer

Trains a single model with shared encoder and per-horizon prediction heads.
Supports both TensorFlow and PyTorch backends.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class MultiHorizonTrainer:
    """
    Trainer for multi-horizon prediction.

    Architecture:
        Input → [Shared Encoder] → Split
                                   ├── [Head 5m]  → pred_5m
                                   ├── [Head 15m] → pred_15m
                                   └── [Head 60m] → pred_60m

    Config:
        multi_horizon:
          shared_layers: [256, 128]
          head_layers: [64]
          dropout: 0.2
          batch_norm: true
          loss_weights:
            fwd_ret_5m: 1.0
            fwd_ret_15m: 1.0
            fwd_ret_60m: 0.5
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.horizons = []
        self.targets = []
        self._backend = self.config.get('backend', 'tensorflow')

    def train(
        self,
        X_tr: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        X_va: Optional[np.ndarray] = None,
        y_va_dict: Optional[Dict[str, np.ndarray]] = None,
        horizons: Optional[List[int]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train multi-horizon model.

        Args:
            X_tr: Training features (N, D)
            y_dict: Dict of target_name → target_values (N,)
            X_va: Validation features
            y_va_dict: Validation targets
            horizons: List of horizons in minutes
            loss_weights: Per-target loss weights

        Returns:
            Training results dict
        """
        self.targets = list(y_dict.keys())
        self.horizons = horizons or self._infer_horizons(self.targets)

        # Stack targets for multi-output
        y_tr = np.column_stack([y_dict[t] for t in self.targets])
        y_va = None
        if y_va_dict:
            y_va = np.column_stack([y_va_dict[t] for t in self.targets])

        # Build model
        input_dim = X_tr.shape[1]
        self.model = self._build_model(input_dim, len(self.targets), loss_weights)

        # Train
        if self._backend == 'tensorflow':
            return self._train_tf(X_tr, y_tr, X_va, y_va, **kwargs)
        else:
            return self._train_torch(X_tr, y_tr, X_va, y_va, **kwargs)

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict for all horizons.

        Returns:
            Dict of target_name → predictions
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        preds = self.model.predict(X)

        # Split multi-output back to dict
        result = {}
        for i, target in enumerate(self.targets):
            result[target] = preds[:, i]

        return result

    def _build_model(
        self,
        input_dim: int,
        n_outputs: int,
        loss_weights: Optional[Dict[str, float]]
    ):
        """Build multi-head model."""
        if self._backend == 'tensorflow':
            return self._build_tf_model(input_dim, n_outputs, loss_weights)
        else:
            return self._build_torch_model(input_dim, n_outputs, loss_weights)

    def _build_tf_model(self, input_dim: int, n_outputs: int, loss_weights):
        """Build TensorFlow multi-head model."""
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        shared_layers = self.config.get('shared_layers', [256, 128])
        head_layers = self.config.get('head_layers', [64])
        dropout = self.config.get('dropout', 0.2)
        use_bn = self.config.get('batch_norm', True)

        # Input
        inputs = layers.Input(shape=(input_dim,))
        x = inputs

        # Shared encoder
        for units in shared_layers:
            x = layers.Dense(units, activation='relu')(x)
            if use_bn:
                x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)

        shared_output = x

        # Per-horizon heads
        outputs = []
        for i, target in enumerate(self.targets):
            head = shared_output
            for units in head_layers:
                head = layers.Dense(units, activation='relu', name=f'head_{target}_{units}')(head)
                if use_bn:
                    head = layers.BatchNormalization()(head)
                head = layers.Dropout(dropout)(head)

            output = layers.Dense(1, activation='linear', name=f'output_{target}')(head)
            outputs.append(output)

        # Concatenate outputs
        if len(outputs) > 1:
            combined_output = layers.Concatenate()(outputs)
        else:
            combined_output = outputs[0]

        model = Model(inputs=inputs, outputs=combined_output)

        # Compile with weighted loss
        weights = loss_weights or {t: 1.0 for t in self.targets}
        weight_list = [weights.get(t, 1.0) for t in self.targets]

        # Custom loss that applies weights
        def weighted_mse(y_true, y_pred):
            losses = []
            for i in range(n_outputs):
                mse = tf.reduce_mean(tf.square(y_true[:, i] - y_pred[:, i]))
                losses.append(weight_list[i] * mse)
            return tf.reduce_sum(losses)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get('lr', 0.001)),
            loss=weighted_mse
        )

        return model

    def _train_tf(self, X_tr, y_tr, X_va, y_va, **kwargs) -> Dict[str, Any]:
        """Train TensorFlow model."""
        import tensorflow as tf

        epochs = kwargs.get('epochs', self.config.get('epochs', 100))
        batch_size = kwargs.get('batch_size', self.config.get('batch_size', 256))

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_va is not None else 'loss',
                patience=self.config.get('patience', 10),
                restore_best_weights=True
            )
        ]

        validation_data = (X_va, y_va) if X_va is not None else None

        history = self.model.fit(
            X_tr, y_tr,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=kwargs.get('verbose', 0)
        )

        return {
            'final_loss': history.history['loss'][-1],
            'epochs_trained': len(history.history['loss']),
            'targets': self.targets,
            'horizons': self.horizons
        }

    def _infer_horizons(self, targets: List[str]) -> List[int]:
        """Infer horizons from target names."""
        from TRAINING.common.horizon_bundle import parse_horizon_from_target

        horizons = []
        for target in targets:
            _, horizon = parse_horizon_from_target(target)
            if horizon:
                horizons.append(horizon)
            else:
                horizons.append(0)  # Unknown

        return horizons

    def get_model_metadata(self) -> Dict[str, Any]:
        """Get metadata for model serialization."""
        return {
            'model_type': 'multi_horizon',
            'backend': self._backend,
            'targets': self.targets,
            'horizons': self.horizons,
            'config': self.config
        }
```

---

## Subphase 8.3: Training Loop Integration

### Goal
Integrate multi-horizon training into the main training loop.

### Modified: `TRAINING/training_strategies/execution/training.py`

Add new strategy handling:

```python
# In train_models_for_interval_comprehensive()

if strategy == 'multi_horizon_bundle':
    from TRAINING.orchestration.horizon_ranker import select_top_bundles
    from TRAINING.model_fun.multi_horizon_trainer import MultiHorizonTrainer

    # Get bundles
    bundles = select_top_bundles(
        targets=targets,
        y_dict=y_dict,
        target_scores=target_predictability_scores,
        top_n=bundle_config.get('top_n', 3),
        min_diversity=bundle_config.get('min_diversity', 0.3)
    )

    for bundle in bundles:
        # Prepare data for bundle
        bundle_y = {t: y_dict[t] for t in bundle.targets}

        # Train multi-horizon model
        trainer = MultiHorizonTrainer(config=bundle_config)
        result = trainer.train(
            X_tr, bundle_y,
            X_va=X_va, y_va_dict=bundle_y_va,
            horizons=bundle.horizons,
            loss_weights=bundle.loss_weights,
            **kwargs
        )

        # Save model and metadata
        results[f'bundle_{bundle.base_name}'] = {
            'bundle': bundle,
            'trainer': trainer,
            'result': result
        }
```

---

## Subphase 8.4: Config + Tests

### Config: `CONFIG/pipeline/training/intelligent.yaml`

Add new strategy options:

```yaml
intelligent_training:
  # Existing options...

  # Multi-horizon bundle strategy
  multi_horizon_bundle:
    enabled: false

    # Bundle selection
    auto_discover: true
    min_horizons: 2
    max_horizons: 5
    min_diversity: 0.3
    top_n_bundles: 3

    # Model architecture
    shared_layers: [256, 128]
    head_layers: [64]
    dropout: 0.2
    batch_norm: true
    backend: tensorflow  # or pytorch

    # Training
    epochs: 100
    batch_size: 256
    patience: 10
    lr: 0.001

    # Loss weighting
    weight_by_horizon: false  # If true, shorter horizons get higher weight
    weight_decay_halflife: 30  # minutes
```

### Example Experiment Config

```yaml
# CONFIG/experiments/multi_horizon_example.yaml

experiment:
  name: multi_horizon_fwd_ret
  description: Multi-horizon training for forward returns

intelligent_training:
  auto_targets: true
  exclude_target_patterns: ["will_peak", "will_valley", "mdd", "mfe"]

  strategy: multi_horizon_bundle

  multi_horizon_bundle:
    enabled: true
    auto_discover: true
    min_horizons: 2
    max_horizons: 4
    min_diversity: 0.3
    top_n_bundles: 2

    shared_layers: [256, 128]
    head_layers: [64, 32]
    dropout: 0.3

    weight_by_horizon: true
    weight_decay_halflife: 30
```

---

## Validation Checklist

### Subphase 8.1
- [ ] `parse_horizon_from_target()` handles all target formats
- [ ] `create_bundles_from_targets()` groups correctly
- [ ] `compute_bundle_diversity()` returns sensible values
- [ ] `rank_bundles()` orders by quality

### Subphase 8.2
- [ ] TensorFlow multi-head model builds correctly
- [ ] Loss weights applied per horizon
- [ ] `predict()` returns dict with all horizons
- [ ] Early stopping works

### Subphase 8.3
- [ ] `multi_horizon_bundle` strategy recognized
- [ ] Bundles trained in deterministic order
- [ ] Results saved with proper metadata

### Subphase 8.4
- [ ] Config loads without error
- [ ] Example experiment runs end-to-end
- [ ] All tests pass

---

## Test Commands

```bash
# Unit tests
pytest tests/test_horizon_bundle.py -v

# Integration test
pytest tests/test_multi_horizon_trainer.py -v

# E2E test (requires data)
python -m TRAINING.orchestration.intelligent_trainer \
    --experiment-config CONFIG/experiments/multi_horizon_example.yaml \
    --output-dir test_multi_horizon
```

---

## Next Steps After Phase 8

1. **Phase 9**: Cross-horizon ensemble (blend predictions from Phase 8)
2. **Metrics collection**: Track per-horizon vs combined metrics
3. **Hyperparameter tuning**: Optimize shared_layers, head_layers, loss_weights
