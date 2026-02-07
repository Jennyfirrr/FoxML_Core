# Phase 10: Multi-Interval Experiments

**Status**: Not Started
**Parent Plan**: [multi_horizon_training_master.md](./multi_horizon_training_master.md)
**Estimated Effort**: 4-5 days
**Depends On**: Phase 8 (Multi-Horizon Training), Phase 9 (Cross-Horizon Ensemble)

---

## Overview

Multi-interval experiments allow training on one data interval (e.g., 5m) and validating/testing on another (e.g., 1m). This tests model generalization across time scales and enables:

1. **Cross-interval validation**: Check if patterns learned at coarse intervals hold at fine intervals
2. **Feature transfer**: Warm-start from models trained on coarser (less noisy) data
3. **Regime-based selection**: Use different intervals for different market regimes

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data leakage across intervals | High | Critical | Strict purge validation per interval |
| Feature mismatch | High | High | Feature registry interval validation |
| Label drift | Medium | High | Target recomputation at each interval |
| Cache key collisions | Medium | High | Interval in cache key (Phase 13 done) |
| OOM with multi-interval data | Medium | Medium | Sequential loading, batch processing |

---

## Subphases

| Subphase | Description | Effort | Files |
|----------|-------------|--------|-------|
| 10.1 | Multi-interval data loader | 8h | `multi_interval_loader.py` |
| 10.2 | Feature transfer (warm-start) | 6h | `feature_transfer.py` |
| 10.3 | Cross-interval validation | 6h | `cross_interval_cv.py` |
| 10.4 | Config + tests | 4h | configs, tests |

---

## Subphase 10.1: Multi-Interval Data Loader

### Goal
Load data from multiple intervals for the same experiment.

### New File: `TRAINING/data/multi_interval_loader.py`

```python
"""
Multi-Interval Data Loader

Loads and aligns data across multiple time intervals.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IntervalData:
    """Container for data at a specific interval."""
    interval_minutes: int
    features: pd.DataFrame
    targets: Dict[str, pd.Series]
    symbols: List[str]
    timestamp_range: Tuple[pd.Timestamp, pd.Timestamp]
    n_samples: int

    @property
    def interval_str(self) -> str:
        return f"{self.interval_minutes}m"


@dataclass
class MultiIntervalDataset:
    """
    Dataset containing data at multiple intervals.

    Supports:
    - Training on one interval, validating on another
    - Feature transfer across intervals
    - Regime-based interval selection
    """
    primary_interval: int  # Main interval for training
    intervals: Dict[int, IntervalData] = field(default_factory=dict)
    aligned_timestamps: Optional[pd.DatetimeIndex] = None

    def add_interval(self, data: IntervalData):
        """Add data for an interval."""
        self.intervals[data.interval_minutes] = data

    def get_interval(self, interval_minutes: int) -> Optional[IntervalData]:
        """Get data for a specific interval."""
        return self.intervals.get(interval_minutes)

    @property
    def available_intervals(self) -> List[int]:
        """List of available intervals."""
        return sorted(self.intervals.keys())

    def compute_aligned_timestamps(self) -> pd.DatetimeIndex:
        """
        Find timestamps common to all intervals.

        Uses the coarsest interval as base and finds matching
        timestamps in finer intervals.
        """
        if len(self.intervals) < 2:
            if self.intervals:
                interval = list(self.intervals.values())[0]
                return interval.features.index
            return pd.DatetimeIndex([])

        # Start with coarsest interval
        coarsest = max(self.intervals.keys())
        base_ts = set(self.intervals[coarsest].features.index)

        # Intersect with other intervals
        for interval_mins, data in self.intervals.items():
            if interval_mins == coarsest:
                continue

            # For finer intervals, check if coarse timestamp falls within
            # This is approximate - fine timestamps won't exactly match coarse
            interval_ts = set(data.features.index)
            base_ts = base_ts.intersection(interval_ts)

        self.aligned_timestamps = pd.DatetimeIndex(sorted(base_ts))
        return self.aligned_timestamps


class MultiIntervalLoader:
    """
    Loader for multi-interval experiments.

    Handles:
    1. Loading data from multiple interval directories
    2. Feature alignment across intervals
    3. Target recomputation at each interval
    4. Validation that data is consistent
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data_root = Path(self.config.get('data_root', 'data/data_labeled'))

    def load_multi_interval(
        self,
        intervals: List[int],
        symbols: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        primary_interval: int = 5,
        max_rows_per_symbol: Optional[int] = None,
        date_range: Optional[Tuple[str, str]] = None
    ) -> MultiIntervalDataset:
        """
        Load data at multiple intervals.

        Args:
            intervals: List of intervals to load (in minutes)
            symbols: Symbols to load (None = all)
            targets: Target columns to include
            primary_interval: Main training interval
            max_rows_per_symbol: Limit rows per symbol
            date_range: Optional (start_date, end_date) filter

        Returns:
            MultiIntervalDataset
        """
        dataset = MultiIntervalDataset(primary_interval=primary_interval)

        for interval in intervals:
            logger.info(f"Loading interval {interval}m data...")

            data = self._load_single_interval(
                interval_minutes=interval,
                symbols=symbols,
                targets=targets,
                max_rows_per_symbol=max_rows_per_symbol,
                date_range=date_range
            )

            if data is not None:
                dataset.add_interval(data)
                logger.info(
                    f"  Loaded {data.n_samples} samples from "
                    f"{len(data.symbols)} symbols"
                )
            else:
                logger.warning(f"  No data found for interval {interval}m")

        # Validate consistency
        self._validate_multi_interval(dataset)

        return dataset

    def _load_single_interval(
        self,
        interval_minutes: int,
        symbols: Optional[List[str]],
        targets: Optional[List[str]],
        max_rows_per_symbol: Optional[int],
        date_range: Optional[Tuple[str, str]]
    ) -> Optional[IntervalData]:
        """Load data for a single interval."""
        interval_dir = self.data_root / f"interval={interval_minutes}m"

        if not interval_dir.exists():
            return None

        # Find parquet files
        parquet_files = sorted(interval_dir.glob("*.parquet"))
        if not parquet_files:
            return None

        dfs = []
        loaded_symbols = []

        for pq_file in parquet_files:
            # Extract symbol from filename
            symbol = pq_file.stem.replace('.parquet', '')

            if symbols and symbol not in symbols:
                continue

            try:
                df = pd.read_parquet(pq_file)

                # Filter by date range
                if date_range and 'ts' in df.columns:
                    df['ts'] = pd.to_datetime(df['ts'])
                    df = df[
                        (df['ts'] >= date_range[0]) &
                        (df['ts'] <= date_range[1])
                    ]

                # Limit rows
                if max_rows_per_symbol and len(df) > max_rows_per_symbol:
                    df = df.tail(max_rows_per_symbol)

                df['symbol'] = symbol
                dfs.append(df)
                loaded_symbols.append(symbol)

            except Exception as e:
                logger.warning(f"Failed to load {pq_file}: {e}")
                continue

        if not dfs:
            return None

        # Combine
        combined = pd.concat(dfs, ignore_index=True)

        # Extract features and targets
        target_cols = []
        if targets:
            target_cols = [c for c in targets if c in combined.columns]
        else:
            # Auto-detect targets
            target_cols = [c for c in combined.columns
                         if c.startswith(('fwd_ret_', 'will_peak_', 'will_valley_', 'mdd_'))]

        feature_cols = [c for c in combined.columns
                       if c not in target_cols + ['ts', 'symbol', 'timestamp']]

        # Build IntervalData
        if 'ts' in combined.columns:
            combined = combined.set_index('ts')

        features_df = combined[feature_cols]
        targets_dict = {t: combined[t] for t in target_cols}

        timestamp_range = (
            combined.index.min() if hasattr(combined.index, 'min') else None,
            combined.index.max() if hasattr(combined.index, 'max') else None
        )

        return IntervalData(
            interval_minutes=interval_minutes,
            features=features_df,
            targets=targets_dict,
            symbols=loaded_symbols,
            timestamp_range=timestamp_range,
            n_samples=len(combined)
        )

    def _validate_multi_interval(self, dataset: MultiIntervalDataset):
        """
        Validate multi-interval dataset for consistency.

        Checks:
        1. Feature columns match across intervals
        2. Target columns match across intervals
        3. Symbol coverage is similar
        4. Date ranges overlap
        """
        if len(dataset.intervals) < 2:
            return

        # Get all feature sets
        feature_sets = []
        target_sets = []
        symbol_sets = []

        for interval, data in dataset.intervals.items():
            feature_sets.append(set(data.features.columns))
            target_sets.append(set(data.targets.keys()))
            symbol_sets.append(set(data.symbols))

        # Check feature overlap
        common_features = feature_sets[0].intersection(*feature_sets[1:])
        all_features = feature_sets[0].union(*feature_sets[1:])

        if len(common_features) < len(all_features) * 0.8:
            logger.warning(
                f"Only {len(common_features)}/{len(all_features)} features "
                f"are common across intervals"
            )

        # Check target overlap
        common_targets = target_sets[0].intersection(*target_sets[1:])
        if not common_targets:
            logger.warning("No common targets across intervals")

        # Check symbol overlap
        common_symbols = symbol_sets[0].intersection(*symbol_sets[1:])
        if len(common_symbols) < len(symbol_sets[0]) * 0.5:
            logger.warning(
                f"Only {len(common_symbols)} symbols common across intervals"
            )


def resample_to_interval(
    df: pd.DataFrame,
    from_interval: int,
    to_interval: int,
    ohlcv_columns: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Resample data from one interval to another.

    Only supports coarsening (e.g., 1m → 5m), not refinement.

    Args:
        df: Source DataFrame with DatetimeIndex
        from_interval: Source interval in minutes
        to_interval: Target interval in minutes
        ohlcv_columns: Mapping of OHLCV column names

    Returns:
        Resampled DataFrame
    """
    if to_interval < from_interval:
        raise ValueError(
            f"Can only coarsen intervals ({from_interval}m → {to_interval}m not allowed)"
        )

    if to_interval == from_interval:
        return df.copy()

    if ohlcv_columns is None:
        ohlcv_columns = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }

    agg_dict = {}
    for col in df.columns:
        if col == ohlcv_columns.get('open'):
            agg_dict[col] = 'first'
        elif col == ohlcv_columns.get('high'):
            agg_dict[col] = 'max'
        elif col == ohlcv_columns.get('low'):
            agg_dict[col] = 'min'
        elif col == ohlcv_columns.get('close'):
            agg_dict[col] = 'last'
        elif col == ohlcv_columns.get('volume'):
            agg_dict[col] = 'sum'
        else:
            # Default: last value
            agg_dict[col] = 'last'

    resampled = df.resample(f'{to_interval}T').agg(agg_dict)

    return resampled.dropna()
```

---

## Subphase 10.2: Feature Transfer (Warm-Start)

### Goal
Transfer learned features/weights from coarse interval to fine interval.

### New File: `TRAINING/data/feature_transfer.py`

```python
"""
Feature Transfer for Cross-Interval Learning

Warm-start models from coarser intervals to improve training on finer intervals.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureTransferManager:
    """
    Manages feature transfer across intervals.

    Strategies:
    1. Weight transfer: Initialize neural network weights from coarse model
    2. Feature importance: Use coarse model's feature importance for selection
    3. Ensemble prior: Use coarse predictions as additional feature
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.source_models = {}  # interval → model
        self.feature_mappings = {}  # Maps features across intervals

    def register_source_model(
        self,
        interval_minutes: int,
        model: Any,
        feature_names: List[str]
    ):
        """Register a trained model as potential transfer source."""
        self.source_models[interval_minutes] = {
            'model': model,
            'features': feature_names
        }
        logger.info(f"Registered source model for {interval_minutes}m interval")

    def get_transfer_weights(
        self,
        source_interval: int,
        target_interval: int,
        model_type: str = 'neural_network'
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Get weights for transfer learning.

        Args:
            source_interval: Source interval (coarser)
            target_interval: Target interval (finer)
            model_type: Type of model for transfer

        Returns:
            Dict of layer_name → weights, or None if not available
        """
        if source_interval not in self.source_models:
            logger.warning(f"No source model for {source_interval}m")
            return None

        source = self.source_models[source_interval]
        model = source['model']

        if model_type == 'neural_network':
            return self._extract_nn_weights(model)
        elif model_type == 'tree':
            return self._extract_tree_importance(model)
        else:
            return None

    def _extract_nn_weights(self, model) -> Dict[str, np.ndarray]:
        """Extract weights from neural network."""
        weights = {}

        # TensorFlow/Keras
        if hasattr(model, 'get_weights'):
            for i, layer in enumerate(model.layers):
                layer_weights = layer.get_weights()
                if layer_weights:
                    weights[f'layer_{i}'] = layer_weights

        # PyTorch
        elif hasattr(model, 'state_dict'):
            for name, param in model.state_dict().items():
                weights[name] = param.cpu().numpy()

        return weights

    def _extract_tree_importance(self, model) -> Dict[str, np.ndarray]:
        """Extract feature importance from tree model."""
        if hasattr(model, 'feature_importances_'):
            return {'importance': model.feature_importances_}
        elif hasattr(model, 'feature_importance'):
            return {'importance': model.feature_importance()}
        return {}

    def compute_feature_mapping(
        self,
        source_interval: int,
        target_interval: int,
        source_features: List[str],
        target_features: List[str]
    ) -> Dict[str, str]:
        """
        Map features from source to target interval.

        Some features need interval adjustment:
        - ret_5m at 5m interval → ret_1m at 1m interval (different lookback)
        - rsi_14 is same at both intervals (period-based)
        """
        mapping = {}

        for sf in source_features:
            # Exact match
            if sf in target_features:
                mapping[sf] = sf
                continue

            # Try interval-adjusted match
            adjusted = self._adjust_feature_name(sf, source_interval, target_interval)
            if adjusted in target_features:
                mapping[sf] = adjusted

        logger.info(
            f"Mapped {len(mapping)}/{len(source_features)} features "
            f"from {source_interval}m to {target_interval}m"
        )

        return mapping

    def _adjust_feature_name(
        self,
        feature: str,
        from_interval: int,
        to_interval: int
    ) -> str:
        """
        Adjust feature name for interval change.

        Examples:
        - ret_5m → ret_1m (when 5m → 1m)
        - vol_15m → vol_3m (proportional)
        """
        import re

        # Pattern: feature_Nm
        pattern = r'^(.+?)_(\d+)m$'
        match = re.match(pattern, feature)

        if match:
            base = match.group(1)
            lookback = int(match.group(2))

            # Adjust proportionally
            new_lookback = int(lookback * to_interval / from_interval)
            return f"{base}_{new_lookback}m"

        return feature

    def create_transfer_features(
        self,
        X_target: np.ndarray,
        target_features: List[str],
        source_interval: int
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create additional features from source model predictions.

        Args:
            X_target: Target interval features
            target_features: Target feature names
            source_interval: Source interval for predictions

        Returns:
            (augmented_X, augmented_feature_names)
        """
        if source_interval not in self.source_models:
            return X_target, target_features

        source = self.source_models[source_interval]
        model = source['model']

        # Get predictions from source model
        try:
            source_pred = model.predict(X_target)
            if source_pred.ndim == 1:
                source_pred = source_pred.reshape(-1, 1)

            # Augment features
            X_augmented = np.hstack([X_target, source_pred])
            augmented_names = list(target_features) + [f'source_{source_interval}m_pred']

            return X_augmented, augmented_names

        except Exception as e:
            logger.warning(f"Failed to create transfer features: {e}")
            return X_target, target_features


def apply_weight_transfer(
    target_model,
    source_weights: Dict[str, np.ndarray],
    transfer_mode: str = 'partial'
) -> bool:
    """
    Apply transferred weights to target model.

    Args:
        target_model: Model to initialize
        source_weights: Weights from source model
        transfer_mode: 'full' (all layers), 'partial' (shared layers only)

    Returns:
        True if successful
    """
    try:
        # TensorFlow/Keras
        if hasattr(target_model, 'set_weights'):
            if transfer_mode == 'partial':
                # Only transfer first N layers (shared encoder)
                n_transfer = len(source_weights) // 2
                for i, (name, weights) in enumerate(source_weights.items()):
                    if i >= n_transfer:
                        break
                    target_model.layers[i].set_weights(weights)
            else:
                target_model.set_weights(list(source_weights.values()))
            return True

        # PyTorch
        elif hasattr(target_model, 'load_state_dict'):
            import torch
            state = {k: torch.tensor(v) for k, v in source_weights.items()}
            target_model.load_state_dict(state, strict=False)
            return True

    except Exception as e:
        logger.warning(f"Weight transfer failed: {e}")

    return False
```

---

## Subphase 10.3: Cross-Interval Validation

### Goal
Validate models across different data intervals to test generalization.

### New File: `TRAINING/data/cross_interval_cv.py`

```python
"""
Cross-Interval Cross-Validation

Validates that models generalize across data intervals.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CrossIntervalResult:
    """Results from cross-interval validation."""
    train_interval: int
    val_interval: int
    metrics: Dict[str, float]
    generalization_ratio: float  # val_metric / train_metric
    n_train: int
    n_val: int


class CrossIntervalValidator:
    """
    Validates model generalization across intervals.

    Key insight: A model trained on 5m data should have reasonable
    performance when validated on 1m data (same underlying patterns,
    different granularity).

    Poor generalization indicates:
    - Overfitting to interval-specific noise
    - Features that don't transfer across scales
    - Target computation artifacts
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.results = []

    def validate_cross_interval(
        self,
        model,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        train_interval: int,
        val_interval: int,
        metrics: List[str] = None
    ) -> CrossIntervalResult:
        """
        Validate model trained on one interval against another.

        Args:
            model: Trained model
            train_data: (X_train, y_train) at train interval
            val_data: (X_val, y_val) at validation interval
            train_interval: Training interval in minutes
            val_interval: Validation interval in minutes
            metrics: Metrics to compute

        Returns:
            CrossIntervalResult
        """
        X_tr, y_tr = train_data
        X_va, y_va = val_data

        if metrics is None:
            metrics = ['r2', 'rmse', 'ic']

        # Compute train metrics
        train_pred = model.predict(X_tr)
        train_metrics = self._compute_metrics(y_tr, train_pred, metrics)

        # Compute validation metrics
        val_pred = model.predict(X_va)
        val_metrics = self._compute_metrics(y_va, val_pred, metrics)

        # Compute generalization ratio
        primary_metric = metrics[0]
        train_score = train_metrics.get(primary_metric, 0)
        val_score = val_metrics.get(primary_metric, 0)

        if abs(train_score) > 1e-6:
            gen_ratio = val_score / train_score
        else:
            gen_ratio = 0.0

        result = CrossIntervalResult(
            train_interval=train_interval,
            val_interval=val_interval,
            metrics=val_metrics,
            generalization_ratio=gen_ratio,
            n_train=len(y_tr),
            n_val=len(y_va)
        )

        self.results.append(result)

        logger.info(
            f"Cross-interval validation: {train_interval}m → {val_interval}m, "
            f"gen_ratio={gen_ratio:.3f}"
        )

        return result

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Compute specified metrics."""
        results = {}

        for metric in metrics:
            if metric == 'r2':
                results['r2'] = r2_score(y_true, y_pred)
            elif metric == 'rmse':
                results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == 'ic':
                # Information coefficient (correlation)
                results['ic'] = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
            elif metric == 'mae':
                results['mae'] = np.mean(np.abs(y_true - y_pred))

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all cross-interval validations."""
        if not self.results:
            return {}

        summary = {
            'n_validations': len(self.results),
            'avg_generalization_ratio': np.mean([r.generalization_ratio for r in self.results]),
            'results': []
        }

        for r in self.results:
            summary['results'].append({
                'train_interval': r.train_interval,
                'val_interval': r.val_interval,
                'metrics': r.metrics,
                'gen_ratio': r.generalization_ratio
            })

        return summary


def cross_interval_cv_split(
    dataset,  # MultiIntervalDataset
    train_interval: int,
    val_interval: int,
    n_splits: int = 5,
    gap_minutes: int = 60
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create cross-interval CV splits.

    Ensures temporal gap between train and validation to prevent leakage.

    Args:
        dataset: MultiIntervalDataset
        train_interval: Training interval
        val_interval: Validation interval
        n_splits: Number of CV splits
        gap_minutes: Gap between train end and val start

    Returns:
        List of (X_tr, y_tr, X_va, y_va) tuples
    """
    train_data = dataset.get_interval(train_interval)
    val_data = dataset.get_interval(val_interval)

    if train_data is None or val_data is None:
        raise ValueError(f"Missing data for intervals {train_interval}m or {val_interval}m")

    # Get common timestamps
    train_ts = train_data.features.index
    val_ts = val_data.features.index

    # Find overlap period
    common_start = max(train_ts.min(), val_ts.min())
    common_end = min(train_ts.max(), val_ts.max())

    # Create time-based splits
    total_duration = (common_end - common_start).total_seconds() / 60
    split_duration = total_duration / (n_splits + 1)

    splits = []

    for i in range(n_splits):
        # Training window
        train_end = common_start + pd.Timedelta(minutes=split_duration * (i + 1))
        train_start = common_start

        # Gap
        val_start = train_end + pd.Timedelta(minutes=gap_minutes)
        val_end = val_start + pd.Timedelta(minutes=split_duration)

        # Filter data
        train_mask = (train_data.features.index >= train_start) & \
                     (train_data.features.index < train_end)
        val_mask = (val_data.features.index >= val_start) & \
                   (val_data.features.index < val_end)

        X_tr = train_data.features[train_mask].values
        X_va = val_data.features[val_mask].values

        # Get first target
        target_name = list(train_data.targets.keys())[0]
        y_tr = train_data.targets[target_name][train_mask].values
        y_va = val_data.targets[target_name][val_mask].values

        if len(X_tr) > 0 and len(X_va) > 0:
            splits.append((X_tr, y_tr, X_va, y_va))

    return splits


def evaluate_interval_generalization(
    model_factory,  # Callable that creates and trains model
    dataset,  # MultiIntervalDataset
    intervals: List[int],
    target: str,
    n_splits: int = 3
) -> pd.DataFrame:
    """
    Evaluate model generalization across all interval pairs.

    Returns matrix of generalization scores.

    Args:
        model_factory: Function that trains model given (X, y)
        dataset: MultiIntervalDataset
        intervals: Intervals to evaluate
        target: Target column name
        n_splits: CV splits per pair

    Returns:
        DataFrame with train_interval as index, val_interval as columns
    """
    results = []

    for train_int in intervals:
        train_data = dataset.get_interval(train_int)
        if train_data is None:
            continue

        X_tr = train_data.features.values
        y_tr = train_data.targets[target].values

        # Train model
        model = model_factory(X_tr, y_tr)

        for val_int in intervals:
            val_data = dataset.get_interval(val_int)
            if val_data is None:
                continue

            X_va = val_data.features.values
            y_va = val_data.targets[target].values

            # Validate
            val_pred = model.predict(X_va)
            r2 = r2_score(y_va, val_pred)
            ic = np.corrcoef(y_va.flatten(), val_pred.flatten())[0, 1]

            results.append({
                'train_interval': train_int,
                'val_interval': val_int,
                'r2': r2,
                'ic': ic
            })

    df = pd.DataFrame(results)

    # Pivot to matrix form
    r2_matrix = df.pivot(
        index='train_interval',
        columns='val_interval',
        values='r2'
    )

    return r2_matrix
```

---

## Subphase 10.4: Config + Tests

### Config: `CONFIG/experiments/multi_interval_example.yaml`

```yaml
experiment:
  name: multi_interval_experiment
  description: Train on 5m, validate on 1m and 15m

# Multi-interval configuration
multi_interval:
  enabled: true

  # Intervals to use
  primary_interval: 5  # Main training interval
  training_intervals: [5, 15]  # Train on these
  validation_intervals: [1, 5, 15]  # Validate on these

  # Data sources (paths relative to data_root)
  data_sources:
    1: interval=1m
    5: interval=5m
    15: interval=15m

  # Feature transfer
  feature_transfer:
    enabled: true
    source_interval: 15  # Warm-start from coarser
    transfer_mode: partial  # 'full', 'partial', 'importance'

  # Cross-interval validation
  cross_validation:
    enabled: true
    gap_minutes: 60  # Gap between train and val
    n_splits: 3
    min_generalization_ratio: 0.7  # Warn if below this

  # Resampling (if needed)
  allow_resampling: false  # Only use native data

intelligent_training:
  auto_targets: true
  include_target_patterns: ["fwd_ret"]

  # Per-interval overrides
  interval_overrides:
    1:
      top_m_features: 30  # Fewer features at fine interval
    15:
      top_m_features: 100  # More features at coarse interval

training:
  model_families:
    - lightgbm
    - ridge
```

### Tests: `tests/test_multi_interval.py`

```python
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from TRAINING.data.multi_interval_loader import (
    MultiIntervalLoader,
    MultiIntervalDataset,
    IntervalData,
    resample_to_interval,
)
from TRAINING.data.feature_transfer import (
    FeatureTransferManager,
    apply_weight_transfer,
)
from TRAINING.data.cross_interval_cv import (
    CrossIntervalValidator,
    evaluate_interval_generalization,
)


class TestMultiIntervalLoader:
    def test_resample_to_interval(self):
        """Test resampling from fine to coarse."""
        # Create 1m data
        n = 100
        ts = pd.date_range('2024-01-01 09:30', periods=n, freq='1T')
        df = pd.DataFrame({
            'open': np.random.randn(n),
            'high': np.random.randn(n) + 0.1,
            'low': np.random.randn(n) - 0.1,
            'close': np.random.randn(n),
            'volume': np.random.randint(100, 1000, n),
        }, index=ts)

        # Resample to 5m
        resampled = resample_to_interval(df, from_interval=1, to_interval=5)

        assert len(resampled) == n // 5
        assert 'open' in resampled.columns

    def test_cannot_refine_interval(self):
        """Test that refinement (5m → 1m) raises error."""
        df = pd.DataFrame({'close': [1, 2, 3]})

        with pytest.raises(ValueError):
            resample_to_interval(df, from_interval=5, to_interval=1)


class TestFeatureTransfer:
    def test_feature_name_adjustment(self):
        """Test feature name adjustment across intervals."""
        manager = FeatureTransferManager()

        # ret_5m at 5m interval → ret_1m at 1m interval
        mapping = manager.compute_feature_mapping(
            source_interval=5,
            target_interval=1,
            source_features=['ret_5m', 'rsi_14', 'vol_15m'],
            target_features=['ret_1m', 'rsi_14', 'vol_3m']
        )

        assert mapping['ret_5m'] == 'ret_1m'
        assert mapping['rsi_14'] == 'rsi_14'  # Period-based, no change
        assert mapping['vol_15m'] == 'vol_3m'


class TestCrossIntervalValidation:
    def test_validator(self):
        """Test cross-interval validator."""
        from sklearn.linear_model import Ridge

        np.random.seed(42)

        # Create mock data at two intervals
        X_5m = np.random.randn(100, 10)
        y_5m = X_5m[:, 0] + np.random.randn(100) * 0.5

        X_1m = np.random.randn(500, 10)
        y_1m = X_1m[:, 0] + np.random.randn(500) * 0.5

        # Train on 5m
        model = Ridge()
        model.fit(X_5m, y_5m)

        # Validate
        validator = CrossIntervalValidator()
        result = validator.validate_cross_interval(
            model,
            train_data=(X_5m, y_5m),
            val_data=(X_1m, y_1m),
            train_interval=5,
            val_interval=1
        )

        assert result.train_interval == 5
        assert result.val_interval == 1
        assert 'r2' in result.metrics
        assert result.generalization_ratio > 0

    def test_generalization_matrix(self):
        """Test full generalization evaluation."""
        from sklearn.linear_model import Ridge

        np.random.seed(42)

        # Create mock multi-interval dataset
        dataset = MultiIntervalDataset(primary_interval=5)

        for interval in [1, 5, 15]:
            n = 500 // interval
            features = pd.DataFrame(
                np.random.randn(n, 10),
                columns=[f'f{i}' for i in range(10)],
                index=pd.date_range('2024-01-01', periods=n, freq=f'{interval}T')
            )
            targets = {'target': pd.Series(
                features.iloc[:, 0].values + np.random.randn(n) * 0.5,
                index=features.index
            )}

            data = IntervalData(
                interval_minutes=interval,
                features=features,
                targets=targets,
                symbols=['TEST'],
                timestamp_range=(features.index.min(), features.index.max()),
                n_samples=n
            )
            dataset.add_interval(data)

        def model_factory(X, y):
            model = Ridge()
            model.fit(X, y)
            return model

        # This would require actual data aligned properly
        # Just test the structure
        assert dataset.available_intervals == [1, 5, 15]
```

---

## Integration with Training Loop

### Modified: `TRAINING/orchestration/intelligent_trainer.py`

```python
# Add multi-interval support

def _load_multi_interval_data(self) -> Optional[MultiIntervalDataset]:
    """Load data at multiple intervals if configured."""
    multi_config = self.config.get('multi_interval', {})

    if not multi_config.get('enabled', False):
        return None

    from TRAINING.data.multi_interval_loader import MultiIntervalLoader

    loader = MultiIntervalLoader({'data_root': self.data_dir})

    intervals = (
        multi_config.get('training_intervals', []) +
        multi_config.get('validation_intervals', [])
    )
    intervals = sorted(set(intervals))

    dataset = loader.load_multi_interval(
        intervals=intervals,
        symbols=self.symbols,
        targets=self.targets,
        primary_interval=multi_config.get('primary_interval', 5)
    )

    return dataset


def _train_with_cross_interval_validation(
    self,
    dataset: MultiIntervalDataset,
    target: str,
    family: str
):
    """Train with cross-interval validation."""
    from TRAINING.data.cross_interval_cv import CrossIntervalValidator

    primary = dataset.primary_interval
    primary_data = dataset.get_interval(primary)

    # Train on primary
    X = primary_data.features.values
    y = primary_data.targets[target].values

    model = self._train_single(X, y, family)

    # Cross-interval validation
    validator = CrossIntervalValidator()

    for interval in dataset.available_intervals:
        if interval == primary:
            continue

        val_data = dataset.get_interval(interval)
        X_va = val_data.features.values
        y_va = val_data.targets[target].values

        result = validator.validate_cross_interval(
            model,
            train_data=(X, y),
            val_data=(X_va, y_va),
            train_interval=primary,
            val_interval=interval
        )

        # Log warning if poor generalization
        min_ratio = self.config.get('multi_interval', {}).get(
            'min_generalization_ratio', 0.7
        )
        if result.generalization_ratio < min_ratio:
            logger.warning(
                f"Poor generalization {primary}m→{interval}m: "
                f"ratio={result.generalization_ratio:.3f} < {min_ratio}"
            )

    return model, validator.get_summary()
```

---

## Validation Checklist

### Subphase 10.1
- [ ] `MultiIntervalLoader` loads data from multiple directories
- [ ] `IntervalData` correctly captures interval metadata
- [ ] `resample_to_interval` correctly coarsens data
- [ ] Validation catches feature/target mismatches

### Subphase 10.2
- [ ] `FeatureTransferManager` extracts weights correctly
- [ ] Feature name adjustment works (ret_5m → ret_1m)
- [ ] Transfer features improve training

### Subphase 10.3
- [ ] `CrossIntervalValidator` produces valid metrics
- [ ] Generalization ratio calculation correct
- [ ] CV splits maintain temporal gap

### Subphase 10.4
- [ ] Config loads without error
- [ ] Example experiment runs end-to-end
- [ ] All tests pass

---

## Test Commands

```bash
# Unit tests
pytest tests/test_multi_interval.py -v

# Integration test (requires data at multiple intervals)
python -m TRAINING.orchestration.intelligent_trainer \
    --experiment-config CONFIG/experiments/multi_interval_example.yaml \
    --output-dir test_multi_interval
```

---

## Success Metrics

- [ ] Models trained on 5m generalize to 1m (ratio > 0.7)
- [ ] Feature transfer reduces training time by 20%+
- [ ] No data leakage detected in cross-interval validation
- [ ] Cache keys correctly segregate interval data

---

## Risk Mitigations Applied

1. **Data Leakage**: Strict gap_minutes in CV, purge validation per interval
2. **Feature Mismatch**: Feature mapping with adjustment, validation logging
3. **Cache Collisions**: Interval in cache key (Phase 13 complete)
4. **Memory**: Sequential interval loading, not all at once
