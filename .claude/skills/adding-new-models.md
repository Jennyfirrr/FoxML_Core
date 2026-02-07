# Adding New Models

Skill for extending the model trainer system with new model families.

## BaseModelTrainer Contract

All model trainers inherit from `TRAINING/model_fun/base_trainer.py`. The contract defines:

### Required Methods

```python
@abstractmethod
def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
          feature_names: List[str] = None, **kwargs) -> Dict[str, Any]:
    """Train the model. Must call preprocess_data() first."""
    pass

@abstractmethod
def predict(self, X_tr: np.ndarray) -> np.ndarray:
    """Make predictions. Must call preprocess_data() for inference."""
    pass
```

### Inherited Methods (Use These)

| Method | Purpose | SST/Determinism Impact |
|--------|---------|----------------------|
| `preprocess_data(X, y)` | Imputation, NaN handling | Fits imputer on train, reuses on predict |
| `_get_seed()` | Get deterministic seed | Sources from `BASE_SEED` (SST) |
| `_get_test_split_params()` | Get test_size and seed | Sources from config (SST) |
| `_get_learning_rate(default)` | Get learning rate | Sources from config (SST) |
| `get_callbacks(family_name)` | Get training callbacks | Sources from callbacks config (SST) |
| `post_fit_sanity(X, name)` | Validate predictions | Ensures finite outputs |
| `save_model(filepath)` | Serialize model | Includes imputer and colmask |
| `load_model(filepath)` | Deserialize model | Restores full state |

### Threading Helpers

```python
# Use these for thread-safe training/prediction
def fit_with_threads(self, estimator, X, y, sample_weight=None, *, phase="fit"):
    """Fit with automatic threading configuration."""

def predict_with_threads(self, estimator, X, *, phase="predict"):
    """Predict with automatic threading configuration."""
```

**Threading safety requirements:**
- Use `RLock` (not `Lock`) for nested contexts - prevents deadlocks
- If using multiprocessing, use `spawn` start method (not `fork`)

```python
# For internal locks in trainers
import threading
self._lock = threading.RLock()  # ✅ RLock for nested contexts

# For multiprocessing (if needed)
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)  # ✅ Not 'fork'
```

**See `determinism-and-reproducibility.md`** for full threading/multiprocessing guidelines.

## Implementation Checklist

### 1. Create Trainer File

Create `TRAINING/model_fun/{family}_trainer.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

import numpy as np
import logging
from typing import Any, Dict, List
from pathlib import Path
import sys

from .base_trainer import BaseModelTrainer

logger = logging.getLogger(__name__)

# SST: Add CONFIG to path for centralized config loading
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

try:
    from config_loader import load_model_config
    _USE_CENTRALIZED_CONFIG = True
except ImportError:
    logger.warning("Could not import config_loader, falling back to defaults")
    _USE_CENTRALIZED_CONFIG = False


class MyFamilyTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # SST: Load from centralized CONFIG if not provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("my_family")
                logger.info("Loaded MyFamily config from CONFIG/models/my_family.yaml")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
                config = {}

        super().__init__(config or {})

        # Set defaults (should match CONFIG/models/my_family.yaml)
        self.config.setdefault("learning_rate", 0.01)
        self.config.setdefault("n_estimators", 100)
        # ... other defaults

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        # 1) Preprocess data (handles NaN, imputation)
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]

        # 2) Split if no external validation (use SST seed)
        if X_va is None or y_va is None:
            from sklearn.model_selection import train_test_split
            test_size, seed = self._get_test_split_params()
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=test_size, random_state=seed
            )

        # 3) Build and train model
        model = self._build_model()
        model.fit(X_tr, y_tr)

        # 4) Store state and sanity check
        self.model = model
        self.is_trained = True
        self.post_fit_sanity(X_tr, "MyFamily")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        Xp, _ = self.preprocess_data(X, None)
        preds = self.model.predict(Xp)
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)

    def _build_model(self):
        """Build the model instance."""
        # Get seed from determinism system (SST)
        seed = self._get_seed()

        # Build model with config values
        from some_library import SomeModel
        return SomeModel(
            learning_rate=self.config["learning_rate"],
            n_estimators=self.config["n_estimators"],
            random_state=seed,  # Determinism: use SST seed
        )
```

### 2. Create Config File

Create `CONFIG/models/{family}.yaml`:

```yaml
# Model Family: MyFamily
# Hyperparameters for MyFamily model trainer

hyperparameters:
  learning_rate: 0.01
  n_estimators: 100
  # Add other hyperparameters

# Optional: variants for different use cases
variants:
  conservative:
    learning_rate: 0.005
    n_estimators: 50
  aggressive:
    learning_rate: 0.05
    n_estimators: 200

# Metadata
metadata:
  family: my_family
  description: "Description of this model family"
  complexity: medium  # low, medium, high
  gpu_capable: false
```

### 3. Register in __init__.py

Edit `TRAINING/model_fun/__init__.py`:

```python
# In appropriate section (CPU-only or TF/Torch)
from .my_family_trainer import MyFamilyTrainer

__all__ = [
    # ... existing trainers
    'MyFamilyTrainer',
]
```

### 4. Add to Family Config

Edit `CONFIG/pipeline/training/families.yaml`:

```yaml
families:
  # ... existing families
  my_family:
    enabled: true
    complexity: medium
    gpu_capable: false
```

## Determinism Requirements

### Seeds
- **Always** use `self._get_seed()` for random state
- **Never** use `time.time()`, `datetime.now()`, or unseeded randomness
- **Never** use `np.random.seed()` globally - use per-instance seeds

### Config Access
- **Always** load config via `load_model_config(family)` (SST)
- **Always** use `self.config.setdefault()` for defaults
- **Never** hardcode config values without setdefault fallback

### Threading
- Use `self._threads()` to get thread count
- Use `fit_with_threads()` for BLAS-heavy operations
- Respect `OMP_NUM_THREADS` environment variable

### Predictions
- **Always** call `self.preprocess_data(X, None)` before predict
- **Always** handle NaN in output: `np.nan_to_num(preds, nan=0.0)`
- **Always** cast to consistent dtype: `.astype(np.float32)`

## Testing New Trainers

### Basic Smoke Test

```python
def test_my_family_trainer():
    from TRAINING.model_fun.my_family_trainer import MyFamilyTrainer

    # Create synthetic data
    X = np.random.randn(1000, 50).astype(np.float32)
    y = np.random.randn(1000).astype(np.float64)

    # Train
    trainer = MyFamilyTrainer()
    trainer.train(X, y)

    # Predict
    preds = trainer.predict(X[:100])
    assert preds.shape == (100,)
    assert np.isfinite(preds).all()
```

### Determinism Test

```python
def test_my_family_determinism():
    """Same inputs produce identical outputs."""
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100).astype(np.float64)

    trainer1 = MyFamilyTrainer()
    trainer1.train(X.copy(), y.copy())
    preds1 = trainer1.predict(X[:10])

    trainer2 = MyFamilyTrainer()
    trainer2.train(X.copy(), y.copy())
    preds2 = trainer2.predict(X[:10])

    np.testing.assert_array_equal(preds1, preds2)
```

## Reference Implementation

See `TRAINING/model_fun/lightgbm_trainer.py` for a complete reference implementation showing:
- Config loading from SST
- Seed handling from determinism system
- Thread configuration
- Early stopping with validation
- Post-fit sanity checks

## Common Mistakes

| Mistake | Impact | Fix |
|---------|--------|-----|
| Not calling `preprocess_data()` | NaN in training | Always call first in train() and predict() |
| Using `np.random.seed()` | Non-deterministic | Use `self._get_seed()` for per-instance seeds |
| Hardcoding config values | SST violation | Use `load_model_config()` + `setdefault()` |
| Missing `post_fit_sanity()` | Silent NaN predictions | Call after setting `self.model` |
| Wrong dtype in predict | Inconsistent outputs | Cast to `np.float32` |

## Sequence Model Trainers (RAW_SEQUENCE Mode)

For neural models that accept raw OHLCV sequences instead of engineered features, implement a sequence-capable trainer.

### Input Modes

| Mode | Input Shape | Use Case |
|------|-------------|----------|
| `FEATURES` | `(samples, features)` | Tree models, classic ML |
| `RAW_SEQUENCE` | `(samples, sequence_len, channels)` | LSTM, Transformer, CNN1D |

### Implementing a Sequence Trainer

```python
class MySequenceTrainer(BaseModelTrainer):
    """Trainer that accepts raw OHLCV sequences."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})

        # Sequence-specific config
        self.config.setdefault("sequence_length", 60)
        self.config.setdefault("input_channels", 5)  # OHLCV
        self.supports_sequence_input = True  # Mark as sequence-capable

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, **kwargs) -> Any:
        """
        Train on sequence data.

        Args:
            X_tr: Shape (samples, sequence_len, channels) for RAW_SEQUENCE
                  or (samples, features) for FEATURES mode
            y_tr: Shape (samples,) targets
        """
        # Check input shape to determine mode
        if X_tr.ndim == 3:
            # Sequence input: (samples, seq_len, channels)
            return self._train_sequence(X_tr, y_tr, **kwargs)
        else:
            # Tabular input: fall back to base behavior
            X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
            return self._train_tabular(X_tr, y_tr, **kwargs)

    def _train_sequence(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train on sequence data - no feature preprocessing."""
        # Build sequence model
        seq_len = X.shape[1]
        n_channels = X.shape[2]
        model = self._build_sequence_model(seq_len, n_channels)

        # Train (handle NaN in sequences)
        X = np.nan_to_num(X, nan=0.0)
        model.fit(X, y, ...)

        self.model = model
        self.is_trained = True
        return model
```

### Sequence Data Format

In `RAW_SEQUENCE` mode, the pipeline provides data as:

```python
# X shape: (samples, sequence_length, 5)
# Channels: [open, high, low, close, volume]

# Example config in experiment YAML:
intelligent_training:
  input_mode: RAW_SEQUENCE
  sequence_length: 60  # 60 bars of history
  sequence_channels: [open, high, low, close, volume]
```

### Existing Sequence Trainers

| Trainer | File | Input Shape |
|---------|------|-------------|
| LSTM | `lstm_trainer.py` | `(N, seq_len, channels)` |
| Transformer | `transformer_trainer.py` | `(N, seq_len, channels)` |
| CNN1D | `cnn1d_trainer.py` | `(N, seq_len, channels)` |

### Hybrid Mode

Some models can accept both tabular features AND sequence data:

```python
class HybridTrainer(BaseModelTrainer):
    """Trainer that combines features with sequence data."""

    def train(self, X_features, y, X_sequence=None, **kwargs):
        if X_sequence is not None:
            # Combine tabular features with sequence embedding
            seq_embedding = self.sequence_encoder(X_sequence)
            X_combined = np.concatenate([X_features, seq_embedding], axis=1)
        else:
            X_combined = X_features
        # ... train on combined features
```

## Related Skills

- `determinism-and-reproducibility.md` - Determinism requirements for model training
- `configuration-management.md` - Config loading patterns
- `testing-guide.md` - Testing new trainers
- `architecture-overview.md` - Pipeline input modes

## Related Documentation

- `TRAINING/model_fun/base_trainer.py` - Base class implementation
- `TRAINING/model_fun/lightgbm_trainer.py` - Reference implementation
- `CONFIG/models/` - Model config files
- `INTERNAL/docs/references/DETERMINISTIC_PATTERNS.md` - Determinism patterns
