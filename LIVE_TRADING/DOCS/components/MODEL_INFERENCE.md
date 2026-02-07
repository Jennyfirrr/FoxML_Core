# Model Inference

This document describes how the LIVE_TRADING module loads and runs trained models from the TRAINING pipeline.

---

## Overview

The model inference subsystem handles:

1. **Model Loading**: Loading models from TRAINING artifacts (supports pickle, joblib, HDF5, Keras, LightGBM native formats)
2. **Family-Specific Inference**: Different inference paths for different model families (tree, Keras, sequential)
3. **Feature Building**: Real-time technical indicator calculation from OHLCV data
4. **Security**: H2 FIX - SHA256 checksum verification before loading pickle files
5. **Interval Validation**: Phase 17 - Validates data interval matches model training interval

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODEL INFERENCE LAYER                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌───────────────────┐     ┌──────────────────┐   │
│  │ ModelLoader  │────►│  InferenceEngine  │────►│  Predictions     │   │
│  │              │     │                   │     │                  │   │
│  │ - Load from  │     │ - Tree families   │     │ - Per horizon    │   │
│  │   TRAINING   │     │ - Keras families  │     │ - Per family     │   │
│  │ - Checksum   │     │ - Sequential      │     │ - Standardized   │   │
│  │   verify     │     │                   │     │                  │   │
│  └──────────────┘     └───────────────────┘     └──────────────────┘   │
│         │                      │                                         │
│         │                      │                                         │
│         ▼                      ▼                                         │
│  ┌──────────────┐     ┌───────────────────┐                             │
│  │ TRAINING     │     │  FeatureBuilder   │                             │
│  │ Artifacts    │     │                   │                             │
│  │              │     │ - RSI, MACD       │                             │
│  │ - models/    │     │ - Bollinger       │                             │
│  │ - manifests  │     │ - ATR, Returns    │                             │
│  └──────────────┘     └───────────────────┘                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### ModelLoader

**Location**: `LIVE_TRADING/models/loader.py`

Responsible for loading models from TRAINING artifact directories.

```python
from LIVE_TRADING.models import ModelLoader

loader = ModelLoader(run_root="/path/to/RESULTS/runs/my_run")

# Load a specific model
model = loader.load(
    target="fwd_ret_15m",
    family="lightgbm"
)

# Load all models for a symbol
models = loader.load_all(
    targets=["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m"],
    families=["lightgbm", "xgboost", "ridge"]
)
```

#### Model Discovery

The loader searches for models in the TRAINING artifact structure (SST-compliant paths):

```
RESULTS/runs/{run_id}/{timestamp}/
├── manifest.json                          # Contains target_index
└── targets/
    └── {target}/                          # e.g., "ret_5m", "fwd_ret_15m"
        └── models/
            └── view=CROSS_SECTIONAL/      # or view=SEQUENTIAL
                ├── routing_decision.json  # Selected family info (optional)
                └── family={family}/       # e.g., "family=LightGBM"
                    ├── model.pkl          # Tree models (pickle)
                    ├── model.joblib       # Alternative format
                    ├── model.txt          # LightGBM native format
                    ├── model.h5           # Keras models
                    ├── model.keras        # Keras SavedModel
                    └── model_meta.json    # REQUIRED: feature list, metrics, checksum
```

#### Model Metadata (`model_meta.json`)

```python
{
    "family": "LightGBM",
    "feature_list": ["ret_1d", "vol_10d", "rsi_14", ...],  # Ordered feature names
    "metrics": {"auc": 0.65, "ic": 0.08},
    "model_checksum": "sha256hash...",    # H2 FIX: Security
    "interval_minutes": 1440.0,           # Phase 17: Training interval
    "sequence_length": 20,                # Sequential models only
}
```

#### Security Features (H2 FIX)

```python
# Checksum verification before pickle loading
loader = ModelLoader(
    run_root=path,
    verify_checksums=True,    # Default: from config
    strict_checksums=False,   # If True, raises on mismatch; False = warn only
)

# Verification happens automatically in load_model():
# - Computes SHA256 of model file
# - Compares against "model_checksum" in model_meta.json
# - Strict mode: raises ModelLoadError on mismatch
# - Non-strict mode: logs warning and continues
```

---

### InferenceEngine

**Location**: `LIVE_TRADING/models/inference.py`

Handles family-specific inference logic.

```python
from LIVE_TRADING.models import InferenceEngine

engine = InferenceEngine()

# Tree family inference (LightGBM, XGBoost, CatBoost)
predictions = engine.predict(model, features, family="lightgbm")

# Keras family inference (MLP, CNN1D, LSTM)
predictions = engine.predict(model, features, family="mlp")

# Sequential family inference (needs buffer state)
predictions = engine.predict_sequential(
    model,
    buffer,
    family="lstm"
)
```

#### Model Family Categories

| Category | Families | Input Shape | Notes |
|----------|----------|-------------|-------|
| **Tree** | lightgbm, xgboost, catboost, random_forest, extra_trees | `(N, F)` | Direct pickle predict |
| **Keras** | mlp, cnn1d, lstm, transformer | `(N, F)` or `(N, T, F)` | TensorFlow .h5 load |
| **Sequential** | tab_lstm, tab_transformer, cnn1d_transformer | `(N, T, F)` | Requires SeqBufferManager |
| **Probabilistic** | ngboost, quantile_lightgbm | `(N, F)` | Returns distribution |
| **Meta** | ensemble, meta_learning | `(N, F)` | Multi-model stacking |

#### Inference Dispatch

```python
def predict(self, model, features: np.ndarray, family: str) -> np.ndarray:
    """Dispatch to family-specific inference."""
    if family in TREE_FAMILIES:
        return self._predict_tree(model, features)
    elif family in KERAS_FAMILIES:
        return self._predict_keras(model, features)
    elif family in SEQUENTIAL_FAMILIES:
        raise ValueError(f"Use predict_sequential for {family}")
    else:
        raise ValueError(f"Unknown family: {family}")

def _predict_tree(self, model, features: np.ndarray) -> np.ndarray:
    """Tree model prediction."""
    return model.predict(features)

def _predict_keras(self, model, features: np.ndarray) -> np.ndarray:
    """Keras model prediction."""
    return model.predict(features, verbose=0).flatten()
```

---

### FeatureBuilder

**Location**: `LIVE_TRADING/models/feature_builder.py`

Builds technical indicators from market data for model input.

```python
from LIVE_TRADING.models import FeatureBuilder

builder = FeatureBuilder()

# Build features from OHLCV data
features = builder.build(
    ohlcv_data,  # DataFrame with open, high, low, close, volume
    lookback=60  # bars
)

# Available features
# - Returns (1, 5, 10, 20, 60 bar)
# - RSI (14 period)
# - MACD (12, 26, 9)
# - Bollinger Bands (20, 2)
# - ATR (14 period)
# - Volume moving averages
```

#### Available Indicators

| Indicator | Formula | Parameters |
|-----------|---------|------------|
| **Returns** | `(close - close[n]) / close[n]` | n = 1, 5, 10, 20, 60 |
| **RSI** | Standard RSI | period = 14 |
| **MACD** | EMA12 - EMA26, Signal = EMA9 | 12, 26, 9 |
| **Bollinger** | mean ± 2σ | period = 20, std = 2 |
| **ATR** | TR average | period = 14 |
| **Volume MA** | volume / MA(volume) | period = 20 |

#### Feature Calculation

```python
def build(self, df: pd.DataFrame, lookback: int = 60) -> np.ndarray:
    """Build feature vector from OHLCV data."""
    features = {}

    # Returns at multiple horizons
    for n in [1, 5, 10, 20, 60]:
        features[f'ret_{n}'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n)

    # RSI
    features['rsi_14'] = self._calculate_rsi(df['close'], 14)

    # MACD
    macd, signal, hist = self._calculate_macd(df['close'], 12, 26, 9)
    features['macd'] = macd
    features['macd_signal'] = signal
    features['macd_hist'] = hist

    # Bollinger Bands
    upper, lower = self._calculate_bollinger(df['close'], 20, 2)
    features['bb_upper'] = upper
    features['bb_lower'] = lower
    features['bb_pct'] = (df['close'] - lower) / (upper - lower)

    # ATR
    features['atr_14'] = self._calculate_atr(df, 14)

    # Volume
    features['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    return pd.DataFrame(features).iloc[-1].values
```

---

## Sequential Model Support

### SeqBufferManager

For sequential models (LSTM, Transformer), we need to maintain a rolling buffer of historical features.

```python
from TRAINING.common.live.seq_ring_buffer import SeqBufferManager

# Initialize buffer
buffer = SeqBufferManager(
    seq_len=60,       # Sequence length
    n_features=200,   # Number of features
    symbol="AAPL"
)

# Push new data
buffer.push(new_features)  # Shape: (1, n_features)

# Get sequence for prediction
sequence = buffer.get_sequence()  # Shape: (1, seq_len, n_features)

# Predict
predictions = model.predict(sequence)
```

### Buffer State Persistence

```python
# Save buffer state between sessions
buffer_state = buffer.to_dict()
write_atomic_json("state/buffer_AAPL.json", buffer_state)

# Restore buffer
buffer_state = read_json("state/buffer_AAPL.json")
buffer = SeqBufferManager.from_dict(buffer_state)
```

---

## Multi-Horizon Inference

### Prediction Flow

```python
from LIVE_TRADING.models import ModelLoader, InferenceEngine, FeatureBuilder
from LIVE_TRADING.common.constants import HORIZONS, FAMILIES

loader = ModelLoader(run_root=run_root)
engine = InferenceEngine()
builder = FeatureBuilder()

def predict_all_horizons(market_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Generate predictions for all horizons and families."""
    features = builder.build(market_data)

    predictions = {}
    for horizon in HORIZONS:
        predictions[horizon] = {}
        for family in FAMILIES:
            model = loader.load(target=f"fwd_ret_{horizon}", family=family)
            if model is not None:
                pred = engine.predict(model, features.reshape(1, -1), family)
                predictions[horizon][family] = float(pred[0])

    return predictions
```

### Lazy Loading

Models are loaded on first access and cached:

```python
class ModelLoader:
    def __init__(self, run_root: str):
        self._run_root = Path(run_root)
        self._cache: Dict[Tuple[str, str], Any] = {}

    def load(self, target: str, family: str) -> Optional[Any]:
        cache_key = (target, family)
        if cache_key not in self._cache:
            model = self._load_from_disk(target, family)
            self._cache[cache_key] = model
        return self._cache[cache_key]
```

---

## Model Families Reference

### Tree Families

| Family | Library | Predict Method | Output |
|--------|---------|----------------|--------|
| `lightgbm` | LightGBM | `model.predict(X)` | `(N,)` |
| `xgboost` | XGBoost | `model.predict(X)` | `(N,)` |
| `catboost` | CatBoost | `model.predict(X)` | `(N,)` |
| `random_forest` | sklearn | `model.predict(X)` | `(N,)` |
| `extra_trees` | sklearn | `model.predict(X)` | `(N,)` |

### Neural Network Families

| Family | Library | Input Shape | Output |
|--------|---------|-------------|--------|
| `mlp` | TensorFlow | `(N, F)` | `(N, 1)` |
| `cnn1d` | TensorFlow | `(N, T, F)` | `(N, 1)` |
| `lstm` | TensorFlow | `(N, T, F)` | `(N, 1)` |
| `transformer` | TensorFlow | `(N, T, F)` | `(N, 1)` |

### Probabilistic Families

| Family | Library | Output |
|--------|---------|--------|
| `ngboost` | NGBoost | Distribution params |
| `quantile_lightgbm` | LightGBM | Quantile predictions |

---

## Error Handling

### ModelLoadError

```python
from LIVE_TRADING.common.exceptions import ModelLoadError

try:
    model = loader.load(target="fwd_ret_5m", family="lightgbm")
except ModelLoadError as e:
    logger.warning(f"Failed to load model: {e}")
    # Use fallback or skip this model
```

### Graceful Degradation

```python
def predict_with_fallback(self, target: str, families: List[str]) -> Optional[float]:
    """Try multiple families, return first successful prediction."""
    for family in families:
        try:
            model = self.loader.load(target, family)
            if model is not None:
                pred = self.engine.predict(model, features, family)
                return float(pred[0])
        except (ModelLoadError, ValueError) as e:
            logger.debug(f"Family {family} failed: {e}")
            continue
    return None  # All families failed
```

---

## Configuration

```yaml
# CONFIG/live_trading/live_trading.yaml

live_trading:
  models:
    # Lazy loading
    lazy_load: true
    cache_models: true

    # Security
    verify_checksums: true
    allow_unsigned: false  # Reject models without checksums

    # Fallback order
    family_priority:
      - lightgbm
      - xgboost
      - ridge
      - mlp

    # Sequential models
    seq_buffer_size: 60
    seq_warmup_bars: 20  # Minimum bars before prediction
```

---

## Testing

```bash
# Run model inference tests
pytest LIVE_TRADING/tests/test_model_loader.py -v      # 19 tests
pytest LIVE_TRADING/tests/test_inference.py -v         # 22 tests

# Test feature builder
pytest LIVE_TRADING/tests/test_feature_builder.py -v
```

---

## Related Documentation

- [MULTI_HORIZON_AND_INTERVAL.md](MULTI_HORIZON_AND_INTERVAL.md) - Multi-horizon prediction pipeline
- [../architecture/PIPELINE_STAGES.md](../architecture/PIPELINE_STAGES.md) - Prediction stage details
- [../reference/CONFIGURATION_REFERENCE.md](../reference/CONFIGURATION_REFERENCE.md) - Full configuration options
