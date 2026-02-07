# Model Inference

Guidelines for loading and running trained models.

## Model Loading (LIVE_TRADING)

Use the dedicated loader in `LIVE_TRADING/models/loader.py`:

```python
from LIVE_TRADING.models.loader import load_model_for_target

# Load model with checksum verification (H2 security)
model, meta = load_model_for_target(
    run_dir=Path("/path/to/run"),
    target="fwd_ret_5m",
    model_family="lightgbm",
)

# Model is ready for inference
predictions = model.predict(X)
```

## Path Resolution (SST Pattern)

For custom loading, use SST path helpers with required arguments:

```python
from TRAINING.orchestration.utils.target_first_paths import run_root, get_target_dir

# run_root() requires output_dir argument (walks up to find run root)
run_dir = run_root(output_dir=some_artifact_path)

# get_target_dir() builds target-first path
model_path = get_target_dir(run_dir, target_name, model_family)
```

**Note**: `run_root(output_dir)` walks up from `output_dir` to find the run directory containing `targets/`, `globals/`, or `cache/` directories.

## Model Families

See `CONFIG/models/*.yaml` for all available families and their configs.

**Categories:**
- **Tabular**: LightGBM, XGBoost, CatBoost, MLP, etc.
- **Sequential**: CNN1D, LSTM, Transformer, TabLSTM, TabTransformer
- **Probabilistic**: NGBoost, QuantileLightGBM
- **Specialized**: VAE, GAN, GMMRegime, ChangePoint, FTRL
- **Meta**: Ensemble, MetaLearning, MultiTask, RewardBased

## Input Formats

**Tabular Models**:
```python
# Shape: (N_samples, N_features)
X = np.array([[f1, f2, ...], ...])
predictions = model.predict(X)
```

**Sequential Models**:
```python
from TRAINING.common.live.seq_ring_buffer import SeqBufferManager

# Shape: (N_samples, sequence_length, N_features)
buffer = SeqBufferManager(seq_len=60, n_features=200)
buffer.push(new_features)
X = buffer.get_sequence()
predictions = model.predict(X)
```

## Multi-Horizon Inference

```python
horizons = ['5m', '10m', '15m', '30m', '60m', '1d']
predictions = {}

for horizon in horizons:
    horizon_preds = {}
    for family in available_families:
        model = load_model(horizon, family)
        if model:
            horizon_preds[family] = model.predict(X)
    predictions[horizon] = horizon_preds
```

## Standardization

Apply z-score standardization:
```python
s = np.clip((raw_pred - rolling_mean) / rolling_std, -3, 3)
```

Rolling statistics: N ≈ 5-10 trading days

## Related Skills

- `signal-generation.md` - Using predictions for signals
- `execution-engine.md` - Prediction layer integration
- `adding-new-models.md` - Training new model families

## Related Documentation

- `TRAINING/models/registry.py` - Model registry
- `TRAINING/common/live/seq_ring_buffer.py` - Sequential buffer implementation
