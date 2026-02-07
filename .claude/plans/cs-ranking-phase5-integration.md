# Phase 5: Pipeline Integration

**Parent**: `cross-sectional-ranking-objective.md`
**Status**: ✅ MOSTLY COMPLETE (Foundation wired, training loop TBD)
**Estimated Effort**: 2-3 hours
**Completed**: 2026-01-21

## Objective

Integrate cross-sectional ranking into the existing pipeline with minimal disruption. Enable toggling between pointwise and ranking modes via config.

## Current Pipeline Flow

```
intelligent_trainer.py
    │
    ├── Stage 1: Target Ranking (discover predictable targets)
    │       └── Uses: feature-based predictability scoring
    │
    ├── Stage 2: Feature Selection (per-target feature sets)
    │       └── Uses: MI, correlation, leakage detection
    │
    └── Stage 3: Model Training
            └── Uses: pointwise loss, flat batches
```

## Modified Pipeline Flow (CS Ranking Mode)

```
intelligent_trainer.py
    │
    ├── [SKIP] Stage 1: Target Ranking
    │       └── Not needed: CS ranking defines its own objective
    │
    ├── [SKIP] Stage 2: Feature Selection
    │       └── Not needed: Raw OHLCV sequences are the features
    │
    └── Stage 3: Model Training (MODIFIED)
            ├── Data: CrossSectionalDataset (Phase 2)
            ├── Targets: CS percentile (Phase 1)
            ├── Loss: Pairwise/Listwise (Phase 3)
            └── Metrics: Spearman IC (Phase 4)
```

## Implementation

### 1. Config Schema Extension

```yaml
# CONFIG/pipeline/pipeline.yaml additions

pipeline:
  # Existing
  input_mode: "features"  # "features" or "raw_sequence"

  # NEW: Cross-sectional ranking configuration
  cross_sectional_ranking:
    enabled: false  # Master switch

    # Target construction (Phase 1)
    target:
      type: "cs_percentile"      # "cs_percentile", "cs_zscore", "vol_scaled"
      residualize: true          # Subtract market mean
      winsorize: [0.01, 0.99]    # Clip extremes
      # For vol_scaled only
      vol_lookback_bars: 20
      vol_method: "realized"

    # Loss function (Phase 3)
    loss:
      type: "pairwise"           # "pairwise", "listwise", "pointwise"
      # Pairwise settings
      top_pct: 0.2               # Top 20% as winners
      bottom_pct: 0.2            # Bottom 20% as losers
      max_pairs_per_timestamp: 100
      margin: 0.0                # Optional hinge margin
      # Listwise settings
      temperature: 1.0           # Softmax temperature

    # Batching (Phase 2)
    batching:
      timestamps_per_batch: 32   # B dimension
      symbols_per_timestamp: null # M dimension (null = all)
      min_symbols_per_timestamp: 50
      shuffle_timestamps: true   # Shuffle across epochs

    # Metrics (Phase 4)
    metrics:
      primary: "spearman_ic"
      top_pct: 0.1               # For spread calculation
      bottom_pct: 0.1
      cost_per_trade_bps: 5.0    # For cost-adjusted metrics
```

### 2. intelligent_trainer.py Modifications

```python
# TRAINING/orchestration/intelligent_trainer.py

from TRAINING.common.input_mode import InputMode, get_input_mode
from CONFIG.config_loader import get_cfg


class IntelligentTrainer:
    def __init__(self, ...):
        # ...
        self.input_mode = get_input_mode()
        self.cs_ranking_enabled = get_cfg(
            "pipeline.cross_sectional_ranking.enabled", default=False
        )

    def run(self):
        """Main orchestration with CS ranking support."""

        # Stage 1: Target Ranking
        if self._should_run_target_ranking():
            self._run_target_ranking()
        else:
            logger.info("Skipping target ranking (CS ranking mode)")

        # Stage 2: Feature Selection
        if self._should_run_feature_selection():
            self._run_feature_selection()
        else:
            logger.info("Skipping feature selection (CS ranking mode)")

        # Stage 3: Model Training
        self._run_model_training()

    def _should_run_target_ranking(self) -> bool:
        """Target ranking not needed for CS ranking mode."""
        if self.cs_ranking_enabled:
            return False
        if self.input_mode == InputMode.RAW_SEQUENCE:
            return False
        return True

    def _should_run_feature_selection(self) -> bool:
        """Feature selection not needed for CS ranking or raw sequence."""
        if self.cs_ranking_enabled:
            return False
        if self.input_mode == InputMode.RAW_SEQUENCE:
            return False
        return True

    def _get_training_config(self) -> Dict:
        """Build training config based on mode."""
        if self.cs_ranking_enabled:
            return self._get_cs_ranking_training_config()
        else:
            return self._get_standard_training_config()

    def _get_cs_ranking_training_config(self) -> Dict:
        """Config for cross-sectional ranking training."""
        cs_cfg = get_cfg("pipeline.cross_sectional_ranking", default={})
        return {
            'mode': 'cross_sectional_ranking',
            'target_config': cs_cfg.get('target', {}),
            'loss_config': cs_cfg.get('loss', {}),
            'batching_config': cs_cfg.get('batching', {}),
            'metrics_config': cs_cfg.get('metrics', {}),
        }
```

### 3. Model Trainer Modifications

```python
# TRAINING/training_strategies/execution/training.py

from TRAINING.data.cross_sectional_dataset import (
    CrossSectionalDataset,
    collate_cross_sectional,
)
from TRAINING.losses.ranking_losses import get_ranking_loss
from TRAINING.metrics.ranking_metrics import compute_ranking_metrics


def train_model(
    model_family: str,
    data: Dict,
    config: Dict,
    ...
) -> Dict:
    """
    Train model with support for CS ranking mode.
    """
    mode = config.get('mode', 'standard')

    if mode == 'cross_sectional_ranking':
        return _train_cs_ranking(model_family, data, config, ...)
    else:
        return _train_standard(model_family, data, config, ...)


def _train_cs_ranking(
    model_family: str,
    data: Dict,
    config: Dict,
    ...
) -> Dict:
    """
    Training loop for cross-sectional ranking.
    """
    # Build dataset
    dataset = CrossSectionalDataset(
        sequences=data['sequences'],
        targets=data['cs_targets'],
        timestamps=data['timestamps'],
        symbols=data['symbols'],
        min_symbols_per_timestamp=config['batching_config'].get(
            'min_symbols_per_timestamp', 50
        ),
    )

    # DataLoader with custom collate
    batch_size = config['batching_config'].get('timestamps_per_batch', 32)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=config['batching_config'].get('shuffle_timestamps', True),
        collate_fn=collate_cross_sectional,
        num_workers=0,  # Determinism
    )

    # Get loss function
    loss_fn = get_ranking_loss(
        config['loss_config'].get('type', 'pairwise'),
        config['loss_config'],
    )

    # Training loop
    model = _create_model(model_family, data['input_shape'])
    optimizer = _create_optimizer(model, config)

    best_ic = -1.0
    for epoch in range(config.get('epochs', 50)):
        # Train
        train_loss = _train_epoch_cs(model, loader, optimizer, loss_fn, config)

        # Evaluate
        val_metrics = _evaluate_cs(model, val_loader, config['metrics_config'])

        if val_metrics['spearman_ic'] > best_ic:
            best_ic = val_metrics['spearman_ic']
            _save_checkpoint(model, ...)

        logger.info(
            f"Epoch {epoch}: loss={train_loss:.4f}, "
            f"IC={val_metrics['spearman_ic']:.4f}, "
            f"spread={val_metrics['spread']:.4f}"
        )

    return {
        'best_ic': best_ic,
        'final_metrics': val_metrics,
        'model_path': checkpoint_path,
    }


def _train_epoch_cs(model, loader, optimizer, loss_fn, config):
    """Single epoch of CS ranking training."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        X = batch['X'].to(device)      # (B, M, L, F)
        y = batch['y'].to(device)      # (B, M)
        mask = batch['mask'].to(device) # (B, M)

        optimizer.zero_grad()

        # Forward pass: model outputs (B, M) scores
        scores = model(X, mask)

        # Ranking loss
        loss = loss_fn(scores, y, mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def _evaluate_cs(model, loader, metrics_config):
    """Evaluate model with ranking metrics."""
    model.eval()
    all_scores = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            X = batch['X'].to(device)
            scores = model(X, batch['mask'].to(device))

            all_scores.append(scores.cpu().numpy())
            all_targets.append(batch['y'].numpy())
            all_masks.append(batch['mask'].numpy())

    # Concatenate
    scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # Compute metrics
    return compute_ranking_metrics(scores, targets, masks, metrics_config)
```

### 4. Model Architecture Wrapper

```python
# TRAINING/models/cross_sectional_wrapper.py

import torch
import torch.nn as nn


class CrossSectionalModelWrapper(nn.Module):
    """
    Wraps a sequence model to handle (B, M, L, F) input.

    The wrapped model processes each symbol independently,
    producing (B, M) scores for ranking.
    """

    def __init__(self, base_model: nn.Module):
        """
        Args:
            base_model: Sequence model that takes (batch, L, F) -> (batch, 1)
        """
        super().__init__()
        self.base_model = base_model

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, M, L, F) input tensor
            mask: (B, M) validity mask

        Returns:
            scores: (B, M) ranking scores
        """
        B, M, L, F = X.shape

        # Flatten batch and symbol dimensions
        X_flat = X.view(B * M, L, F)

        # Forward through base model
        scores_flat = self.base_model(X_flat).squeeze(-1)  # (B*M,)

        # Reshape back
        scores = scores_flat.view(B, M)

        # Mask invalid symbols (optional - loss handles this)
        # scores = scores * mask

        return scores
```

### 5. Inference Path for Live Trading

```python
# LIVE_TRADING/prediction/cs_ranking_predictor.py

import torch
import numpy as np
from typing import Dict, List


class CrossSectionalRankingPredictor:
    """
    Inference wrapper for CS ranking models in live trading.

    At each timestamp:
    1. Collect OHLCV sequences for all symbols
    2. Run model to get scores
    3. Rank symbols by score
    4. Return top/bottom symbols for trading
    """

    def __init__(
        self,
        model_path: str,
        model_config: Dict,
        device: str = 'cpu',
    ):
        self.model = self._load_model(model_path, model_config)
        self.model.to(device)
        self.model.eval()
        self.device = device

        # Config
        self.seq_length = model_config.get('sequence_length', 64)
        self.normalization = model_config.get('normalization', 'log_return')
        self.top_pct = model_config.get('top_pct', 0.1)
        self.bottom_pct = model_config.get('bottom_pct', 0.1)

    def predict(
        self,
        symbol_sequences: Dict[str, np.ndarray],  # symbol -> (L, F)
    ) -> Dict:
        """
        Generate rankings for current timestamp.

        Args:
            symbol_sequences: Dict mapping symbol to its (L, F) sequence

        Returns:
            {
                'scores': Dict[str, float],  # symbol -> score
                'rankings': List[str],       # symbols sorted by score (desc)
                'long_symbols': List[str],   # top decile
                'short_symbols': List[str],  # bottom decile
            }
        """
        symbols = sorted(symbol_sequences.keys())
        M = len(symbols)

        if M == 0:
            return {'scores': {}, 'rankings': [], 'long_symbols': [], 'short_symbols': []}

        # Stack sequences: (1, M, L, F)
        L, F = next(iter(symbol_sequences.values())).shape
        X = np.zeros((1, M, L, F), dtype=np.float32)
        mask = np.ones((1, M), dtype=np.float32)

        for i, symbol in enumerate(symbols):
            seq = symbol_sequences.get(symbol)
            if seq is not None and seq.shape == (L, F):
                X[0, i] = self._normalize(seq)
            else:
                mask[0, i] = 0.0

        # Inference
        with torch.no_grad():
            X_t = torch.from_numpy(X).to(self.device)
            mask_t = torch.from_numpy(mask).to(self.device)
            scores = self.model(X_t, mask_t).cpu().numpy()[0]  # (M,)

        # Build results
        symbol_scores = {sym: float(scores[i]) for i, sym in enumerate(symbols) if mask[0, i] > 0.5}
        rankings = sorted(symbol_scores.keys(), key=lambda s: symbol_scores[s], reverse=True)

        k_top = max(1, int(len(rankings) * self.top_pct))
        k_bottom = max(1, int(len(rankings) * self.bottom_pct))

        return {
            'scores': symbol_scores,
            'rankings': rankings,
            'long_symbols': rankings[:k_top],
            'short_symbols': rankings[-k_bottom:],
        }

    def _normalize(self, seq: np.ndarray) -> np.ndarray:
        """Apply same normalization as training."""
        from TRAINING.training_strategies.utils import _normalize_ohlcv_sequence
        return _normalize_ohlcv_sequence(seq, self.normalization)

    def _load_model(self, path: str, config: Dict):
        """Load trained model."""
        # Implementation depends on model family
        raise NotImplementedError("Implement based on model serialization format")
```

### 6. Model Metadata Extension

```python
# Update model_meta.json schema for CS ranking models

{
    "model_family": "LSTM",
    "model_checksum": "abc123...",

    # Standard fields
    "feature_list": null,  # Not used for CS ranking
    "interval_minutes": 5,

    # CS ranking specific
    "input_mode": "raw_sequence",
    "cross_sectional_ranking": {
        "enabled": true,
        "target_type": "cs_percentile",
        "loss_type": "pairwise",
        "sequence_length": 64,
        "normalization": "log_return",
        "training_metrics": {
            "best_ic": 0.045,
            "best_spread": 0.0012,
            "epochs_trained": 50
        }
    }
}
```

## Integration Checklist

### Config System
- [x] Add `cross_sectional_ranking` section to `pipeline.yaml`
- [x] Add experiment config: `CONFIG/experiments/cs_ranking_baseline.yaml`
- [x] Validate config schema in `config_loader.py` (works via get_cfg())

### intelligent_trainer.py
- [x] Add CS ranking mode detection (`is_cs_ranking_enabled()`)
- [x] Add stage skip logic (target ranking + feature selection)
- [x] Route to CS training function (passes `cs_ranking_config`)
- [x] Export helpers from package `__init__.py`

### Model Training
- [x] Add `cs_ranking_config` parameter to `train_models_for_interval_comprehensive()`
- [x] Log CS ranking mode when enabled
- [x] Create `train_cs_ranking_model()` in `TRAINING/training_strategies/execution/cs_ranking_trainer.py`
- [x] Integrate `CrossSectionalDataset` with `cs_collate_fn()` for batching
- [x] Integrate ranking losses via `get_ranking_loss()`
- [x] Integrate ranking metrics via `compute_ranking_metrics()`
- [x] Add `create_cs_model_metadata()` for model_meta.json (contracts-compliant)
- [ ] Wire CS ranking trainer into main training.py target loop

### Live Trading
- [ ] Create `CrossSectionalRankingPredictor` (inference - TBD)
- [ ] Update model loader for CS models (TBD)
- [ ] Update signal generation for rankings (TBD)

### Artifacts
- [x] Extend `model_meta.json` schema in INTEGRATION_CONTRACTS.md (v1.4)
- [x] Update `INTEGRATION_CONTRACTS.md` with consumer rules (Rule 7)
- [x] Add `create_cs_model_metadata()` for CS ranking fields

## Backward Compatibility

| Component | Impact | Migration |
|-----------|--------|-----------|
| Existing models | None | Continue working |
| Feature-based training | None | Default mode unchanged |
| Config files | None | New section is optional |
| Live trading | Low | Add CS predictor alongside existing |

## Definition of Done

- [ ] Config schema implemented and documented
- [ ] intelligent_trainer.py routes correctly based on config
- [ ] Model training works end-to-end for CS ranking
- [ ] Metrics computed and logged correctly
- [ ] Live trading predictor functional
- [ ] Integration tests passing
- [ ] Backward compatibility verified
- [ ] INTEGRATION_CONTRACTS.md updated

## Testing Strategy

### Unit Tests
```python
def test_cs_config_loading():
    """Config loads with all CS ranking options."""

def test_stage_skip_logic():
    """Target ranking and feature selection skipped when enabled."""

def test_training_loop_cs():
    """Full training loop completes with CS ranking."""
```

### Integration Tests
```python
def test_end_to_end_cs_ranking():
    """Full pipeline: data -> training -> inference."""

def test_backward_compat_standard_mode():
    """Standard mode unaffected by CS ranking additions."""
```

### Smoke Tests
```python
def test_cs_ranking_quick_run():
    """Quick CS ranking training with minimal data."""
```
