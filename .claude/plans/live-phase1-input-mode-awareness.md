# Phase 1: Input Mode Awareness

**Status**: Ready for implementation
**Parent**: `live-trading-inference-master.md`
**Scope**: 3 files, ~60 lines
**Depends on**: Phase 0 (barrier gate fix)
**Blocks**: Phase 2 (raw OHLCV inference)

## Goal

Make every LIVE_TRADING inference component aware of `input_mode` from `model_meta.json`. This phase adds detection and routing — no new inference logic yet. Raw OHLCV models will raise `NotImplementedError` until Phase 2.

## Changes

### 1. `LIVE_TRADING/models/loader.py`

#### a) New method: `get_input_mode()`

```python
def get_input_mode(
    self,
    target: str,
    family: str,
    view: str = "CROSS_SECTIONAL",
) -> str:
    """
    Get the input mode for a model.

    CONTRACT: INTEGRATION_CONTRACTS.md v1.3
    - "features" (default): Traditional feature-based input
    - "raw_sequence": Raw OHLCV bar sequences

    Returns "features" for models without input_mode field (backward compat).
    """
    _, metadata = self.load_model(target, family, view)
    return metadata.get("input_mode", "features")
```

#### b) Fix `get_feature_list()` — suppress false warning for raw_sequence

Current code at line 377 warns "No feature list found" for raw_sequence models where `feature_list=[]` is correct.

```python
def get_feature_list(self, target, family, view="CROSS_SECTIONAL"):
    _, metadata = self.load_model(target, family, view)

    # Raw sequence models have empty feature_list by contract
    input_mode = metadata.get("input_mode", "features")
    if input_mode == "raw_sequence":
        return []  # Empty is correct — no warning

    feature_list = metadata.get("feature_list")
    if feature_list is not None:
        return list(feature_list)
    # ... existing legacy fallback ...
```

#### c) New method: `get_sequence_config()`

```python
def get_sequence_config(
    self,
    target: str,
    family: str,
    view: str = "CROSS_SECTIONAL",
) -> Dict[str, Any]:
    """
    Get sequence configuration for raw_sequence models.

    Returns:
        Dict with sequence_length, sequence_channels, sequence_normalization.
        Empty dict for feature-based models.
    """
    _, metadata = self.load_model(target, family, view)
    if metadata.get("input_mode", "features") != "raw_sequence":
        return {}

    return {
        "sequence_length": metadata.get("sequence_length", 64),
        "sequence_channels": metadata.get("sequence_channels",
            ["open", "high", "low", "close", "volume"]),
        "sequence_normalization": metadata.get("sequence_normalization", "returns"),
    }
```

### 2. `LIVE_TRADING/models/inference.py`

#### a) Fix `_init_sequential_buffer()` — use sequence_channels for raw models

```python
def _init_sequential_buffer(self, target, family, metadata):
    seq_length = metadata.get("sequence_length", 20)
    input_mode = metadata.get("input_mode", "features")

    if input_mode == "raw_sequence":
        # Raw OHLCV: F = number of channels (typically 5)
        channels = metadata.get("sequence_channels",
            ["open", "high", "low", "close", "volume"])
        n_features = len(channels)
    else:
        # Feature-based: F = number of features
        feature_list = (metadata.get("feature_list")
                       or metadata.get("features")
                       or metadata.get("feature_names") or [])
        n_features = len(feature_list)

    if n_features == 0:
        logger.warning(f"No features/channels for {target}:{family}, skipping buffer")
        return

    buffer_key = f"{target}:{family}"
    from CONFIG.config_loader import get_cfg
    ttl_seconds = get_cfg("pipeline.training.sequential.live.ttl_seconds", default=300.0)
    self._seq_buffers[buffer_key] = SeqBufferManager(
        T=seq_length, F=n_features, ttl_seconds=ttl_seconds,
    )
    logger.debug(f"Buffer {buffer_key}: T={seq_length}, F={n_features}, mode={input_mode}")
```

#### b) Store input_mode in metadata cache

The `predict()` method already caches `self._metadata[cache_key]`. No change needed — metadata already contains `input_mode`. Just add the routing:

```python
def predict(self, target, family, features, symbol="default", view="CROSS_SECTIONAL"):
    # ... existing load/cache code ...
    model = self._models[cache_key]
    metadata = self._metadata[cache_key]

    input_mode = metadata.get("input_mode", "features")

    try:
        if input_mode == "raw_sequence":
            if family in SEQUENTIAL_FAMILIES:
                return self._predict_raw_sequential(
                    model, features, target, family, symbol, metadata
                )
            else:
                raise InferenceError(family, symbol,
                    f"raw_sequence mode only supported for sequential families, got {family}")

        # Existing feature-based paths
        if family in TREE_FAMILIES:
            return self._predict_tree(model, features, family)
        elif family in SEQUENTIAL_FAMILIES:
            return self._predict_sequential(model, features, target, family, symbol)
        elif family in TF_FAMILIES:
            return self._predict_keras(model, features)
        else:
            return self._predict_generic(model, features, family, symbol)
    # ... existing error handling ...
```

#### c) Stub `_predict_raw_sequential()` — raises until Phase 2

```python
def _predict_raw_sequential(self, model, ohlcv_row, target, family, symbol, metadata):
    """Predict with raw OHLCV sequential model. Implemented in Phase 2."""
    raise NotImplementedError(
        f"Raw OHLCV inference not yet implemented for {family}/{target}. "
        f"See .claude/plans/live-phase2-raw-ohlcv-inference.md"
    )
```

### 3. `LIVE_TRADING/prediction/predictor.py`

#### a) Branch `_predict_single()` on input_mode

```python
def _predict_single(self, target, horizon, family, prices, symbol, data_timestamp, adv, planned_dollars):
    """Generate single model prediction."""

    # Check input mode
    input_mode = self.loader.get_input_mode(target, family)

    if input_mode == "raw_sequence":
        # Phase 2: prepare raw OHLCV instead of building features
        features = self._prepare_raw_sequence(prices, target, family)
    else:
        # Existing: build computed features
        builder = self._get_feature_builder(target, family)
        features = builder.build_features(prices, symbol)

    if features is None or np.any(np.isnan(features)):
        logger.warning(f"Invalid features for {family}/{symbol}")
        return None

    # ... rest unchanged (inference, standardize, confidence) ...
```

#### b) Stub `_prepare_raw_sequence()` — raises until Phase 2

```python
def _prepare_raw_sequence(self, prices, target, family):
    """Prepare raw OHLCV sequence for inference. Implemented in Phase 2."""
    raise NotImplementedError(
        f"Raw OHLCV preparation not yet implemented for {family}/{target}. "
        f"See .claude/plans/live-phase2-raw-ohlcv-inference.md"
    )
```

## Verification

- [ ] `get_input_mode()` returns `"features"` for models without `input_mode` field
- [ ] `get_input_mode()` returns `"raw_sequence"` for raw sequence models
- [ ] `get_feature_list()` returns `[]` without warning for raw_sequence models
- [ ] Buffer init uses `sequence_channels` for raw models (F=5)
- [ ] `predict()` routes raw_sequence to `_predict_raw_sequential()`
- [ ] Existing feature-based models still work identically (no behavior change)
- [ ] `NotImplementedError` raised with helpful message for raw models (until Phase 2)

## Files Changed

1. `LIVE_TRADING/models/loader.py` — `get_input_mode()`, `get_sequence_config()`, fix `get_feature_list()`
2. `LIVE_TRADING/models/inference.py` — fix `_init_sequential_buffer()`, add routing in `predict()`, stub `_predict_raw_sequential()`
3. `LIVE_TRADING/prediction/predictor.py` — branch `_predict_single()`, stub `_prepare_raw_sequence()`
