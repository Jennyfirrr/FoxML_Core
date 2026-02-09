# Phase 0: Barrier Gate Quick Fix

**Status**: Ready for implementation
**Parent**: `live-trading-inference-master.md`
**Scope**: 2 files, ~20 lines
**Blocking**: Nothing (standalone)

## Problem

`LIVE_TRADING/engine/trading_engine.py:1295,1309` calls `self.predictor.predict_single_target()` which does not exist on `MultiHorizonPredictor`. The class has:
- `predict_all_horizons(target, prices, symbol, ...)` → `AllPredictions`
- `predict_single_horizon(target, horizon, prices, symbol, ...)` → `HorizonPredictions`

The barrier gate wants a single prediction for a specific target (e.g., `will_peak_5m`). This is caught by a bare `except Exception` which logs at DEBUG level — meaning **barrier gates are silently non-functional**.

## Fix

### Option A: Add `predict_single_target()` method (RECOMMENDED)

Add a convenience method to `MultiHorizonPredictor` that:
1. Calls `predict_single_horizon()` with the first available horizon
2. Returns the blended/best prediction from that horizon
3. Returns `None` if no models available for the target

```python
# LIVE_TRADING/prediction/predictor.py

def predict_single_target(
    self,
    target: str,
    prices: pd.DataFrame,
    symbol: str,
    data_timestamp: datetime | None = None,
) -> Optional[ModelPrediction]:
    """
    Get a single prediction for a target (first available horizon + family).

    Used by barrier gate for peak/valley predictions.
    """
    if data_timestamp is None:
        data_timestamp = datetime.now(timezone.utc)

    # Use first configured horizon
    horizon = self.horizons[0] if self.horizons else "5m"

    available_families = self.loader.list_available_families(target)
    if not available_families:
        return None

    # Try first available family
    for family in available_families:
        try:
            pred = self._predict_single(
                target=target,
                horizon=horizon,
                family=family,
                prices=prices,
                symbol=symbol,
                data_timestamp=data_timestamp,
                adv=float("inf"),
                planned_dollars=0.0,
            )
            if pred is not None:
                return pred
        except Exception as e:
            logger.debug(f"predict_single_target failed for {family}/{target}: {e}")
            continue

    return None
```

### Barrier gate call site (no changes needed)

The existing call sites at `trading_engine.py:1295,1309` already handle `None` return and `AttributeError` via try/except. Once the method exists, they'll work.

## Verification

- [ ] Method exists on `MultiHorizonPredictor`
- [ ] Returns `ModelPrediction` with `.alpha` attribute (used by barrier gate sigmoid)
- [ ] Returns `None` when no model available for barrier target
- [ ] `_get_barrier_predictions()` no longer falls through to except block

## Files Changed

1. `LIVE_TRADING/prediction/predictor.py` — add `predict_single_target()` method
2. No changes to `trading_engine.py` (call sites are already correct)
