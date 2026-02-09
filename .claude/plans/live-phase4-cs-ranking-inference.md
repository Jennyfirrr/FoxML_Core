# Phase 4: Cross-Sectional Ranking Inference

**Status**: ✅ COMPLETE
**Parent**: `live-trading-inference-master.md`
**Scope**: 1 new file + modifications, ~250 lines
**Depends on**: Phase 1 (input mode awareness)
**Independent of**: Phases 2-3 (raw OHLCV inference)

## Problem

Models trained with cross-sectional ranking (pairwise logistic loss, cs_percentile targets) output **relative** scores — they rank symbols against each other, not absolute return forecasts. The current `MultiHorizonPredictor` treats all predictions as absolute, feeding them through z-score standardization individually per symbol.

For CS-trained models:
- A prediction of 0.8 means "this symbol is near the top of the cross-section"
- A prediction of 0.2 means "this symbol is near the bottom"
- The values only have meaning relative to other symbols at the same timestamp

### Contract Fields (from INTEGRATION_CONTRACTS.md v1.3)

```json
"cross_sectional_ranking": {
    "enabled": true,
    "target_type": "cs_percentile",
    "loss_type": "pairwise_logistic",
    "sequence_length": 64,
    "normalization": "log_returns",
    "training_metrics": {
        "best_ic": 0.08,
        "best_spread": 0.003,
        "epochs_trained": 25
    }
}
```

## Architecture

```
All symbols in universe
    ↓
For each symbol: run model → raw_score
    ↓
Collect all raw_scores at timestamp
    ↓
CrossSectionalRankingPredictor:
    1. Rank symbols by score → percentile rank
    2. Compute spread (long top / short bottom)
    3. Map percentile rank → alpha signal
    ↓
Feed ranked alphas to blending/arbitration (existing pipeline)
```

## Implementation

### 1. New file: `LIVE_TRADING/prediction/cs_ranking_predictor.py`

```python
class CrossSectionalRankingPredictor:
    """
    Handles inference for models trained with cross-sectional ranking loss.

    Unlike pointwise models that produce absolute return forecasts,
    CS ranking models produce relative scores. This class:
    1. Collects raw predictions for all symbols
    2. Ranks them cross-sectionally
    3. Converts ranks to alpha signals
    """

    def __init__(self, loader, engine, horizons, config):
        self.loader = loader
        self.engine = engine
        self.horizons = horizons
        self.config = config

    def predict_universe(
        self,
        target: str,
        universe: Dict[str, pd.DataFrame],  # symbol → prices
        data_timestamp: datetime,
    ) -> Dict[str, AllPredictions]:
        """
        Generate ranked predictions for all symbols in universe.

        Steps:
        1. Get raw score per symbol per family
        2. Rank cross-sectionally per family
        3. Convert to alpha signals
        """
        # Collect raw scores
        raw_scores = {}  # symbol → {family → raw_score}
        for symbol, prices in sorted_items(universe):
            raw_scores[symbol] = self._predict_all_families(target, prices, symbol, data_timestamp)

        # Rank cross-sectionally per family
        ranked = self._rank_cross_sectionally(raw_scores)

        # Convert to AllPredictions per symbol
        return self._build_predictions(ranked, data_timestamp)

    def _rank_cross_sectionally(
        self,
        raw_scores: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Rank symbols per family, return percentile ranks [0, 1]."""
        # Get all families
        all_families = set()
        for scores in raw_scores.values():
            all_families.update(scores.keys())

        ranked = {sym: {} for sym in raw_scores}

        for family in sorted(all_families):
            # Collect scores for this family
            family_scores = {
                sym: scores.get(family)
                for sym, scores in raw_scores.items()
                if scores.get(family) is not None
            }

            if len(family_scores) < 2:
                continue

            # Rank via scipy or numpy
            symbols = sorted(family_scores.keys())
            values = np.array([family_scores[s] for s in symbols])
            ranks = scipy.stats.rankdata(values) / len(values)  # [0, 1]

            for sym, rank in zip(symbols, ranks):
                ranked[sym][family] = rank

        return ranked
```

### 2. Modifications to `LIVE_TRADING/prediction/predictor.py`

Detection of CS ranking models:

```python
def _predict_single(self, target, horizon, family, prices, symbol, ...):
    # Check for CS ranking
    _, metadata = self.loader.load_model(target, family)
    cs_config = metadata.get("cross_sectional_ranking")

    if cs_config and cs_config.get("enabled"):
        # CS ranking predictions must go through CrossSectionalRankingPredictor
        # Single-symbol prediction returns raw score (not yet ranked)
        logger.debug(f"CS ranking model {family}/{target}: returning raw score for later ranking")
        # ... compute raw score, mark as needs_ranking=True ...
```

### 3. Modifications to `LIVE_TRADING/engine/trading_engine.py`

The trading engine processes symbols individually in `_process_symbol()`. For CS ranking, it needs to:
1. Collect raw predictions for all symbols first
2. Then rank cross-sectionally
3. Then proceed with blending/arbitration per symbol

This requires a change to the cycle processing loop — predict all symbols, then rank, then trade.

## Open Questions

1. **Cycle ordering**: Currently symbols are processed independently. CS ranking needs all predictions before any trading decision. Should we add a "prediction phase" before "decision phase"?

2. **Mixed models**: What if a target has both pointwise and CS models? Should ranking only apply to CS families?

3. **Minimum universe size**: CS ranking with < 5 symbols is unreliable. Should we fall back to pointwise for small universes?

4. **Blending integration**: After cross-sectional ranking, how do ranked signals flow into the horizon blender? The blender expects z-scored predictions, not percentile ranks.

## Verification

- [x] CS ranking detected from model metadata
- [x] Raw scores collected for all symbols before ranking
- [x] Percentile ranks computed correctly
- [x] Ranked signals feed into blending pipeline (probit transform → z-score scale)
- [x] Non-CS models in same target unaffected
- [x] Minimum universe size enforced

## Files Changed

1. `LIVE_TRADING/prediction/cs_ranking_predictor.py` — NEW
2. `LIVE_TRADING/prediction/predictor.py` — CS detection in `_predict_single()`
3. `LIVE_TRADING/engine/trading_engine.py` — cycle processing changes
