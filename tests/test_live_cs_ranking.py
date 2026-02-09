"""
Unit tests for Phase 4: Cross-Sectional Ranking Inference.

Tests:
- _percentile_rank(): correct ranking with ties and edge cases
- _rank_to_signal(): probit transform maps to z-score scale
- CrossSectionalRankingPredictor: CS detection, ranking, min universe size
- ModelLoader.get_cs_ranking_config(): metadata extraction
- Engine integration: CS predictions merge with pointwise predictions
- Backward compatibility: non-CS models unaffected
- Contract compliance: cross_sectional_ranking fields consumed correctly
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from LIVE_TRADING.models.loader import ModelLoader
from LIVE_TRADING.prediction.cs_ranking_predictor import (
    CrossSectionalRankingPredictor,
    _percentile_rank,
    _rank_to_signal,
)
from LIVE_TRADING.prediction.predictor import (
    AllPredictions,
    HorizonPredictions,
    ModelPrediction,
    MultiHorizonPredictor,
)


# ---------------------------------------------------------------------------
# Picklable dummy model
# ---------------------------------------------------------------------------

class _DummyModel:
    """Picklable dummy model for tests."""
    def __init__(self, value: float = 0.5):
        self._value = value

    def predict(self, X):
        return np.array([self._value])


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_run_root(
    tmp_path: Path,
    families: Dict[str, Dict[str, Any]] | None = None,
    target: str = "ret_5m",
    view: str = "CROSS_SECTIONAL",
) -> Path:
    """
    Create a run root with multiple families and metadata.

    Args:
        tmp_path: pytest tmp_path
        families: Dict[family_name, metadata_overrides]
        target: Target name
        view: View type
    """
    run_root = tmp_path / "run_root"

    base_meta = {
        "target": target,
        "feature_list": [f"feat_{i}" for i in range(10)],
        "n_features": 10,
        "metrics": {"mean_IC": 0.05},
        "model_checksum": None,
        "interval_minutes": 1.0,
    }

    if families is None:
        families = {"Ridge": {}}

    trained_families = list(families.keys())
    manifest = {"target_index": {target: {"status": "complete", "families_trained": trained_families}}}

    # Write manifest
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(json.dumps(manifest))

    for family, overrides in families.items():
        family_dir = run_root / "targets" / target / "models" / f"view={view}" / f"family={family}"
        family_dir.mkdir(parents=True, exist_ok=True)

        meta = {**base_meta, "family": family, **overrides}
        (family_dir / "model_meta.json").write_text(json.dumps(meta))

        # Determine model value from metadata
        # Use a value derived from family name for deterministic different scores
        val = hash(family) % 100 / 100.0
        (family_dir / "model.pkl").write_bytes(pickle.dumps(_DummyModel(val)))

    return run_root


def _make_prices(n_bars: int = 100) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame."""
    dates = pd.date_range("2026-01-01", periods=n_bars, freq="min")
    base = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.1, n_bars))
    return pd.DataFrame({
        "Open": base,
        "High": base + 0.5,
        "Low": base - 0.5,
        "Close": base + 0.2,
        "Volume": np.random.default_rng(42).integers(1000, 10000, n_bars),
    }, index=dates)


# ---------------------------------------------------------------------------
# _percentile_rank tests
# ---------------------------------------------------------------------------

class TestPercentileRank:

    def test_basic_ranking(self):
        """Values are ranked correctly."""
        values = np.array([10.0, 30.0, 20.0])
        ranks = _percentile_rank(values)
        # 10 → rank 1/3, 20 → rank 2/3, 30 → rank 3/3
        np.testing.assert_allclose(ranks, [1 / 3, 3 / 3, 2 / 3])

    def test_single_value(self):
        """Single value gets rank 1.0."""
        ranks = _percentile_rank(np.array([42.0]))
        assert ranks[0] == 1.0

    def test_tied_values(self):
        """Tied values get average rank."""
        values = np.array([1.0, 2.0, 2.0, 3.0])
        ranks = _percentile_rank(values)
        # 1→1/4, 2→(2+3)/2/4=2.5/4, 2→2.5/4, 3→4/4
        np.testing.assert_allclose(ranks, [0.25, 0.625, 0.625, 1.0])

    def test_empty(self):
        """Empty array returns empty."""
        ranks = _percentile_rank(np.array([]))
        assert len(ranks) == 0

    def test_descending_input(self):
        """Correctly ranks descending input."""
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ranks = _percentile_rank(values)
        expected = np.array([5 / 5, 4 / 5, 3 / 5, 2 / 5, 1 / 5])
        np.testing.assert_allclose(ranks, expected)


# ---------------------------------------------------------------------------
# _rank_to_signal tests
# ---------------------------------------------------------------------------

class TestRankToSignal:

    def test_neutral(self):
        """Rank 0.5 maps to ~0 signal (neutral)."""
        signal = _rank_to_signal(0.5)
        assert abs(signal) < 0.01

    def test_high_rank_positive(self):
        """High rank maps to positive signal."""
        signal = _rank_to_signal(0.95)
        assert signal > 1.5  # ~1.64

    def test_low_rank_negative(self):
        """Low rank maps to negative signal."""
        signal = _rank_to_signal(0.05)
        assert signal < -1.5  # ~-1.64

    def test_clipping(self):
        """Extreme ranks are clipped to [-3, 3]."""
        assert _rank_to_signal(0.0001) >= -3.0
        assert _rank_to_signal(0.9999) <= 3.0

    def test_monotonic(self):
        """Higher ranks always produce higher signals."""
        ranks = np.linspace(0.01, 0.99, 50)
        signals = [_rank_to_signal(r) for r in ranks]
        for i in range(1, len(signals)):
            assert signals[i] > signals[i - 1]


# ---------------------------------------------------------------------------
# ModelLoader.get_cs_ranking_config tests
# ---------------------------------------------------------------------------

class TestModelLoaderCSConfig:

    def test_cs_ranking_config_present(self, tmp_path):
        """CS ranking config extracted when enabled."""
        run_root = _make_run_root(tmp_path, families={
            "Ridge": {
                "cross_sectional_ranking": {
                    "enabled": True,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                    "training_metrics": {"best_ic": 0.08, "best_spread": 0.003, "epochs_trained": 25},
                },
            },
        })
        loader = ModelLoader(run_root, verify_checksums=False)
        config = loader.get_cs_ranking_config("ret_5m", "Ridge")
        assert config["enabled"] is True
        assert config["target_type"] == "cs_percentile"
        assert config["loss_type"] == "pairwise_logistic"

    def test_cs_ranking_config_absent(self, tmp_path):
        """Non-CS models return empty dict."""
        run_root = _make_run_root(tmp_path, families={"Ridge": {}})
        loader = ModelLoader(run_root, verify_checksums=False)
        config = loader.get_cs_ranking_config("ret_5m", "Ridge")
        assert config == {}

    def test_cs_ranking_config_disabled(self, tmp_path):
        """CS ranking with enabled=False returns empty dict."""
        run_root = _make_run_root(tmp_path, families={
            "Ridge": {
                "cross_sectional_ranking": {
                    "enabled": False,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                },
            },
        })
        loader = ModelLoader(run_root, verify_checksums=False)
        config = loader.get_cs_ranking_config("ret_5m", "Ridge")
        assert config == {}


# ---------------------------------------------------------------------------
# CrossSectionalRankingPredictor tests
# ---------------------------------------------------------------------------

class TestCSRankingPredictor:

    def _make_predictor(self, tmp_path, families):
        """Helper to create predictor with given families."""
        run_root = _make_run_root(tmp_path, families=families)
        loader = ModelLoader(run_root, verify_checksums=False)
        predictor = MultiHorizonPredictor(
            run_root=str(run_root), horizons=["5m"]
        )
        cs_pred = CrossSectionalRankingPredictor(
            loader=loader,
            engine=predictor.engine,
            predictor=predictor,
            min_universe_size=3,
        )
        return cs_pred

    def test_is_cs_ranking_model_true(self, tmp_path):
        """Detects CS ranking model from metadata."""
        cs_pred = self._make_predictor(tmp_path, {
            "Ridge": {
                "cross_sectional_ranking": {
                    "enabled": True,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                },
            },
        })
        assert cs_pred.is_cs_ranking_model("ret_5m", "Ridge") is True

    def test_is_cs_ranking_model_false(self, tmp_path):
        """Non-CS model detected correctly."""
        cs_pred = self._make_predictor(tmp_path, {"Ridge": {}})
        assert cs_pred.is_cs_ranking_model("ret_5m", "Ridge") is False

    def test_get_cs_families(self, tmp_path):
        """Only CS families returned."""
        cs_pred = self._make_predictor(tmp_path, {
            "Ridge": {
                "cross_sectional_ranking": {
                    "enabled": True,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                },
            },
            "ElasticNet": {},
        })
        cs_fams = cs_pred.get_cs_families("ret_5m")
        assert cs_fams == ["Ridge"]

    def test_get_ranking_config(self, tmp_path):
        """_get_ranking_config() returns contract-compliant config."""
        cs_pred = self._make_predictor(tmp_path, {
            "Ridge": {
                "cross_sectional_ranking": {
                    "enabled": True,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                    "training_metrics": {"best_ic": 0.08},
                },
            },
        })
        config = cs_pred._get_ranking_config("ret_5m", "Ridge")
        assert config["target_type"] == "cs_percentile"
        assert config["loss_type"] == "pairwise_logistic"


class TestCSRankingPrediction:
    """Tests that mock _get_raw_score to isolate ranking logic from FeatureBuilder."""

    # Deterministic raw scores per symbol (different values → different ranks)
    RAW_SCORES = {"AAPL": 0.1, "MSFT": 0.3, "GOOG": 0.5, "AMZN": 0.7, "META": 0.9}

    @staticmethod
    def _mock_raw_score(target, family, prices, symbol, data_timestamp):
        """Return deterministic raw score per symbol."""
        return TestCSRankingPrediction.RAW_SCORES.get(symbol)

    def test_predict_ranks_universe(self, tmp_path):
        """predict() produces ranked predictions for all symbols."""
        families = {
            "Ridge": {
                "cross_sectional_ranking": {
                    "enabled": True,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                },
            },
        }
        run_root = _make_run_root(tmp_path, families=families)
        loader = ModelLoader(run_root, verify_checksums=False)
        predictor = MultiHorizonPredictor(run_root=str(run_root), horizons=["5m"])
        cs_pred = CrossSectionalRankingPredictor(
            loader=loader,
            engine=predictor.engine,
            predictor=predictor,
            min_universe_size=3,
        )

        universe = {sym: _make_prices() for sym in ["AAPL", "MSFT", "GOOG", "AMZN", "META"]}
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        with patch.object(cs_pred, "_get_raw_score", side_effect=self._mock_raw_score):
            result = cs_pred.predict(
                target="ret_5m",
                universe=universe,
                horizons=["5m"],
                data_timestamp=ts,
            )

        # Should have predictions for all symbols
        assert len(result) == 5
        for sym in ["AAPL", "MSFT", "GOOG", "AMZN", "META"]:
            assert sym in result
            all_preds = result[sym]
            assert isinstance(all_preds, AllPredictions)
            assert "5m" in all_preds.horizons

            hp = all_preds.horizons["5m"]
            assert "Ridge" in hp.predictions

            mp = hp.predictions["Ridge"]
            # raw is the percentile rank (0, 1]
            assert 0.0 < mp.raw <= 1.0
            # standardized is the probit-transformed signal
            assert -3.0 <= mp.standardized <= 3.0
            # calibrated is signal * confidence
            assert isinstance(mp.calibrated, float)

        # Verify ranking order: META (0.9) > AMZN (0.7) > GOOG (0.5) > ...
        rank_meta = result["META"].horizons["5m"].predictions["Ridge"].raw
        rank_aapl = result["AAPL"].horizons["5m"].predictions["Ridge"].raw
        assert rank_meta > rank_aapl

    def test_min_universe_size_enforced(self, tmp_path):
        """Universe smaller than min_universe_size produces no CS predictions."""
        families = {
            "Ridge": {
                "cross_sectional_ranking": {
                    "enabled": True,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                },
            },
        }
        run_root = _make_run_root(tmp_path, families=families)
        loader = ModelLoader(run_root, verify_checksums=False)
        predictor = MultiHorizonPredictor(run_root=str(run_root), horizons=["5m"])
        cs_pred = CrossSectionalRankingPredictor(
            loader=loader,
            engine=predictor.engine,
            predictor=predictor,
            min_universe_size=10,  # Require 10 symbols
        )

        universe = {sym: _make_prices() for sym in ["AAPL", "MSFT", "GOOG"]}
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        with patch.object(cs_pred, "_get_raw_score", side_effect=self._mock_raw_score):
            result = cs_pred.predict(
                target="ret_5m", universe=universe, horizons=["5m"],
                data_timestamp=ts,
            )
        assert result == {}

    def test_no_cs_families_returns_empty(self, tmp_path):
        """Target with no CS families produces empty result."""
        run_root = _make_run_root(tmp_path, families={"Ridge": {}})
        loader = ModelLoader(run_root, verify_checksums=False)
        predictor = MultiHorizonPredictor(run_root=str(run_root), horizons=["5m"])
        cs_pred = CrossSectionalRankingPredictor(
            loader=loader,
            engine=predictor.engine,
            predictor=predictor,
        )

        universe = {sym: _make_prices() for sym in ["AAPL", "MSFT", "GOOG", "AMZN", "META"]}
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        result = cs_pred.predict(
            target="ret_5m", universe=universe, horizons=["5m"],
            data_timestamp=ts,
        )
        assert result == {}

    def test_ranks_sum_correctly(self, tmp_path):
        """Percentile ranks across symbols sum to expected value."""
        families = {
            "Ridge": {
                "cross_sectional_ranking": {
                    "enabled": True,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                },
            },
        }
        run_root = _make_run_root(tmp_path, families=families)
        loader = ModelLoader(run_root, verify_checksums=False)
        predictor = MultiHorizonPredictor(run_root=str(run_root), horizons=["5m"])
        cs_pred = CrossSectionalRankingPredictor(
            loader=loader,
            engine=predictor.engine,
            predictor=predictor,
            min_universe_size=3,
        )

        universe = {sym: _make_prices() for sym in ["AAPL", "MSFT", "GOOG", "AMZN", "META"]}
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        with patch.object(cs_pred, "_get_raw_score", side_effect=self._mock_raw_score):
            result = cs_pred.predict(
                target="ret_5m", universe=universe, horizons=["5m"],
                data_timestamp=ts,
            )

        # Raw values (percentile ranks) should sum to N*(N+1)/2N = (N+1)/2
        # For 5 symbols: ranks sum to 1/5 + 2/5 + 3/5 + 4/5 + 5/5 = 3.0
        raw_ranks = [
            result[sym].horizons["5m"].predictions["Ridge"].raw
            for sym in result
        ]
        np.testing.assert_allclose(sum(raw_ranks), 3.0, atol=0.01)

    def test_multiple_horizons(self, tmp_path):
        """CS predictions replicated across all requested horizons."""
        families = {
            "Ridge": {
                "cross_sectional_ranking": {
                    "enabled": True,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                },
            },
        }
        run_root = _make_run_root(tmp_path, families=families)
        loader = ModelLoader(run_root, verify_checksums=False)
        predictor = MultiHorizonPredictor(run_root=str(run_root), horizons=["5m", "15m"])
        cs_pred = CrossSectionalRankingPredictor(
            loader=loader,
            engine=predictor.engine,
            predictor=predictor,
            min_universe_size=3,
        )

        universe = {sym: _make_prices() for sym in ["AAPL", "MSFT", "GOOG"]}
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        with patch.object(cs_pred, "_get_raw_score", side_effect=self._mock_raw_score):
            result = cs_pred.predict(
                target="ret_5m", universe=universe, horizons=["5m", "15m"],
                data_timestamp=ts,
            )

        assert len(result) == 3
        for sym in result:
            assert "5m" in result[sym].horizons
            assert "15m" in result[sym].horizons
            # Same rank for both horizons (CS ranking is timestamp-level)
            r_5m = result[sym].horizons["5m"].predictions["Ridge"].raw
            r_15m = result[sym].horizons["15m"].predictions["Ridge"].raw
            assert r_5m == r_15m


class TestCSRankingBackwardCompat:

    def test_non_cs_models_unaffected(self, tmp_path):
        """Models without cross_sectional_ranking continue to work normally."""
        run_root = _make_run_root(tmp_path, families={"Ridge": {}})
        loader = ModelLoader(run_root, verify_checksums=False)
        predictor = MultiHorizonPredictor(run_root=str(run_root), horizons=["5m"])
        cs_pred = CrossSectionalRankingPredictor(
            loader=loader,
            engine=predictor.engine,
            predictor=predictor,
        )

        assert cs_pred.get_cs_families("ret_5m") == []
        assert cs_pred.is_cs_ranking_model("ret_5m", "Ridge") is False

    def test_mixed_cs_and_pointwise_families(self, tmp_path):
        """CS families ranked; non-CS families excluded from CS predictor."""
        families = {
            "Ridge": {},  # pointwise
            "ElasticNet": {
                "cross_sectional_ranking": {
                    "enabled": True,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                },
            },
        }
        run_root = _make_run_root(tmp_path, families=families)
        loader = ModelLoader(run_root, verify_checksums=False)
        predictor = MultiHorizonPredictor(run_root=str(run_root), horizons=["5m"])
        cs_pred = CrossSectionalRankingPredictor(
            loader=loader,
            engine=predictor.engine,
            predictor=predictor,
            min_universe_size=3,
        )

        assert cs_pred.get_cs_families("ret_5m") == ["ElasticNet"]

        raw_scores = {"AAPL": 0.1, "MSFT": 0.3, "GOOG": 0.5, "AMZN": 0.7, "META": 0.9}

        universe = {sym: _make_prices() for sym in raw_scores}
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        with patch.object(
            cs_pred, "_get_raw_score",
            side_effect=lambda t, f, p, s, ts: raw_scores.get(s),
        ):
            result = cs_pred.predict(
                target="ret_5m", universe=universe, horizons=["5m"],
                data_timestamp=ts,
            )

        # CS predictor only returns ElasticNet predictions
        assert len(result) == 5
        for sym in result:
            preds = result[sym].horizons["5m"].predictions
            assert "ElasticNet" in preds
            assert "Ridge" not in preds


class TestCSRankingContractFields:
    """Verify INTEGRATION_CONTRACTS.md v1.4 fields are consumed correctly."""

    def test_all_cs_ranking_fields_consumed(self, tmp_path):
        """All cross_sectional_ranking.* fields accessible via _get_ranking_config()."""
        families = {
            "Ridge": {
                "cross_sectional_ranking": {
                    "enabled": True,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                    "sequence_length": 64,
                    "normalization": "log_returns",
                    "training_metrics": {
                        "best_ic": 0.08,
                        "best_spread": 0.003,
                        "epochs_trained": 25,
                        "ic_ir": 0.5,
                        "ic_hit_rate": 0.55,
                        "turnover": 0.3,
                        "net_spread": 0.002,
                    },
                },
            },
        }
        run_root = _make_run_root(tmp_path, families=families)
        loader = ModelLoader(run_root, verify_checksums=False)
        predictor = MultiHorizonPredictor(run_root=str(run_root), horizons=["5m"])
        cs_pred = CrossSectionalRankingPredictor(
            loader=loader,
            engine=predictor.engine,
            predictor=predictor,
        )

        config = cs_pred._get_ranking_config("ret_5m", "Ridge")

        # Contract-required fields (INTEGRATION_CONTRACTS.md v1.4)
        assert config["enabled"] is True
        assert config["target_type"] == "cs_percentile"
        assert config["loss_type"] == "pairwise_logistic"
        assert config["sequence_length"] == 64
        assert config["normalization"] == "log_returns"

        metrics = config["training_metrics"]
        assert metrics["best_ic"] == 0.08
        assert metrics["best_spread"] == 0.003
        assert metrics["epochs_trained"] == 25
        assert metrics["ic_ir"] == 0.5

    def test_predictions_have_correct_structure(self, tmp_path):
        """CS predictions use standard ModelPrediction/AllPredictions types."""
        families = {
            "Ridge": {
                "cross_sectional_ranking": {
                    "enabled": True,
                    "target_type": "cs_percentile",
                    "loss_type": "pairwise_logistic",
                },
            },
        }
        run_root = _make_run_root(tmp_path, families=families)
        loader = ModelLoader(run_root, verify_checksums=False)
        predictor = MultiHorizonPredictor(run_root=str(run_root), horizons=["5m"])
        cs_pred = CrossSectionalRankingPredictor(
            loader=loader,
            engine=predictor.engine,
            predictor=predictor,
            min_universe_size=3,
        )

        raw_scores = {"AAPL": 0.1, "MSFT": 0.3, "GOOG": 0.5}
        universe = {sym: _make_prices() for sym in raw_scores}
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        with patch.object(
            cs_pred, "_get_raw_score",
            side_effect=lambda t, f, p, s, ts: raw_scores.get(s),
        ):
            result = cs_pred.predict(
                target="ret_5m", universe=universe, horizons=["5m"],
                data_timestamp=ts,
            )

        assert len(result) == 3
        for sym, all_preds in result.items():
            assert isinstance(all_preds, AllPredictions)
            assert all_preds.symbol == sym
            for h, hp in all_preds.horizons.items():
                assert isinstance(hp, HorizonPredictions)
                for family, mp in hp.predictions.items():
                    assert isinstance(mp, ModelPrediction)
                    assert mp.family == family
                    assert mp.horizon == h
                    # to_dict() should work (blending pipeline uses this)
                    d = mp.to_dict()
                    assert "raw" in d
                    assert "standardized" in d
                    assert "calibrated" in d
                    assert "confidence" in d
