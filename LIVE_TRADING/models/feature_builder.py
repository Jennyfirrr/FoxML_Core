"""
Feature Builder
===============

Builds features from market data matching the training feature set.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.exceptions import FeatureBuildError

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Builds features from market data for model inference.

    Must produce features in the same order as training.

    INTERVAL AWARENESS (Phase 11):
    - Features are computed assuming daily (1d) bar data by default
    - The interval_minutes parameter tracks what interval is being used
    - For multi-interval support, callers should verify interval matches training
    """

    def __init__(
        self,
        feature_list: List[str],
        interval_minutes: float = 1440.0,  # Default: 1 day = 1440 minutes
    ):
        """
        Initialize feature builder.

        Args:
            feature_list: Ordered list of feature names (from model metadata)
            interval_minutes: Data bar interval in minutes (default: 1440.0 = 1 day).
                             Used for provenance tracking and future interval-aware features.
        """
        self.feature_list = feature_list
        self.interval_minutes = interval_minutes
        self._feature_funcs = self._register_feature_funcs()

        # Phase 11: Log interval for provenance tracking
        if interval_minutes != 1440.0:
            logger.info(
                f"FeatureBuilder initialized with non-daily interval: {interval_minutes} minutes"
            )

    def _register_feature_funcs(self) -> Dict[str, Callable[[pd.DataFrame], float]]:
        """Register feature computation functions."""
        return {
            # Returns
            "ret_1d": self._calc_ret_1d,
            "ret_5d": self._calc_ret_5d,
            "ret_10d": self._calc_ret_10d,
            "ret_20d": self._calc_ret_20d,
            "ret_60d": self._calc_ret_60d,
            # Volatility
            "vol_5d": self._calc_vol_5d,
            "vol_10d": self._calc_vol_10d,
            "vol_20d": self._calc_vol_20d,
            "vol_60d": self._calc_vol_60d,
            # Technical indicators
            "rsi_14": self._calc_rsi,
            "rsi_7": lambda p: self._calc_rsi_period(p, 7),
            "rsi_21": lambda p: self._calc_rsi_period(p, 21),
            "ma_ratio_10": self._calc_ma_ratio_10,
            "ma_ratio_20": self._calc_ma_ratio_20,
            "ma_ratio_50": self._calc_ma_ratio_50,
            "bb_position": self._calc_bb_position,
            "bb_width": self._calc_bb_width,
            # MACD
            "macd": self._calc_macd,
            "macd_signal": self._calc_macd_signal,
            "macd_hist": self._calc_macd_hist,
            # Momentum
            "mom_5d": self._calc_mom_5d,
            "mom_10d": self._calc_mom_10d,
            "mom_20d": self._calc_mom_20d,
            # Range-based
            "atr_14": self._calc_atr_14,
            "high_low_range": self._calc_high_low_range,
            # Volume
            "volume_ma_ratio": self._calc_volume_ma_ratio,
            "volume_change": self._calc_volume_change,
            # Price position
            "close_to_high_52w": self._calc_close_to_high_52w,
            "close_to_low_52w": self._calc_close_to_low_52w,
        }

    def build_features(
        self,
        prices: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> np.ndarray:
        """
        Build feature array from price data.

        Args:
            prices: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
            symbol: Optional symbol for logging

        Returns:
            Feature array matching feature_list order

        Raises:
            FeatureBuildError: If feature computation fails
        """
        if len(prices) == 0:
            raise FeatureBuildError(symbol or "unknown", "Empty price data")

        features = []

        for feat_name in self.feature_list:
            try:
                if feat_name in self._feature_funcs:
                    value = self._feature_funcs[feat_name](prices)
                else:
                    # Unknown feature - use NaN
                    logger.warning(f"Unknown feature: {feat_name}")
                    value = np.nan

                features.append(value)
            except Exception as e:
                logger.warning(f"Failed to compute {feat_name}: {e}")
                features.append(np.nan)

        result = np.array(features, dtype=np.float32)

        # Log warning if any features are NaN
        nan_count = np.isnan(result).sum()
        if nan_count > 0:
            logger.warning(
                f"Feature build for {symbol}: {nan_count}/{len(features)} NaN values"
            )

        return result

    # =========================================================================
    # Return features
    # =========================================================================

    def _calc_ret_1d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change().iloc[-1])

    def _calc_ret_5d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change(5).iloc[-1])

    def _calc_ret_10d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change(10).iloc[-1])

    def _calc_ret_20d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change(20).iloc[-1])

    def _calc_ret_60d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change(60).iloc[-1])

    # =========================================================================
    # Volatility features
    # =========================================================================

    def _calc_vol_5d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change().rolling(5).std().iloc[-1])

    def _calc_vol_10d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change().rolling(10).std().iloc[-1])

    def _calc_vol_20d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change().rolling(20).std().iloc[-1])

    def _calc_vol_60d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.pct_change().rolling(60).std().iloc[-1])

    # =========================================================================
    # RSI features
    # =========================================================================

    def _calc_rsi(self, prices: pd.DataFrame, period: int = 14) -> float:
        return self._calc_rsi_period(prices, period)

    def _calc_rsi_period(self, prices: pd.DataFrame, period: int) -> float:
        close = prices["Close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    # =========================================================================
    # Moving average ratio features
    # =========================================================================

    def _calc_ma_ratio_10(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        ma = close.rolling(10).mean()
        return float((close.iloc[-1] / ma.iloc[-1]) - 1)

    def _calc_ma_ratio_20(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        ma = close.rolling(20).mean()
        return float((close.iloc[-1] / ma.iloc[-1]) - 1)

    def _calc_ma_ratio_50(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        ma = close.rolling(50).mean()
        return float((close.iloc[-1] / ma.iloc[-1]) - 1)

    # =========================================================================
    # Bollinger Band features
    # =========================================================================

    def _calc_bb_position(self, prices: pd.DataFrame, period: int = 20) -> float:
        close = prices["Close"]
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        band_width = upper.iloc[-1] - lower.iloc[-1]
        if band_width == 0:
            return 0.5
        position = (close.iloc[-1] - lower.iloc[-1]) / band_width
        return float(position)

    def _calc_bb_width(self, prices: pd.DataFrame, period: int = 20) -> float:
        close = prices["Close"]
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        width = (4 * std / ma).iloc[-1]  # Normalized width
        return float(width)

    # =========================================================================
    # MACD features
    # =========================================================================

    def _calc_macd(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        return float(macd.iloc[-1])

    def _calc_macd_signal(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return float(signal.iloc[-1])

    def _calc_macd_hist(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return float(hist.iloc[-1])

    # =========================================================================
    # Momentum features
    # =========================================================================

    def _calc_mom_5d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.iloc[-1] - close.iloc[-6])

    def _calc_mom_10d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.iloc[-1] - close.iloc[-11])

    def _calc_mom_20d(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        return float(close.iloc[-1] - close.iloc[-21])

    # =========================================================================
    # ATR and range features
    # =========================================================================

    def _calc_atr_14(self, prices: pd.DataFrame) -> float:
        high = prices["High"]
        low = prices["Low"]
        close = prices["Close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        return float(atr.iloc[-1])

    def _calc_high_low_range(self, prices: pd.DataFrame) -> float:
        high = prices["High"]
        low = prices["Low"]
        close = prices["Close"]
        return float((high.iloc[-1] - low.iloc[-1]) / close.iloc[-1])

    # =========================================================================
    # Volume features
    # =========================================================================

    def _calc_volume_ma_ratio(self, prices: pd.DataFrame) -> float:
        if "Volume" not in prices.columns:
            return np.nan
        volume = prices["Volume"]
        ma = volume.rolling(20).mean()
        if ma.iloc[-1] == 0:
            return np.nan
        return float(volume.iloc[-1] / ma.iloc[-1])

    def _calc_volume_change(self, prices: pd.DataFrame) -> float:
        if "Volume" not in prices.columns:
            return np.nan
        volume = prices["Volume"]
        return float(volume.pct_change().iloc[-1])

    # =========================================================================
    # 52-week features
    # =========================================================================

    def _calc_close_to_high_52w(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        high_52w = close.rolling(252, min_periods=20).max()
        return float(close.iloc[-1] / high_52w.iloc[-1])

    def _calc_close_to_low_52w(self, prices: pd.DataFrame) -> float:
        close = prices["Close"]
        low_52w = close.rolling(252, min_periods=20).min()
        if low_52w.iloc[-1] == 0:
            return np.nan
        return float(close.iloc[-1] / low_52w.iloc[-1])


def build_features_from_prices(
    prices: pd.DataFrame,
    feature_list: List[str],
    symbol: Optional[str] = None,
    interval_minutes: float = 1440.0,
) -> np.ndarray:
    """
    Convenience function to build features.

    Args:
        prices: OHLCV DataFrame
        feature_list: Ordered feature names
        symbol: Optional symbol for logging
        interval_minutes: Data bar interval in minutes (default: 1440.0 = 1 day)

    Returns:
        Feature array
    """
    builder = FeatureBuilder(feature_list, interval_minutes=interval_minutes)
    return builder.build_features(prices, symbol)
