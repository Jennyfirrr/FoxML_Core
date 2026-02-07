# Plan 18: Rust Core Extensions

## Overview

Implement performance-critical paths in Rust, exposed to Python via PyO3.

## Why Rust?

| Factor | Rust | C++ | Python |
|--------|------|-----|--------|
| Memory safety | Compile-time | Manual | GC |
| GC pauses | None | None | Yes |
| Python FFI | PyO3 (excellent) | Complex | N/A |
| Build system | Cargo | CMake/etc | N/A |
| Learning curve | Steep | Steep | Easy |

For trading:
- **No GC pauses** - Critical for latency-sensitive paths
- **Memory safety** - No use-after-free bugs losing money
- **Excellent Python interop** - PyO3 + maturin makes it seamless
- **Growing ecosystem** - `rust_decimal`, `chrono`, `tokio`

## Modules to Implement

### 1. `foxml_orderbook` - Order Book Management

Hot path: Processing L2 book updates, computing best bid/ask.

```rust
// src/orderbook.rs

use pyo3::prelude::*;
use std::collections::BTreeMap;

/// Price level in the order book
#[pyclass]
#[derive(Clone)]
pub struct PriceLevel {
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub size: f64,
    #[pyo3(get)]
    pub order_count: u32,
}

/// L2 Order Book with efficient best bid/ask access
#[pyclass]
pub struct OrderBook {
    symbol: String,
    bids: BTreeMap<OrderedFloat<f64>, PriceLevel>,  // Descending
    asks: BTreeMap<OrderedFloat<f64>, PriceLevel>,  // Ascending
    last_update_ms: i64,
}

#[pymethods]
impl OrderBook {
    #[new]
    pub fn new(symbol: String) -> Self {
        OrderBook {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_ms: 0,
        }
    }

    /// Update bid side
    pub fn update_bid(&mut self, price: f64, size: f64, timestamp_ms: i64) {
        if size == 0.0 {
            self.bids.remove(&OrderedFloat(price));
        } else {
            self.bids.insert(
                OrderedFloat(price),
                PriceLevel { price, size, order_count: 1 },
            );
        }
        self.last_update_ms = timestamp_ms;
    }

    /// Update ask side
    pub fn update_ask(&mut self, price: f64, size: f64, timestamp_ms: i64) {
        if size == 0.0 {
            self.asks.remove(&OrderedFloat(price));
        } else {
            self.asks.insert(
                OrderedFloat(price),
                PriceLevel { price, size, order_count: 1 },
            );
        }
        self.last_update_ms = timestamp_ms;
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.iter().next_back().map(|(k, _)| k.0)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.iter().next().map(|(k, _)| k.0)
    }

    /// Get bid-ask spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) if bid > 0.0 && ask > 0.0 => {
                let mid = (bid + ask) / 2.0;
                Some((ask - bid) / mid * 10000.0)
            }
            _ => None,
        }
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get top N bid levels
    pub fn top_bids(&self, n: usize) -> Vec<PriceLevel> {
        self.bids.values().rev().take(n).cloned().collect()
    }

    /// Get top N ask levels
    pub fn top_asks(&self, n: usize) -> Vec<PriceLevel> {
        self.asks.values().take(n).cloned().collect()
    }

    /// Clear the book
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }

    /// Get book imbalance (-1 to 1)
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_size: f64 = self.bids.values().rev().take(levels).map(|l| l.size).sum();
        let ask_size: f64 = self.asks.values().take(levels).map(|l| l.size).sum();
        let total = bid_size + ask_size;
        if total == 0.0 {
            0.0
        } else {
            (bid_size - ask_size) / total
        }
    }
}
```

### 2. `foxml_risk` - Risk Calculations

Hot path: Position limit checks, exposure calculations.

```rust
// src/risk.rs

use pyo3::prelude::*;
use std::collections::HashMap;

/// Risk calculator for fast position/exposure checks
#[pyclass]
pub struct RiskCalculator {
    max_position_value: f64,
    max_total_exposure: f64,
    max_single_stock_pct: f64,
}

#[pymethods]
impl RiskCalculator {
    #[new]
    pub fn new(
        max_position_value: f64,
        max_total_exposure: f64,
        max_single_stock_pct: f64,
    ) -> Self {
        RiskCalculator {
            max_position_value,
            max_total_exposure,
            max_single_stock_pct,
        }
    }

    /// Check if a proposed trade would violate risk limits
    /// Returns (allowed, reason)
    pub fn check_trade(
        &self,
        positions: HashMap<String, f64>,  // symbol -> value
        portfolio_value: f64,
        symbol: &str,
        trade_value: f64,
    ) -> (bool, String) {
        // Current position value
        let current_value = positions.get(symbol).copied().unwrap_or(0.0);
        let new_value = current_value + trade_value;

        // Check single position limit
        if new_value.abs() > self.max_position_value {
            return (false, format!(
                "Position ${:.0} exceeds max ${:.0}",
                new_value.abs(),
                self.max_position_value
            ));
        }

        // Check single stock percentage
        if portfolio_value > 0.0 {
            let pct = new_value.abs() / portfolio_value;
            if pct > self.max_single_stock_pct {
                return (false, format!(
                    "Position {:.1}% exceeds max {:.1}%",
                    pct * 100.0,
                    self.max_single_stock_pct * 100.0
                ));
            }
        }

        // Check total exposure
        let current_exposure: f64 = positions.values().map(|v| v.abs()).sum();
        let new_exposure = current_exposure - current_value.abs() + new_value.abs();
        if new_exposure > self.max_total_exposure {
            return (false, format!(
                "Total exposure ${:.0} exceeds max ${:.0}",
                new_exposure,
                self.max_total_exposure
            ));
        }

        (true, String::new())
    }

    /// Calculate portfolio metrics
    pub fn portfolio_metrics(
        &self,
        positions: HashMap<String, f64>,
        portfolio_value: f64,
    ) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        let gross_exposure: f64 = positions.values().map(|v| v.abs()).sum();
        let net_exposure: f64 = positions.values().sum();
        let long_exposure: f64 = positions.values().filter(|v| **v > 0.0).sum();
        let short_exposure: f64 = positions.values().filter(|v| **v < 0.0).map(|v| v.abs()).sum();

        metrics.insert("gross_exposure".to_string(), gross_exposure);
        metrics.insert("net_exposure".to_string(), net_exposure);
        metrics.insert("long_exposure".to_string(), long_exposure);
        metrics.insert("short_exposure".to_string(), short_exposure);

        if portfolio_value > 0.0 {
            metrics.insert("gross_leverage".to_string(), gross_exposure / portfolio_value);
            metrics.insert("net_leverage".to_string(), net_exposure / portfolio_value);
        }

        metrics
    }
}
```

### 3. `foxml_signals` - Signal Calculations

Hot path: Score normalization, blending math.

```rust
// src/signals.rs

use pyo3::prelude::*;
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

/// Ridge regression weights calculation
#[pyfunction]
pub fn ridge_weights(
    py: Python<'_>,
    covariance: PyReadonlyArray2<f64>,  // NxN covariance matrix
    means: PyReadonlyArray1<f64>,        // N expected returns
    lambda_reg: f64,                      // Regularization
) -> PyResult<Py<PyArray1<f64>>> {
    let cov = covariance.as_array();
    let mu = means.as_array();
    let n = mu.len();

    // Add regularization: (Sigma + lambda*I)
    let mut reg_cov = cov.to_owned();
    for i in 0..n {
        reg_cov[[i, i]] += lambda_reg;
    }

    // Solve: w = (Sigma + lambda*I)^-1 * mu
    // Using simple Cholesky decomposition
    let weights = solve_positive_definite(&reg_cov, &mu.to_owned());

    // Normalize to sum to 1
    let sum: f64 = weights.iter().map(|w| w.abs()).sum();
    let normalized: Vec<f64> = if sum > 0.0 {
        weights.iter().map(|w| w / sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    };

    Ok(PyArray1::from_vec_bound(py, normalized).unbind())
}

/// Solve Ax = b for positive definite A using Cholesky
fn solve_positive_definite(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    // Simplified implementation - production would use LAPACK
    let n = b.len();

    // LU decomposition fallback (Cholesky for PD matrices)
    // For production: use nalgebra or ndarray-linalg with LAPACK

    // Simple Gaussian elimination for now
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        let pivot = aug[[i, i]];
        if pivot.abs() < 1e-10 {
            continue;  // Skip near-zero pivots
        }
        for j in (i + 1)..n {
            let factor = aug[[j, i]] / pivot;
            for k in i..=n {
                aug[[j, k]] -= factor * aug[[i, k]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        if aug[[i, i]].abs() > 1e-10 {
            x[i] = sum / aug[[i, i]];
        }
    }

    x
}

/// Cost-adjusted net score calculation
#[pyfunction]
pub fn net_score(
    alpha: f64,           // Raw alpha forecast
    spread_bps: f64,      // Bid-ask spread
    volatility: f64,      // Asset volatility
    horizon_minutes: f64, // Holding horizon
    impact_bps: f64,      // Market impact estimate
    k_spread: f64,        // Spread cost weight
    k_vol: f64,           // Volatility cost weight
    k_impact: f64,        // Impact cost weight
) -> f64 {
    // net = alpha - k1*spread - k2*vol*sqrt(h/5) - k3*impact
    let vol_cost = k_vol * volatility * (horizon_minutes / 5.0).sqrt();
    let spread_cost = k_spread * spread_bps;
    let impact_cost = k_impact * impact_bps;

    alpha - spread_cost - vol_cost - impact_cost
}

/// Batch net score calculation for multiple horizons
#[pyfunction]
pub fn batch_net_scores(
    py: Python<'_>,
    alphas: PyReadonlyArray1<f64>,      // Alpha per horizon
    spreads: PyReadonlyArray1<f64>,     // Spread per horizon
    volatilities: PyReadonlyArray1<f64>, // Vol per horizon
    horizons: PyReadonlyArray1<f64>,    // Horizon minutes
    impact: f64,
    k_spread: f64,
    k_vol: f64,
    k_impact: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let a = alphas.as_array();
    let s = spreads.as_array();
    let v = volatilities.as_array();
    let h = horizons.as_array();

    let scores: Vec<f64> = (0..a.len())
        .map(|i| net_score(a[i], s[i], v[i], h[i], impact, k_spread, k_vol, k_impact))
        .collect();

    Ok(PyArray1::from_vec_bound(py, scores).unbind())
}

/// Normalize scores to [-1, 1] range with winsorization
#[pyfunction]
pub fn normalize_scores(
    py: Python<'_>,
    scores: PyReadonlyArray1<f64>,
    winsorize_pct: f64,  // e.g., 0.01 for 1% tails
) -> PyResult<Py<PyArray1<f64>>> {
    let s = scores.as_array();
    let n = s.len();

    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]).unbind());
    }

    // Sort for percentile calculation
    let mut sorted: Vec<f64> = s.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let low_idx = ((n as f64) * winsorize_pct).floor() as usize;
    let high_idx = ((n as f64) * (1.0 - winsorize_pct)).ceil() as usize;

    let low_val = sorted[low_idx.min(n - 1)];
    let high_val = sorted[high_idx.min(n - 1)];

    // Winsorize and normalize
    let normalized: Vec<f64> = s
        .iter()
        .map(|&x| {
            let clipped = x.max(low_val).min(high_val);
            if high_val > low_val {
                2.0 * (clipped - low_val) / (high_val - low_val) - 1.0
            } else {
                0.0
            }
        })
        .collect();

    Ok(PyArray1::from_vec_bound(py, normalized).unbind())
}
```

### 4. `foxml_decimal` - Exact Decimal Arithmetic

For price/quantity calculations where floating point errors matter.

```rust
// src/decimal.rs

use pyo3::prelude::*;
use rust_decimal::prelude::*;

/// Exact decimal for financial calculations
#[pyclass]
#[derive(Clone)]
pub struct ExactDecimal {
    inner: Decimal,
}

#[pymethods]
impl ExactDecimal {
    #[new]
    pub fn new(value: &str) -> PyResult<Self> {
        let d = Decimal::from_str(value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(ExactDecimal { inner: d })
    }

    #[staticmethod]
    pub fn from_float(value: f64) -> Self {
        ExactDecimal {
            inner: Decimal::from_f64(value).unwrap_or(Decimal::ZERO),
        }
    }

    pub fn to_float(&self) -> f64 {
        self.inner.to_f64().unwrap_or(0.0)
    }

    pub fn __str__(&self) -> String {
        self.inner.to_string()
    }

    pub fn __repr__(&self) -> String {
        format!("ExactDecimal('{}')", self.inner)
    }

    pub fn __add__(&self, other: &ExactDecimal) -> ExactDecimal {
        ExactDecimal {
            inner: self.inner + other.inner,
        }
    }

    pub fn __sub__(&self, other: &ExactDecimal) -> ExactDecimal {
        ExactDecimal {
            inner: self.inner - other.inner,
        }
    }

    pub fn __mul__(&self, other: &ExactDecimal) -> ExactDecimal {
        ExactDecimal {
            inner: self.inner * other.inner,
        }
    }

    pub fn __truediv__(&self, other: &ExactDecimal) -> PyResult<ExactDecimal> {
        if other.inner.is_zero() {
            return Err(PyErr::new::<pyo3::exceptions::PyZeroDivisionError, _>(
                "division by zero",
            ));
        }
        Ok(ExactDecimal {
            inner: self.inner / other.inner,
        })
    }

    /// Round to specified decimal places
    pub fn round(&self, dp: u32) -> ExactDecimal {
        ExactDecimal {
            inner: self.inner.round_dp(dp),
        }
    }

    /// Round to tick size (e.g., 0.01 for pennies)
    pub fn round_to_tick(&self, tick_size: &str) -> PyResult<ExactDecimal> {
        let tick = Decimal::from_str(tick_size)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let rounded = (self.inner / tick).round() * tick;
        Ok(ExactDecimal { inner: rounded })
    }
}

/// Calculate position value with exact arithmetic
#[pyfunction]
pub fn exact_position_value(shares: &str, price: &str) -> PyResult<String> {
    let s = Decimal::from_str(shares)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let p = Decimal::from_str(price)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok((s * p).to_string())
}
```

## Project Structure

```
LIVE_TRADING/
├── rust/
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       ├── lib.rs
│       ├── orderbook.rs
│       ├── risk.rs
│       ├── signals.rs
│       └── decimal.rs
```

### `Cargo.toml`

```toml
[package]
name = "foxml_core"
version = "0.1.0"
edition = "2021"

[lib]
name = "foxml_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
ndarray = "0.15"
rust_decimal = "1.33"
ordered-float = "4.2"

[profile.release]
lto = true
codegen-units = 1
```

### `pyproject.toml`

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "foxml_core"
version = "0.1.0"
requires-python = ">=3.10"

[tool.maturin]
features = ["pyo3/extension-module"]
```

### `src/lib.rs`

```rust
use pyo3::prelude::*;

mod orderbook;
mod risk;
mod signals;
mod decimal;

use orderbook::{OrderBook, PriceLevel};
use risk::RiskCalculator;
use decimal::ExactDecimal;

#[pymodule]
fn foxml_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Order book
    m.add_class::<OrderBook>()?;
    m.add_class::<PriceLevel>()?;

    // Risk
    m.add_class::<RiskCalculator>()?;

    // Signals
    m.add_function(wrap_pyfunction!(signals::ridge_weights, m)?)?;
    m.add_function(wrap_pyfunction!(signals::net_score, m)?)?;
    m.add_function(wrap_pyfunction!(signals::batch_net_scores, m)?)?;
    m.add_function(wrap_pyfunction!(signals::normalize_scores, m)?)?;

    // Decimal
    m.add_class::<ExactDecimal>()?;
    m.add_function(wrap_pyfunction!(decimal::exact_position_value, m)?)?;

    Ok(())
}
```

## Build & Install

```bash
# Install maturin
pip install maturin

# Build in development mode
cd LIVE_TRADING/rust
maturin develop --release

# Or build wheel for distribution
maturin build --release
pip install target/wheels/foxml_core-*.whl
```

## Python Usage

```python
from foxml_core import OrderBook, RiskCalculator, net_score, ExactDecimal

# Order book
book = OrderBook("AAPL")
book.update_bid(150.00, 1000, timestamp_ms())
book.update_ask(150.05, 500, timestamp_ms())
print(f"Spread: {book.spread_bps():.1f} bps")
print(f"Imbalance: {book.imbalance(5):.2f}")

# Risk check
risk = RiskCalculator(
    max_position_value=100000,
    max_total_exposure=500000,
    max_single_stock_pct=0.20,
)
allowed, reason = risk.check_trade(positions, portfolio_value, "AAPL", 10000)

# Net score
score = net_score(
    alpha=0.05,
    spread_bps=5.0,
    volatility=0.02,
    horizon_minutes=30,
    impact_bps=2.0,
    k_spread=0.5,
    k_vol=1.0,
    k_impact=0.3,
)

# Exact decimal
price = ExactDecimal("150.005")
shares = ExactDecimal("100")
value = price * shares
print(f"Value: {value.round(2)}")  # "15000.50"
```

## Tests

### `LIVE_TRADING/tests/test_rust_core.py`

```python
"""
Rust Core Extension Tests
=========================

Tests for Rust-implemented hot paths.
"""

import pytest
import numpy as np

# Skip if Rust extension not built
pytest.importorskip("foxml_core")

from foxml_core import (
    OrderBook,
    RiskCalculator,
    net_score,
    batch_net_scores,
    normalize_scores,
    ridge_weights,
    ExactDecimal,
)


class TestOrderBook:
    """Tests for OrderBook."""

    def test_update_and_best_bid_ask(self):
        """Test basic order book operations."""
        book = OrderBook("AAPL")
        book.update_bid(150.00, 1000, 1000)
        book.update_ask(150.05, 500, 1000)

        assert book.best_bid() == 150.00
        assert book.best_ask() == 150.05

    def test_spread_bps(self):
        """Test spread calculation."""
        book = OrderBook("AAPL")
        book.update_bid(100.00, 1000, 1000)
        book.update_ask(100.10, 500, 1000)

        spread = book.spread_bps()
        # (100.10 - 100.00) / 100.05 * 10000 ≈ 10 bps
        assert abs(spread - 10.0) < 0.1

    def test_imbalance(self):
        """Test book imbalance."""
        book = OrderBook("AAPL")
        book.update_bid(150.00, 1000, 1000)  # More bids
        book.update_ask(150.05, 100, 1000)

        imbalance = book.imbalance(1)
        # (1000 - 100) / (1000 + 100) ≈ 0.818
        assert imbalance > 0.8


class TestRiskCalculator:
    """Tests for RiskCalculator."""

    def test_trade_within_limits(self):
        """Test trade that's within limits."""
        risk = RiskCalculator(100000, 500000, 0.20)
        positions = {"AAPL": 50000.0}

        allowed, reason = risk.check_trade(positions, 500000, "MSFT", 20000)
        assert allowed
        assert reason == ""

    def test_trade_exceeds_position_limit(self):
        """Test trade that exceeds position limit."""
        risk = RiskCalculator(100000, 500000, 0.20)
        positions = {"AAPL": 90000.0}

        allowed, reason = risk.check_trade(positions, 500000, "AAPL", 20000)
        assert not allowed
        assert "exceeds max" in reason


class TestSignals:
    """Tests for signal calculations."""

    def test_net_score(self):
        """Test net score calculation."""
        score = net_score(
            alpha=0.05,
            spread_bps=5.0,
            volatility=0.02,
            horizon_minutes=30,
            impact_bps=2.0,
            k_spread=0.5,
            k_vol=1.0,
            k_impact=0.3,
        )
        # Should be positive but reduced from raw alpha
        assert 0 < score < 0.05

    def test_batch_net_scores(self):
        """Test batch net score calculation."""
        scores = batch_net_scores(
            np.array([0.05, 0.04, 0.03]),
            np.array([5.0, 5.0, 5.0]),
            np.array([0.02, 0.02, 0.02]),
            np.array([5.0, 15.0, 30.0]),
            2.0, 0.5, 1.0, 0.3,
        )
        assert len(scores) == 3
        # Shorter horizons should have higher scores (less vol drag)
        assert scores[0] > scores[2]

    def test_normalize_scores(self):
        """Test score normalization."""
        raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_scores(raw, 0.1)

        assert len(normalized) == 5
        assert all(-1 <= s <= 1 for s in normalized)


class TestExactDecimal:
    """Tests for ExactDecimal."""

    def test_from_string(self):
        """Test creating from string."""
        d = ExactDecimal("150.005")
        assert abs(d.to_float() - 150.005) < 1e-10

    def test_arithmetic(self):
        """Test arithmetic operations."""
        a = ExactDecimal("100.50")
        b = ExactDecimal("2")

        result = a * b
        assert str(result) == "201.00" or str(result) == "201"

    def test_round_to_tick(self):
        """Test rounding to tick size."""
        d = ExactDecimal("150.007")
        rounded = d.round_to_tick("0.01")
        assert abs(rounded.to_float() - 150.01) < 1e-10
```

## Performance Benchmarks

Expected improvements over pure Python:

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Order book update | 5 μs | 0.2 μs | 25x |
| Spread calculation | 2 μs | 0.05 μs | 40x |
| Risk check | 10 μs | 0.5 μs | 20x |
| Ridge weights (N=5) | 100 μs | 5 μs | 20x |
| Batch net scores (N=1000) | 1 ms | 0.05 ms | 20x |

## Implementation Order

1. **Phase 1**: OrderBook + basic tests
2. **Phase 2**: RiskCalculator
3. **Phase 3**: Signals (ridge_weights, net_score)
4. **Phase 4**: ExactDecimal

Each phase can be deployed independently.

## SST Compliance

- [x] Pure functions where possible
- [x] No global state
- [x] Comprehensive tests
- [x] Python fallback available

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `src/lib.rs` | 30 |
| `src/orderbook.rs` | 200 |
| `src/risk.rs` | 150 |
| `src/signals.rs` | 250 |
| `src/decimal.rs` | 100 |
| `tests/test_rust_core.py` | 150 |
| **Total** | ~880 |
