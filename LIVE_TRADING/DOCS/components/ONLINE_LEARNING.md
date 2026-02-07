# Online Learning

This document describes the online learning capabilities in the LIVE_TRADING module, particularly the Exp3-IX multi-armed bandit algorithm for adaptive weight optimization.

---

## Overview

The LIVE_TRADING module implements online learning to adapt model weights based on actual trading performance. This allows the system to:

1. **Learn from experience**: Adapt weights based on realized P&L
2. **Explore vs exploit**: Balance using best-known weights with trying alternatives
3. **Handle non-stationarity**: Adapt to changing market conditions
4. **Improve over time**: Continuously optimize the ensemble

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ONLINE LEARNING SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │  Exp3IXBandit   │────►│  RewardTracker  │────►│  Weight Updates │   │
│  │                 │     │                 │     │                 │   │
│  │  - Arms         │     │  - P&L tracking │     │  - Blending     │   │
│  │  - Weights      │     │  - Net rewards  │     │  - Arbitration  │   │
│  │  - Selection    │     │  - Attribution  │     │  - Persistence  │   │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘   │
│         │                        │                        │             │
│         └────────────────────────┼────────────────────────┘             │
│                                  │                                       │
│                                  ▼                                       │
│                    ┌─────────────────────────┐                          │
│                    │  EnsembleWeightOptimizer │                         │
│                    │                         │                          │
│                    │  Coordinates bandit     │                          │
│                    │  across all arms        │                          │
│                    └─────────────────────────┘                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Exp3-IX Algorithm

### Overview

Exp3-IX (Exponential-weight algorithm for Exploration and Exploitation with Implicit eXploration) is a multi-armed bandit algorithm designed for adversarial environments.

**Reference**: Neu, Gergely. "Explore no more: Improved high-probability regret bounds for non-stochastic bandits"

### Key Properties

1. **No assumptions**: Works without assuming reward distribution
2. **Adversarial robustness**: Handles changing reward distributions
3. **Regret bounds**: O(√(K T ln K)) regret guarantee
4. **Implicit exploration**: Better exploration-exploitation balance

### Algorithm

```
Initialize: u_i = 1 for all arms i

For each round t:
    1. Compute probabilities:
       p_i = (1 - γ) × (u_i / Σu) + γ / K

    2. Select arm i_t with probability p_i

    3. Observe reward r_{i_t}

    4. Compute importance-weighted estimate:
       r̂_i = r_i / p_i  (if i = i_t, else 0)

    5. Update weights:
       u_i ← u_i × exp(η × r̂_i)
```

---

## Components

### Exp3IXBandit

**Location**: `LIVE_TRADING/learning/bandit.py`

Main bandit implementation.

```python
from LIVE_TRADING.learning import Exp3IXBandit

# Initialize bandit
bandit = Exp3IXBandit(
    n_arms=5,
    arm_names=["lightgbm", "xgboost", "ridge", "mlp", "lstm"],
    gamma=0.05,   # Exploration rate
    eta=None,     # Auto-compute learning rate
    seed=42,
)

# Select arm
arm_idx = bandit.select_arm()
arm_name = bandit.arm_names[arm_idx]

# After observing reward
bandit.update(arm=arm_idx, reward=10.5)  # Net P&L in bps

# Get current weights
weights = bandit.get_weights()       # Normalized array
weights_dict = bandit.get_weights_dict()  # Dict with names
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.05 | Exploration rate (fraction of uniform exploration) |
| `eta` | auto | Learning rate (auto-computed from K and T) |
| `eta_max` | 0.07 | Maximum learning rate |

### Learning Rate Adaptation

```python
# Adaptive learning rate
eta = min(eta_max, sqrt(ln(K) / (K × T)))

Where:
- K = number of arms
- T = total time steps
- eta_max = 0.07 (from config)
```

### Probability Calculation

```python
def get_probabilities(self) -> np.ndarray:
    """
    p_i = (1 - γ) × (u_i / Σu) + γ / K

    Mixes exploitation (normalized weights) with
    exploration (uniform distribution).
    """
    normalized = self._weights / np.sum(self._weights)
    K = self._n_arms
    probs = (1 - self._gamma) * normalized + self._gamma / K
    return probs
```

### Weight Update

```python
def update(self, arm: int, reward: float) -> None:
    """
    Importance-weighted update:
    r̂_i = r_i / p_i
    u_i ← u_i × exp(η × r̂_i)
    """
    probs = self.get_probabilities()
    p_arm = probs[arm]

    # Importance-weighted reward
    r_hat = reward / p_arm

    # Weight update
    self._weights[arm] *= math.exp(self._eta * r_hat)
```

---

### RewardTracker

**Location**: `LIVE_TRADING/learning/reward_tracker.py`

Tracks P&L and calculates rewards for bandit updates.

```python
from LIVE_TRADING.learning import RewardTracker

tracker = RewardTracker()

# Record trade
tracker.record_trade(
    arm="lightgbm",
    entry_price=150.00,
    quantity=100,
    timestamp=datetime.now(),
)

# Close trade
reward = tracker.close_trade(
    arm="lightgbm",
    exit_price=150.50,
    fees=0.50,
    slippage=0.20,
)
# Returns: net P&L in bps = ((150.50 - 150.00) * 100 - 0.50 - 0.20) / (150.00 * 100) * 10000
#                         = (50 - 0.70) / 15000 * 10000 = 32.87 bps
```

### Reward Calculation

```
reward = (exit_price - entry_price) × quantity - fees - slippage
reward_bps = reward / notional × 10000

Where:
- notional = entry_price × quantity
```

---

### EnsembleWeightOptimizer

**Location**: `LIVE_TRADING/learning/weight_optimizer.py`

Coordinates bandit learning across the ensemble.

```python
from LIVE_TRADING.learning import EnsembleWeightOptimizer

optimizer = EnsembleWeightOptimizer(
    families=["lightgbm", "xgboost", "ridge"],
    horizons=["5m", "15m", "60m"],
    mode="family",  # or "horizon" or "family_horizon"
)

# Get optimized weights for blending
weights = optimizer.get_blending_weights(horizon="15m")

# Update after trade
optimizer.update(
    family="lightgbm",
    horizon="15m",
    reward=25.0,  # bps
)
```

### Modes

| Mode | Arms | Use Case |
|------|------|----------|
| `family` | One arm per model family | Optimize family selection |
| `horizon` | One arm per horizon | Optimize horizon selection |
| `family_horizon` | One arm per family×horizon | Full optimization |

---

## Bandit State Management

### Persistence

```python
# Save state
state = bandit.to_dict()
write_atomic_json("state/bandit_state.json", state)

# Load state
state = read_json("state/bandit_state.json")
bandit = Exp3IXBandit.from_dict(state, seed=42)
```

### State Structure

```python
{
    "n_arms": 5,
    "arm_names": ["lightgbm", "xgboost", "ridge", "mlp", "lstm"],
    "gamma": 0.05,
    "eta": 0.042,
    "eta_auto": true,
    "weights": [1.23, 0.98, 1.05, 0.87, 0.92],
    "total_steps": 1500,
    "arm_pulls": [450, 320, 280, 250, 200],
    "cumulative_rewards": [2500, 1800, 1200, 800, 500]
}
```

### Dynamic Arm Management

```python
# Add new model
new_idx = bandit.add_arm(
    arm_name="catboost",
    initial_weight=1.0  # or average of existing weights
)

# Remove underperforming model
bandit.remove_arm("lstm")

# Check if arm exists
if bandit.has_arm("catboost"):
    idx = bandit.get_arm_index("catboost")
```

---

## Integration with Trading Engine

### Blending Integration

```python
class TradingEngine:
    def __init__(self, ...):
        self._bandit = Exp3IXBandit(
            n_arms=len(FAMILIES),
            arm_names=FAMILIES,
        )

    def _get_blending_weights(self, horizon: str) -> Dict[str, float]:
        """Get weights, optionally using bandit."""
        if self.config.use_online_learning:
            # Use bandit weights (mix with base weights)
            bandit_weights = self._bandit.get_weights_dict()
            base_weights = self._ridge_calculator.calculate(...)

            # Blend: 70% base + 30% bandit
            alpha = self.config.bandit_blend_alpha
            return {
                family: (1-alpha) * base + alpha * bandit
                for family, (base, bandit) in
                zip(families, zip(base_weights.values(), bandit_weights.values()))
            }
        else:
            return self._ridge_calculator.calculate(...)
```

### Post-Trade Update

```python
def _update_bandit_after_trade(self, trade_result: TradeResult):
    """Update bandit with trade outcome."""
    if not self.config.use_online_learning:
        return

    # Calculate reward
    reward_bps = (
        (trade_result.exit_price - trade_result.entry_price)
        / trade_result.entry_price * 10000
        - trade_result.fees_bps
        - trade_result.slippage_bps
    )

    # Update all models that contributed to this trade
    for family, contribution in trade_result.model_contributions.items():
        weighted_reward = reward_bps * contribution
        arm_idx = self._bandit.get_arm_index(family)
        self._bandit.update(arm_idx, weighted_reward)
```

---

## Configuration

```yaml
# CONFIG/live_trading/live_trading.yaml

live_trading:
  online_learning:
    enabled: true
    algorithm: "exp3ix"

    # Bandit parameters
    gamma: 0.05         # Exploration rate
    eta_max: 0.07       # Maximum learning rate
    eta_auto: true      # Auto-compute learning rate

    # Weight blending
    blend_alpha: 0.3    # 30% bandit, 70% ridge

    # Persistence
    save_interval: 100  # Save every N updates
    state_file: "state/bandit_state.json"

    # Warm start
    warm_start_steps: 500  # Steps before using bandit weights

    # Arms configuration
    arm_mode: "family"  # or "horizon" or "family_horizon"
```

---

## Monitoring and Debugging

### Statistics

```python
stats = bandit.get_stats()
# {
#     "n_arms": 5,
#     "total_steps": 1500,
#     "gamma": 0.05,
#     "eta": 0.042,
#     "arm_stats": [
#         {
#             "name": "lightgbm",
#             "pulls": 450,
#             "cumulative_reward": 2500,
#             "avg_reward": 5.56,
#             "weight": 0.28,
#             "probability": 0.27,
#         },
#         ...
#     ]
# }
```

### Logging

```python
import logging
logger = logging.getLogger("LIVE_TRADING.learning.bandit")
logger.setLevel(logging.DEBUG)

# Debug output:
# Exp3IX selected arm 0 (lightgbm) with p=0.27
# Exp3IX update: arm=0, reward=25.00bps, r_hat=92.59, eta=0.042, new_weight=1.45
```

---

## Testing

```bash
# Run bandit tests
pytest LIVE_TRADING/tests/test_bandit.py -v

# Key tests:
# - test_weight_update: Verifies correct weight updates
# - test_exploration: Verifies exploration behavior
# - test_convergence: Verifies convergence on synthetic rewards
# - test_persistence: Verifies save/load
# - test_arm_management: Verifies add/remove arms
```

---

## Mathematical Foundations

### Regret Bound

```
Expected regret ≤ O(√(K T ln K))

Where:
- K = number of arms
- T = time horizon
```

### Exploration-Exploitation Balance

```
Probability of selecting arm i:
p_i = (1 - γ) × (u_i / Σu) + γ / K

- First term: Exploit (weight-proportional selection)
- Second term: Explore (uniform random selection)
- γ: Trade-off parameter (0.05 = 5% exploration)
```

### Importance Weighting

```
r̂_i = r_i / p_i

Purpose: Correct for selection bias
- If we select arm i rarely (low p_i), rewards are scaled up
- Creates unbiased estimate of expected reward
```

---

## Best Practices

### 1. Warm Start Period

Allow sufficient exploration before relying on bandit weights:

```python
if bandit.total_steps < warm_start_steps:
    weights = ridge_weights  # Use base weights only
else:
    weights = blend(ridge_weights, bandit_weights)
```

### 2. Reward Normalization

Normalize rewards to prevent extreme updates:

```python
# Clip extreme rewards
reward = np.clip(reward_bps, -100, 100)

# Or use robust normalization
reward = np.tanh(reward_bps / 50) * 50
```

### 3. Regularization

Prevent weight collapse by ensuring minimum weight:

```python
# In update()
self._weights = np.maximum(self._weights, 1e-10)
```

### 4. State Persistence

Save state frequently to prevent loss:

```python
if self._total_steps % save_interval == 0:
    write_atomic_json(state_file, self.to_dict())
```

---

## Related Documentation

- [MULTI_HORIZON_AND_INTERVAL.md](MULTI_HORIZON_AND_INTERVAL.md) - Multi-horizon blending
- [../architecture/MATHEMATICAL_FORMULAS.md](../architecture/MATHEMATICAL_FORMULAS.md) - Mathematical foundations
- [../reference/PLAN_REFERENCES.md](../reference/PLAN_REFERENCES.md) - Source plans
