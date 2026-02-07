# Plan 19: CILS - Continuous Integrated Learning System

## Overview

Implement the Exp3-IX bandit algorithm for online weight adaptation. This system continuously adapts model/horizon weights based on actual realized P&L, providing adaptive ensemble learning in production.

**Source**: `DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md` (Section 5)

## Mathematical Foundation

### Exp3-IX Algorithm

Weight update:
```
u_i <- u_i * exp(eta * r_hat_i)
r_hat_i = r_i / p_i
```

Where:
- `u_i` = weight for arm i (model or horizon)
- `eta` = learning rate: `min(0.07, sqrt(ln(K) / (K * T)))`
- `r_i` = observed reward (net P&L in bps)
- `p_i` = probability of selecting arm i
- `K` = number of arms
- `T` = total time steps

### Probability Selection

```
p_i = (1 - gamma) * (u_i / sum(u)) + gamma / K
```

Where:
- `gamma` = exploration parameter (typically 0.05)
- `K` = total number of arms

### Reward Calculation

```
r_i = net_realized_PnL_bps - fees - slippage
```

## Architecture

### New Files

```
LIVE_TRADING/
├── learning/
│   ├── __init__.py
│   ├── bandit.py              # Exp3-IX implementation
│   ├── reward_tracker.py      # Track realized P&L per arm
│   ├── weight_optimizer.py    # Ensemble weight optimization
│   └── persistence.py         # Save/load bandit state
├── tests/
│   └── test_bandit.py         # Bandit unit tests
```

### Components

#### 1. Exp3IXBandit

```python
class Exp3IXBandit:
    """
    Exp3-IX multi-armed bandit for online weight adaptation.

    Arms can be:
    - Model families (LightGBM, XGBoost, etc.)
    - Horizons (5m, 10m, 15m, etc.)
    - Model-horizon pairs
    """

    def __init__(
        self,
        n_arms: int,
        gamma: float | None = None,  # Exploration rate (default: 0.05)
        eta: float | None = None,    # Learning rate (auto if None)
    ):
        ...

    def select_arm(self) -> int:
        """Select arm using probability distribution."""
        ...

    def update(self, arm: int, reward: float) -> None:
        """Update weights after observing reward."""
        ...

    def get_weights(self) -> np.ndarray:
        """Get normalized weights for blending."""
        ...

    def get_probabilities(self) -> np.ndarray:
        """Get selection probabilities."""
        ...
```

#### 2. RewardTracker

```python
class RewardTracker:
    """
    Tracks realized P&L per arm for bandit feedback.

    Maps trades to arms and calculates net reward after:
    - Execution fees
    - Slippage
    - Market impact
    """

    def __init__(self, fee_bps: float, slippage_bps: float):
        ...

    def record_trade(
        self,
        arm: int,
        symbol: str,
        entry_price: float,
        entry_time: datetime,
    ) -> str:
        """Record trade entry, return trade_id."""
        ...

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        fees: float,
    ) -> float:
        """Record trade exit, return net reward in bps."""
        ...

    def get_pending_trades(self) -> List[PendingTrade]:
        """Get trades awaiting exit."""
        ...
```

#### 3. EnsembleWeightOptimizer

```python
class EnsembleWeightOptimizer:
    """
    Combines bandit learning with static blending weights.

    Final weights = blend(static_ridge_weights, bandit_weights)
    """

    def __init__(
        self,
        arm_names: List[str],
        bandit: Exp3IXBandit,
        blend_ratio: float = 0.3,  # 30% bandit, 70% static
    ):
        ...

    def get_ensemble_weights(
        self,
        static_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Blend static ridge weights with bandit-learned weights."""
        ...

    def update_from_pnl(
        self,
        arm_name: str,
        net_pnl_bps: float,
    ) -> None:
        """Update bandit after observing realized P&L."""
        ...
```

## Integration Points

### 1. Trading Engine Integration

```python
# In TradingEngine.__init__():
self.ensemble_optimizer = EnsembleWeightOptimizer(
    arm_names=self.config.horizons,
    bandit=Exp3IXBandit(len(self.config.horizons)),
)
self.reward_tracker = RewardTracker(
    fee_bps=get_cfg("live_trading.fees.broker_bps"),
    slippage_bps=get_cfg("live_trading.fees.slippage_bps"),
)

# In _execute_trade():
trade_id = self.reward_tracker.record_trade(
    arm=horizon_to_index[decision.horizon],
    symbol=decision.symbol,
    entry_price=fill_price,
    entry_time=current_time,
)
decision.metadata["trade_id"] = trade_id

# On position exit:
reward_bps = self.reward_tracker.record_exit(
    trade_id=trade_id,
    exit_price=exit_price,
    exit_time=exit_time,
    fees=actual_fees,
)
self.ensemble_optimizer.update_from_pnl(decision.horizon, reward_bps)
```

### 2. Horizon Blender Integration

```python
# In HorizonBlender.blend_all_horizons():
if self.use_online_learning:
    static_weights = self._calculate_ridge_weights(predictions)
    final_weights = self.optimizer.get_ensemble_weights(static_weights)
else:
    final_weights = static_weights
```

### 3. State Persistence

Bandit state must be persisted across restarts:
- Arm weights (`u_i`)
- Total steps (`T`)
- Recent rewards history (for analysis)

## Configuration

```yaml
# CONFIG/live_trading/live_trading.yaml
live_trading:
  online_learning:
    enabled: true
    algorithm: "exp3ix"
    gamma: 0.05           # Exploration rate
    eta: null             # Auto-compute learning rate
    blend_ratio: 0.3      # Bandit weight in final blend
    min_samples: 100      # Min trades before bandit influence

  reward:
    fee_bps: 1.0
    slippage_bps: 5.0

  bandit:
    save_interval: 100    # Save state every N updates
    state_path: "state/bandit_state.json"
```

## Implementation Phases

### Phase 1: Core Bandit (~200 LOC)
1. Implement `Exp3IXBandit` class
2. Unit tests for selection and updates
3. Verify convergence on synthetic rewards

### Phase 2: Reward Tracking (~150 LOC)
1. Implement `RewardTracker`
2. Track pending trades
3. Calculate net P&L after costs

### Phase 3: Weight Optimizer (~150 LOC)
1. Implement `EnsembleWeightOptimizer`
2. Blend static and bandit weights
3. Add minimum samples guard

### Phase 4: Integration (~200 LOC)
1. Integrate with TradingEngine
2. Add position exit detection
3. Connect reward feedback loop

### Phase 5: Persistence (~100 LOC)
1. Save/load bandit state
2. Atomic file writes
3. Recovery on startup

### Phase 6: Testing (~200 LOC)
1. Unit tests for all components
2. Integration test with mock trades
3. Convergence test with known reward distribution

**Total Estimated LOC: ~1,000**

## Success Criteria

- [ ] Bandit correctly updates weights on reward signal
- [ ] Weights converge to profitable arms in simulation
- [ ] State persists across engine restarts
- [ ] Blend ratio configurable
- [ ] Minimum samples guard prevents early instability
- [ ] All tests pass

## Risk Considerations

1. **Cold Start**: New arms have no history. Use uniform prior.
2. **Reward Delay**: P&L only known at exit. Use pending trade tracking.
3. **Non-Stationarity**: Market regimes change. Consider decay factor.
4. **Exploration Cost**: Gamma controls exploration/exploitation tradeoff.

## References

- Mathematical Foundations: `DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md`
- Exp3-IX Paper: Neu, Gergely. "Explore no more: Improved high-probability regret bounds for non-stochastic bandits"
- Config Loader: `CONFIG/config_loader.py`

## Status

**Status**: IMPLEMENTED
**Priority**: HIGH (enables adaptive learning)
**Dependencies**: Plans 01-17 complete
**Completed**: 2026-01-19

### Implementation Summary

All phases completed:
- Phase 1: Core `Exp3IXBandit` class with adaptive learning rate
- Phase 2: `RewardTracker` for P&L tracking per arm
- Phase 3: `EnsembleWeightOptimizer` for blending static and bandit weights
- Phase 4: Full TradingEngine integration with CILS hooks
- Phase 5: `BanditPersistence` for state save/load across restarts
- Phase 6: Comprehensive unit tests (38 tests passing)

### Files Created

```
LIVE_TRADING/learning/
├── __init__.py
├── bandit.py           # Exp3IXBandit (~200 LOC)
├── reward_tracker.py   # RewardTracker (~250 LOC)
├── weight_optimizer.py # EnsembleWeightOptimizer (~200 LOC)
└── persistence.py      # BanditPersistence (~200 LOC)

LIVE_TRADING/tests/
└── test_bandit.py      # 38 unit tests (~600 LOC)
```

### TradingEngine Integration

Added to `trading_engine.py`:
- `_init_cils()` - Initialize/restore CILS components
- `_cils_record_trade_entry()` - Record trade for P&L tracking
- `_cils_record_trade_exit()` - Update bandit on trade exit
- `_cils_get_ensemble_weights()` - Get blended weights
- `record_position_exit()` - Public API for position exit feedback
- `get_cils_stats()` - Get detailed CILS statistics
- Added CILS config options to `EngineConfig`

### Configuration

New config keys:
- `live_trading.online_learning.enabled` - Enable/disable CILS
- `live_trading.online_learning.gamma` - Exploration rate (default: 0.05)
- `live_trading.online_learning.blend_ratio` - Bandit weight blend (default: 0.3)
- `live_trading.online_learning.min_samples` - Min trades before bandit influence
- `live_trading.online_learning.enable_horizon_discovery` - Enable dynamic horizon discovery
- `live_trading.online_learning.discovery_interval_cycles` - Discovery check interval (default: 1000)
- `live_trading.bandit.state_dir` - State file directory
- `live_trading.bandit.save_interval` - Save state every N updates

### Dynamic Horizon Discovery (Phase 7)

**Added**: 2026-01-19

Enables CILS to automatically discover and add new horizons/targets at runtime.

#### Features

1. **Dynamic Arm Management**:
   - `Exp3IXBandit.add_arm(name)` - Add new arm with average weight of existing arms
   - `Exp3IXBandit.remove_arm(name)` - Remove arm (minimum 2 arms required)
   - `Exp3IXBandit.has_arm(name)` - Check if arm exists
   - `EnsembleWeightOptimizer.add_arm(name)` - Add new horizon to optimizer
   - `EnsembleWeightOptimizer.remove_arm(name)` - Remove horizon from optimizer

2. **Target Discovery**:
   - `TradingEngine._discover_new_horizons()` - Periodic scan for new model targets
   - `TradingEngine._extract_horizon_from_target(target)` - Extract horizon from target name
   - Supports patterns: `ret_5m`, `fwd_ret_30m`, `spread_1h`, `vol_2h`, `ret_1d`

3. **Automatic Integration**:
   - Discovery runs every `discovery_interval_cycles` (default: 1000)
   - New horizons added to CILS optimizer for learning
   - State persisted after discovery

#### Test Coverage

Added 14 new tests (52 total):
- `test_bandit_add_arm` - Basic arm addition
- `test_bandit_add_arm_uses_average_weight` - Weight initialization
- `test_bandit_add_duplicate_arm_fails` - Duplicate prevention
- `test_bandit_remove_arm` - Basic arm removal
- `test_bandit_remove_nonexistent_arm_fails` - Error handling
- `test_bandit_remove_below_minimum_fails` - Minimum arms guard
- `test_optimizer_add_arm` - Optimizer arm addition
- `test_optimizer_add_arm_preserves_learning` - Learning preservation
- `test_optimizer_remove_arm` - Optimizer arm removal
- `test_optimizer_remove_arm_updates_internal_state` - State consistency
- `test_dynamic_arm_persists` - Persistence of dynamically added arms
- `test_extract_horizon_from_ret_target` - Horizon extraction
- `test_extract_horizon_no_match` - Non-matching targets
- `test_extract_horizon_various_formats` - Format variations

#### Usage

When the engine discovers new trained models (e.g., a `ret_30m` model added after initial setup),
CILS will automatically:
1. Extract the horizon ("30m")
2. Add it as a new arm to the bandit
3. Start learning whether trades on this horizon are profitable
4. Blend its weight with static weights for ensemble decisions
