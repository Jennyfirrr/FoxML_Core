# Additional Features

## Model Selection for LIVE_TRADING

### Overview

Interactive interface to select which model families to use for live trading predictions. Allows per-horizon model selection and real-time configuration updates.

### Features

**Model Browser:**
- Browse all available models from training runs
- View model metadata (AUC, IC, feature count, training date)
- Filter by horizon, target, or model family
- Search models by name or metadata

**Model Selection:**
- Enable/disable model families per horizon
- Example: Use only LightGBM + XGBoost for 5m, add LSTM for 1d
- Visual grid showing selected models
- Real-time validation (check if models exist for selected families)

**Configuration:**
- Save selection to `CONFIG/live_trading/live_trading.yaml`
- Or update service config (`/etc/foxml-trading.conf`)
- Apply changes with service restart

**Example Interface:**
```
┌─────────────────────────────────────────────────────────────┐
│ Model Selector - Choose Models for LIVE_TRADING              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ Run: intelligent-output-20250118-143022  [Select Run ▼]     │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Horizon: 5m                                             │ │
│  ├──────────────┬──────────────┬──────────────┬──────────┤ │
│  │ Model        │ Status       │ AUC          │ Select   │ │
│  ├──────────────┼──────────────┼──────────────┼──────────┤ │
│  │ LightGBM     │ Available    │ 0.65         │ [✓] ON   │ │
│  │ XGBoost      │ Available    │ 0.64         │ [✓] ON   │ │
│  │ LSTM         │ Available    │ 0.62         │ [ ] OFF  │ │
│  │ Transformer  │ Available    │ 0.61         │ [ ] OFF  │ │
│  │ ...          │ ...          │ ...          │ ...      │ │
│  └──────────────┴──────────────┴──────────────┴──────────┘ │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Horizon: 1d                                             │ │
│  ├──────────────┬──────────────┬──────────────┬──────────┤ │
│  │ Model        │ Status       │ AUC          │ Select   │ │
│  ├──────────────┼──────────────┼──────────────┼──────────┤ │
│  │ LightGBM     │ Available    │ 0.58         │ [✓] ON   │ │
│  │ XGBoost      │ Available    │ 0.57         │ [✓] ON   │ │
│  │ LSTM         │ Available    │ 0.55         │ [✓] ON   │ │
│  │ ...          │ ...          │ ...          │ ...      │ │
│  └──────────────┴──────────────┴──────────────┴──────────┘ │
│                                                               │
│ [Save Selection] [Reset to All] [Apply to Service]          │
│                                                               │
│ [↑↓] Navigate  [Space] Toggle  [Enter] Select  [Esc] Back   │
└─────────────────────────────────────────────────────────────┘
```

## Model Health Monitoring (Placeholder)

### Overview

Section for monitoring model health and performance. Currently shows metrics, with placeholder for future autonomous health system.

### Current Features

**Metrics Display:**
- Model performance (AUC, IC, Sharpe)
- Prediction statistics (mean, std, distribution)
- Feature importance trends
- Model age (days since training)

**Placeholder for Future:**
- Autonomous health checks (when system is ready)
- Model degradation detection
- Auto-disable failing models
- Performance trend analysis

**Example Interface:**
```
┌─────────────────────────────────────────────────────────────┐
│ Model Health Monitor                                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Model Health Status                                    │ │
│  ├──────────────┬──────────────┬──────────────┬──────────┤ │
│  │ Model        │ Health       │ Performance  │ Age      │ │
│  ├──────────────┼──────────────┼──────────────┼──────────┤ │
│  │ LightGBM-5m  │ 🟢 Healthy   │ AUC: 0.65    │ 2 days   │ │
│  │ XGBoost-5m   │ 🟢 Healthy   │ AUC: 0.64    │ 2 days   │ │
│  │ LSTM-1d      │ 🟡 Degrading │ AUC: 0.55    │ 30 days  │ │
│  └──────────────┴──────────────┴──────────────┴──────────┘ │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Future: Autonomous Health System                       │ │
│  │                                                         │ │
│  │ [Placeholder]                                          │ │
│  │                                                         │ │
│  │ When autonomous health system is ready, this section   │ │
│  │ will show:                                             │ │
│  │ - Automatic health checks                              │ │
│  │ - Model degradation alerts                             │ │
│  │ - Auto-disable recommendations                         │ │
│  │ - Performance trend analysis                           │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                               │
│ [Refresh] [Export Metrics] [Esc] Back                        │
└─────────────────────────────────────────────────────────────┘
```

## Video Game-Style Config Editor

### Overview

Interactive, visual config editor that makes editing experiments and hyperparameters feel like navigating a game menu. Uses sliders, toggles, and dropdowns instead of raw YAML editing.

### Features

**Interactive Controls:**
- **Sliders**: Numeric values with min/max bounds
  - `learning_rate`: 0.001 → 0.1 (drag slider)
  - `n_estimators`: 100 → 1000 (arrow keys to adjust)
  - `max_depth`: 3 → 20
- **Toggles**: Boolean flags
  - `enable_barrier_gate`: ON/OFF (space to toggle)
  - `market_hours_only`: ON/OFF
- **Dropdowns**: Enum choices
  - `strategy`: single_task/multi_task/cascade (arrow keys to cycle)
  - `broker`: paper/alpaca/ibkr
- **Nested Navigation**: Navigate into nested config sections
  - Enter to expand, Esc to collapse
  - Visual breadcrumb trail

**Visual Feedback:**
- Real-time validation (red border for invalid values)
- Preview of changes before saving
- Default value indicators
- Help text on hover/focus

**Navigation:**
- Arrow keys: Navigate between fields
- Enter: Select/expand nested section
- Esc: Cancel/close
- Space: Toggle boolean
- Tab: Quick jump between sections

**Example Interface:**
```
┌─────────────────────────────────────────────────────────────┐
│ Config Editor - experiments/production_baseline.yaml         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ intelligent_training:                                         │
│   target_ranking:                                            │
│     top_n_targets:        [████████░░] 8                     │
│                            ↑                                  │
│                            Use ←→ to adjust                  │
│                            Min: 1  Max: 20  Default: 10      │
│                                                               │
│     enabled:              [✓] ON                             │
│                            ↑                                  │
│                            Press Space to toggle             │
│                                                               │
│   feature_selection:                                         │
│     top_m_features:       [██████████] 100                    │
│     method:               [multi_model ▼]                    │
│                            ↑                                  │
│                            Press ↑↓ to change                │
│                            Options: multi_model, fast, ...    │
│                                                               │
│   training:                                                  │
│     strategy:             [single_task ▼]                    │
│     families:             [Select...]  [Enter to configure]  │
│                                                               │
│   [Save] [Cancel] [Reset to Defaults] [Help]                  │
│                                                               │
│ [↑↓] Navigate  [←→] Adjust  [Space] Toggle  [Enter] Select  │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Details

**Widget Types:**
- `ConfigSlider`: Numeric slider with bounds
- `ConfigToggle`: Boolean toggle switch
- `ConfigDropdown`: Enum selection dropdown
- `ConfigSection`: Nested section navigator
- `ConfigArray`: Array editor (for lists)

**Validation:**
- Type checking (int, float, bool, string, enum)
- Range validation (min/max for numbers)
- Required field checking
- Schema validation against config schemas

**Persistence:**
- Edits apply to YAML file
- Atomic writes (write to temp, then rename)
- Backup before save (optional)
- Undo/redo support (future)
