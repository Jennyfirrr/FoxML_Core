"""
CILS - Continuous Integrated Learning System
=============================================

Online learning components for adaptive weight optimization.

Components:
- Exp3IXBandit: Multi-armed bandit for weight adaptation
- RewardTracker: Track realized P&L per arm
- EnsembleWeightOptimizer: Blend static and bandit weights
- BanditPersistence: Save/load bandit state

Plan Reference: LIVE_TRADING/plans/19_cils_online_learning.md
"""

from LIVE_TRADING.learning.bandit import Exp3IXBandit
from LIVE_TRADING.learning.reward_tracker import RewardTracker, PendingTrade
from LIVE_TRADING.learning.weight_optimizer import EnsembleWeightOptimizer
from LIVE_TRADING.learning.persistence import BanditPersistence

__all__ = [
    "Exp3IXBandit",
    "RewardTracker",
    "PendingTrade",
    "EnsembleWeightOptimizer",
    "BanditPersistence",
]
