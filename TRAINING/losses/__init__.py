# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Ranking Loss Functions and Model Wrappers
=========================================

This package provides loss functions for cross-sectional ranking training
and model wrappers that adapt sequence models for cross-sectional use.

Loss Functions:
    - pairwise_logistic_loss: Pairwise ranking with logistic loss (recommended)
    - pairwise_hinge_loss: Pairwise ranking with hinge loss
    - listwise_softmax_loss: Listwise ranking with softmax cross-entropy
    - listwise_kl_loss: Listwise ranking with KL divergence
    - pointwise_mse_loss: MSE on CS percentile targets (baseline)
    - pointwise_huber_loss: Huber loss on CS percentile targets
    - hybrid_pairwise_pointwise_loss: Combined pairwise + pointwise

Model Wrappers:
    - CrossSectionalSequenceModel: Wraps any sequence model for CS ranking
    - CrossSectionalMLPModel: Simple MLP baseline
    - CrossSectionalAttentionModel: Cross-symbol attention model

Usage:
    from TRAINING.losses import (
        get_ranking_loss,
        RankingLoss,
        CrossSectionalSequenceModel,
    )

    # Create loss function
    loss_fn = get_ranking_loss("pairwise_logistic", top_pct=0.2)
    loss = loss_fn(scores, targets, mask)

    # Or use Module interface
    criterion = RankingLoss("pairwise_logistic", top_pct=0.2)
    loss = criterion(scores, targets, mask)

    # Wrap existing model
    cs_model = CrossSectionalSequenceModel(base_lstm_model)
    scores = cs_model(X, mask)  # (B, M, L, F) -> (B, M)

See:
    - .claude/plans/cs-ranking-phase3-losses.md (design)
    - CONFIG/pipeline/ranking.yaml (configuration)
"""

from TRAINING.losses.ranking_losses import (
    # Loss functions
    pairwise_logistic_loss,
    pairwise_hinge_loss,
    listwise_softmax_loss,
    listwise_kl_loss,
    pointwise_mse_loss,
    pointwise_huber_loss,
    hybrid_pairwise_pointwise_loss,
    # Factory and module
    get_ranking_loss,
    RankingLoss,
    RankingLossType,
    # Numerical helpers
    stable_log_sigmoid,
    stable_log1p_exp,
)

from TRAINING.losses.cs_model_wrapper import (
    CrossSectionalSequenceModel,
    CrossSectionalMLPModel,
    CrossSectionalAttentionModel,
)

__all__ = [
    # Loss functions
    "pairwise_logistic_loss",
    "pairwise_hinge_loss",
    "listwise_softmax_loss",
    "listwise_kl_loss",
    "pointwise_mse_loss",
    "pointwise_huber_loss",
    "hybrid_pairwise_pointwise_loss",
    # Factory and module
    "get_ranking_loss",
    "RankingLoss",
    "RankingLossType",
    # Numerical helpers
    "stable_log_sigmoid",
    "stable_log1p_exp",
    # Model wrappers
    "CrossSectionalSequenceModel",
    "CrossSectionalMLPModel",
    "CrossSectionalAttentionModel",
]
