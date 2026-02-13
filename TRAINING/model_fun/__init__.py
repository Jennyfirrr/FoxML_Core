# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Model Training Functions

Individual model training functions for each model family.
Uses conditional imports to avoid loading TF/Torch in CPU-only child processes.
"""


import os as _os

# ---- Base trainer (always available) ----
from .base_trainer import BaseModelTrainer

__all__ = [
    'BaseModelTrainer',
]

# ---- CPU-only families (conditional imports for optional dependencies) ----
import logging as _logging
_logger = _logging.getLogger(__name__)

def _try_import(name, module_attr):
    """Import a trainer, returning None if its dependency is missing."""
    try:
        mod = __import__(f"TRAINING.model_fun.{name}", fromlist=[module_attr])
        return getattr(mod, module_attr)
    except ImportError as e:
        _logger.debug(f"{module_attr} not available: {e}")
        return None

_cpu_trainers = {
    'LightGBMTrainer': ('lightgbm_trainer', 'LightGBMTrainer'),
    'QuantileLightGBMTrainer': ('quantile_lightgbm_trainer', 'QuantileLightGBMTrainer'),
    'XGBoostTrainer': ('xgboost_trainer', 'XGBoostTrainer'),
    'RewardBasedTrainer': ('reward_based_trainer', 'RewardBasedTrainer'),
    'NGBoostTrainer': ('ngboost_trainer', 'NGBoostTrainer'),
    'EnsembleTrainer': ('ensemble_trainer', 'EnsembleTrainer'),
    'GMMRegimeTrainer': ('gmm_regime_trainer', 'GMMRegimeTrainer'),
    'ChangePointTrainer': ('change_point_trainer', 'ChangePointTrainer'),
    'FTRLProximalTrainer': ('ftrl_proximal_trainer', 'FTRLProximalTrainer'),
}

for _cls_name, (_mod_name, _attr) in _cpu_trainers.items():
    _cls = _try_import(_mod_name, _attr)
    if _cls is not None:
        globals()[_cls_name] = _cls
        __all__.append(_cls_name)

# ---- TensorFlow families (only import if TF is allowed and available) ----
if _os.getenv("TRAINER_CHILD_NO_TF", "0") != "1":
    try:
        # Import TensorFlow - show warnings so user knows if GPU isn't working
        import tensorflow as tf
        _TF_AVAILABLE = True
    except ImportError:
        _TF_AVAILABLE = False
    
    if _TF_AVAILABLE:
        try:
            from .mlp_trainer import MLPTrainer
            from .cnn1d_trainer import CNN1DTrainer
            from .lstm_trainer import LSTMTrainer
            from .transformer_trainer import TransformerTrainer
            from .tabcnn_trainer import TabCNNTrainer
            from .tablstm_trainer import TabLSTMTrainer
            from .tabtransformer_trainer import TabTransformerTrainer
            from .vae_trainer import VAETrainer
            from .gan_trainer import GANTrainer
            from .meta_learning_trainer import MetaLearningTrainer
            from .multi_task_trainer import MultiTaskTrainer
            
            __all__.extend([
                'MLPTrainer',
                'CNN1DTrainer',
                'LSTMTrainer',
                'TransformerTrainer',
                'TabCNNTrainer',
                'TabLSTMTrainer',
                'TabTransformerTrainer',
                'VAETrainer',
                'GANTrainer',
                'MetaLearningTrainer',
                'MultiTaskTrainer',
            ])
        except ImportError as e:
            # TensorFlow is available but trainer import failed
            import logging
            logging.getLogger(__name__).warning(f"TensorFlow trainers not available: {e}")

# ---- PyTorch families (only import if Torch is allowed) ----
# Currently all sequence models use TensorFlow, so this is empty
# Add pure PyTorch trainers here when implemented
if _os.getenv("TRAINER_CHILD_NO_TORCH", "0") != "1":
    pass  # No pure PyTorch trainers yet