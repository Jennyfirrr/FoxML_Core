# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Multi-Horizon Trainer

Trains a single model with shared encoder and per-horizon prediction heads.
This enables efficient training across related prediction horizons with
shared feature representations.

Architecture:
    Input → [Shared Encoder] → Split
                               ├── [Head 5m]  → pred_5m
                               ├── [Head 15m] → pred_15m
                               └── [Head 60m] → pred_60m

Config:
    multi_horizon:
      shared_layers: [256, 128]
      head_layers: [64]
      dropout: 0.2
      batch_norm: true
      loss_weights:
        fwd_ret_5m: 1.0
        fwd_ret_15m: 1.0
        fwd_ret_60m: 0.5

Usage:
    from TRAINING.model_fun.multi_horizon_trainer import MultiHorizonTrainer

    trainer = MultiHorizonTrainer(config)
    result = trainer.train(X_tr, y_dict, X_va=X_va, y_va_dict=y_va_dict)
    predictions = trainer.predict(X_test)
"""

from __future__ import annotations

import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first (after __future__)

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MultiHorizonTrainer:
    """
    Trainer for multi-horizon prediction.

    Supports TensorFlow backend with shared encoder and per-horizon heads.

    Attributes:
        config: Configuration dict
        model: Trained Keras model
        horizons: List of horizons in minutes
        targets: List of target names
        is_trained: Whether model has been trained
        scaler: Feature scaler (if preprocessing enabled)
        imputer: Missing value imputer (if preprocessing enabled)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MultiHorizonTrainer.

        Args:
            config: Configuration dict with keys:
                - shared_layers: List of units for shared encoder (default: [256, 128])
                - head_layers: List of units per horizon head (default: [64])
                - dropout: Dropout rate (default: 0.2)
                - batch_norm: Use batch normalization (default: True)
                - epochs: Training epochs (default: 100)
                - batch_size: Batch size (default: 256)
                - patience: Early stopping patience (default: 10)
                - lr: Learning rate (default: 0.001)
                - backend: Only "tensorflow" supported currently
        """
        self.config = config or {}
        self.model = None
        self.horizons: List[int] = []
        self.targets: List[str] = []
        self.is_trained = False
        self.scaler = None
        self.imputer = None
        self._backend = self.config.get("backend", "tensorflow")

        if self._backend not in ("tensorflow", "tf"):
            logger.warning(
                f"Backend '{self._backend}' not fully supported. Using tensorflow."
            )
            self._backend = "tensorflow"

    def train(
        self,
        X_tr: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        X_va: Optional[np.ndarray] = None,
        y_va_dict: Optional[Dict[str, np.ndarray]] = None,
        horizons: Optional[List[int]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train multi-horizon model.

        Args:
            X_tr: Training features (N, D)
            y_dict: Dict of target_name → target_values (N,)
            X_va: Validation features (optional)
            y_va_dict: Validation targets (optional)
            horizons: List of horizons in minutes (inferred from targets if None)
            loss_weights: Per-target loss weights (default: equal weights)
            **kwargs: Additional training args (epochs, batch_size, verbose)

        Returns:
            Training results dict with keys:
                - final_loss: Final training loss
                - val_loss: Final validation loss (if validation data provided)
                - epochs_trained: Number of epochs trained
                - targets: List of target names
                - horizons: List of horizons
        """
        self.targets = list(y_dict.keys())
        self.horizons = horizons or self._infer_horizons(self.targets)

        logger.info(
            f"Training multi-horizon model: {len(self.targets)} horizons "
            f"({self.horizons})"
        )

        # Preprocess features
        X_tr_processed = self._preprocess_fit(X_tr)
        X_va_processed = self._preprocess_transform(X_va) if X_va is not None else None

        # Stack targets for multi-output
        y_tr = np.column_stack([y_dict[t] for t in self.targets]).astype(np.float32)
        y_va = None
        if y_va_dict:
            y_va = np.column_stack([y_va_dict[t] for t in self.targets]).astype(
                np.float32
            )

        # Build model
        input_dim = X_tr_processed.shape[1]
        self.model = self._build_model(input_dim, len(self.targets), loss_weights)

        # Train
        return self._train_tf(X_tr_processed, y_tr, X_va_processed, y_va, **kwargs)

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict for all horizons.

        Args:
            X: Features (N, D)

        Returns:
            Dict of target_name → predictions (N,)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X_processed = self._preprocess_transform(X)
        preds = self.model.predict(X_processed, verbose=0)

        # Handle single output case
        if len(self.targets) == 1:
            preds = preds.reshape(-1, 1)

        # Split multi-output back to dict
        result = {}
        for i, target in enumerate(self.targets):
            result[target] = preds[:, i].astype(np.float32)

        return result

    def predict_single(self, X: np.ndarray, target: str) -> np.ndarray:
        """
        Predict for a single horizon.

        Args:
            X: Features (N, D)
            target: Target name

        Returns:
            Predictions (N,)
        """
        predictions = self.predict(X)
        if target not in predictions:
            raise ValueError(f"Target '{target}' not in model. Available: {self.targets}")
        return predictions[target]

    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform preprocessing pipeline."""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        # Impute missing values
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)

        return X_scaled.astype(np.float32)

    def _preprocess_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using fitted preprocessing pipeline."""
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("Preprocessing not fitted. Call train() first.")

        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled.astype(np.float32)

    def _build_model(
        self,
        input_dim: int,
        n_outputs: int,
        loss_weights: Optional[Dict[str, float]],
    ):
        """Build TensorFlow multi-head model."""
        import os
        import tensorflow as tf
        from tensorflow.keras import Model, layers

        # Set TF seed for determinism (TF_DETERMINISTIC_OPS=1 requires explicit seed)
        seed = int(os.environ.get("PYTHONHASHSEED", "42"))
        tf.random.set_seed(seed)

        shared_layers = self.config.get("shared_layers", [256, 128])
        head_layers = self.config.get("head_layers", [64])
        dropout = self.config.get("dropout", 0.2)
        use_bn = self.config.get("batch_norm", True)

        # Input
        inputs = layers.Input(shape=(input_dim,), name="input")
        x = inputs

        # Shared encoder
        for i, units in enumerate(shared_layers):
            x = layers.Dense(units, activation="relu", name=f"shared_dense_{i}")(x)
            if use_bn:
                x = layers.BatchNormalization(name=f"shared_bn_{i}")(x)
            x = layers.Dropout(dropout, name=f"shared_dropout_{i}")(x)

        shared_output = x

        # Per-horizon heads
        outputs = []
        for target_idx, target in enumerate(self.targets):
            head = shared_output
            # Clean target name for layer naming (replace special chars)
            clean_name = target.replace(".", "_").replace("-", "_")

            for layer_idx, units in enumerate(head_layers):
                head = layers.Dense(
                    units,
                    activation="relu",
                    name=f"head_{clean_name}_{layer_idx}",
                )(head)
                if use_bn:
                    head = layers.BatchNormalization(
                        name=f"head_bn_{clean_name}_{layer_idx}"
                    )(head)
                head = layers.Dropout(dropout, name=f"head_dropout_{clean_name}_{layer_idx}")(
                    head
                )

            output = layers.Dense(1, activation="linear", name=f"output_{clean_name}")(
                head
            )
            outputs.append(output)

        # Concatenate outputs
        if len(outputs) > 1:
            combined_output = layers.Concatenate(name="combined_output")(outputs)
        else:
            combined_output = outputs[0]

        model = Model(inputs=inputs, outputs=combined_output, name="multi_horizon")

        # Compile with weighted loss
        weights = loss_weights or {t: 1.0 for t in self.targets}
        weight_list = [weights.get(t, 1.0) for t in self.targets]

        # Custom weighted MSE loss
        def weighted_mse(y_true, y_pred):
            losses = []
            for i in range(n_outputs):
                if n_outputs > 1:
                    mse = tf.reduce_mean(tf.square(y_true[:, i] - y_pred[:, i]))
                else:
                    mse = tf.reduce_mean(tf.square(y_true - y_pred))
                losses.append(weight_list[i] * mse)
            return tf.reduce_sum(losses)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.get("lr", 0.001)
            ),
            loss=weighted_mse,
        )

        logger.info(f"Built multi-horizon model: {model.count_params():,} parameters")

        return model

    def _train_tf(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: Optional[np.ndarray],
        y_va: Optional[np.ndarray],
        **kwargs,
    ) -> Dict[str, Any]:
        """Train TensorFlow model."""
        import tensorflow as tf

        epochs = kwargs.get("epochs", self.config.get("epochs", 100))
        batch_size = kwargs.get("batch_size", self.config.get("batch_size", 256))
        verbose = kwargs.get("verbose", 0)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_va is not None else "loss",
                patience=self.config.get("patience", 10),
                restore_best_weights=True,
            )
        ]

        validation_data = (X_va, y_va) if X_va is not None else None

        logger.info(
            f"Training for up to {epochs} epochs with batch_size={batch_size}"
        )

        history = self.model.fit(
            X_tr,
            y_tr,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        self.is_trained = True

        result = {
            "final_loss": history.history["loss"][-1],
            "epochs_trained": len(history.history["loss"]),
            "targets": self.targets,
            "horizons": self.horizons,
            "model_params": self.model.count_params(),
        }

        if "val_loss" in history.history:
            result["val_loss"] = history.history["val_loss"][-1]

        logger.info(
            f"Training complete: {result['epochs_trained']} epochs, "
            f"loss={result['final_loss']:.6f}"
        )

        return result

    def _infer_horizons(self, targets: List[str]) -> List[int]:
        """Infer horizons from target names."""
        from TRAINING.common.horizon_bundle import parse_horizon_from_target

        horizons = []
        for target in targets:
            _, horizon = parse_horizon_from_target(target)
            if horizon:
                horizons.append(horizon)
            else:
                horizons.append(0)  # Unknown

        return horizons

    def get_model_metadata(self) -> Dict[str, Any]:
        """Get metadata for model serialization."""
        return {
            "model_type": "multi_horizon",
            "backend": self._backend,
            "targets": self.targets,
            "horizons": self.horizons,
            "config": self.config,
            "is_trained": self.is_trained,
        }

    def save(self, path: str) -> None:
        """Save model and metadata to disk."""
        import joblib
        from pathlib import Path
        from TRAINING.common.utils.file_utils import write_atomic_json

        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        if self.model is not None:
            self.model.save(str(path_obj / "model.keras"))

        # Save preprocessing
        if self.scaler is not None:
            joblib.dump(self.scaler, str(path_obj / "scaler.joblib"))
        if self.imputer is not None:
            joblib.dump(self.imputer, str(path_obj / "imputer.joblib"))

        # Save metadata (atomic write for crash consistency)
        metadata = self.get_model_metadata()
        write_atomic_json(path_obj / "metadata.json", metadata)

        logger.info(f"Saved multi-horizon model to {path}")

    @classmethod
    def load(cls, path: str) -> "MultiHorizonTrainer":
        """Load model and metadata from disk."""
        import json
        import os
        import joblib
        import tensorflow as tf

        # Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        trainer = cls(config=metadata.get("config", {}))
        trainer.targets = metadata.get("targets", [])
        trainer.horizons = metadata.get("horizons", [])
        trainer.is_trained = metadata.get("is_trained", False)

        # Load Keras model
        # Note: We load with compile=False because the custom weighted_mse loss
        # can't be easily restored. The model is still usable for inference.
        # For further training, call _build_model() again or use model.compile().
        model_path = os.path.join(path, "model.keras")
        if os.path.exists(model_path):
            trainer.model = tf.keras.models.load_model(model_path, compile=False)
            # Recompile with basic MSE for inference mode (allows evaluate() to work)
            trainer.model.compile(optimizer="adam", loss="mse")

        # Load preprocessing
        scaler_path = os.path.join(path, "scaler.joblib")
        if os.path.exists(scaler_path):
            trainer.scaler = joblib.load(scaler_path)

        imputer_path = os.path.join(path, "imputer.joblib")
        if os.path.exists(imputer_path):
            trainer.imputer = joblib.load(imputer_path)

        logger.info(f"Loaded multi-horizon model from {path}")
        return trainer
