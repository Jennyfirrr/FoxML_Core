# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Memory Management System - Mega Script Integration
Handles memory optimization, monitoring, and cleanup for large-scale training.
"""


import gc
import psutil
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add CONFIG directory to path for centralized config loading
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_memory_config, get_cfg
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.debug("Config loader not available; using hardcoded defaults")

class MemoryManager:
    """Memory management system for large-scale training."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Load from centralized config if available, otherwise use provided config or defaults
        if _CONFIG_AVAILABLE and not self.config:
            try:
                memory_cfg = get_memory_config()
                thresholds = memory_cfg.get('memory', {}).get('thresholds', {})
                chunking = memory_cfg.get('memory', {}).get('chunking', {})
                cleanup = memory_cfg.get('memory', {}).get('cleanup', {})
                
                self.memory_threshold = thresholds.get('memory_threshold', 0.8)
                self.chunk_size = chunking.get('chunk_size', 1000000)  # Matches memory.yaml → memory.chunking.chunk_size
                self.aggressive_cleanup = cleanup.get('aggressive', True)
            except Exception as e:
                logger.debug(f"Failed to load memory config: {e}, using defaults")
                self.memory_threshold = 0.8
                self.chunk_size = 1000000  # Matches memory.yaml → memory.chunking.chunk_size
                self.aggressive_cleanup = True
        else:
            # Use provided config or defaults (must match memory.yaml defaults)
            self.memory_threshold = self.config.get('memory_threshold', 0.8)  # 80% memory usage
            self.chunk_size = self.config.get('chunk_size', 1000000)  # 1M rows per chunk (matches memory.yaml)
            self.aggressive_cleanup = self.config.get('aggressive_cleanup', True)
        
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get system memory
            system_memory = psutil.virtual_memory()
            
            return {
                'rss_gb': memory_info.rss / 1024**3,  # Resident Set Size
                'vms_gb': memory_info.vms / 1024**3,  # Virtual Memory Size
                'system_total_gb': system_memory.total / 1024**3,
                'system_available_gb': system_memory.available / 1024**3,
                'system_percent': system_memory.percent
            }
        except Exception as e:
            logger.warning(f"Memory monitoring failed: {e}")
            return {'rss_gb': 0, 'vms_gb': 0, 'system_total_gb': 0, 'system_available_gb': 0, 'system_percent': 0}
    
    def log_memory_usage(self, stage: str = "Unknown") -> None:
        """Log current memory usage."""
        memory_info = self.get_memory_usage()
        logger.info(f"💾 Memory at {stage}: RSS={memory_info['rss_gb']:.1f}GB, "
                   f"System={memory_info['system_percent']:.1f}%")
        
        # Warn if memory usage is high
        if memory_info['system_percent'] > 90:
            logger.warning(f"⚠️ High system memory usage: {memory_info['system_percent']:.1f}%")
        if memory_info['rss_gb'] > 50:  # 50GB threshold
            logger.warning(f"⚠️ High process memory usage: {memory_info['rss_gb']:.1f}GB")
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed."""
        memory_info = self.get_memory_usage()
        
        # Load RSS threshold from config if available
        if _CONFIG_AVAILABLE:
            try:
                rss_threshold_gb = get_cfg("memory.thresholds.process_rss_warning_gb", default=50, config_name="memory_config")
            except Exception:
                rss_threshold_gb = 50
        else:
            rss_threshold_gb = 50
        
        return (memory_info['system_percent'] > self.memory_threshold * 100 or 
                memory_info['rss_gb'] > rss_threshold_gb)
    
    def cleanup(self, aggressive: bool = None) -> None:
        """Perform memory cleanup."""
        if aggressive is None:
            aggressive = self.aggressive_cleanup
            
        logger.info("🧹 Performing memory cleanup...")
        
        # Standard cleanup
        gc.collect()
        
        if aggressive:
            # Aggressive cleanup (mega script approach)
            self._aggressive_cleanup()
        
        # Log memory after cleanup
        self.log_memory_usage("after cleanup")
    
    def _aggressive_cleanup(self) -> None:
        """Perform aggressive memory cleanup."""
        try:
            # Clear TensorFlow sessions
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
            except Exception:
                pass
            
            # Clear PyTorch cache if available
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
                
        except Exception as e:
            logger.warning(f"Aggressive cleanup failed: {e}")
    
    def cap_data(self, X: np.ndarray, y: np.ndarray, max_samples: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Cap data to prevent memory issues (mega script approach).

        Args:
            X: Feature matrix
            y: Target vector
            max_samples: Maximum number of samples
            seed: Random seed for reproducibility (uses global determinism if None)
        """
        if len(X) <= max_samples:
            return X, y

        logger.info(f"📊 Capping data from {len(X)} to {max_samples} samples")

        # Use deterministic seed for reproducible sampling
        if seed is None:
            try:
                from TRAINING.common.determinism import BASE_SEED
                seed = BASE_SEED if BASE_SEED is not None else 42
            except ImportError:
                seed = 42
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X), max_samples, replace=False)
        return X[indices], y[indices]
    
    def chunk_data(self, X: np.ndarray, y: np.ndarray, chunk_size: int = None) -> list:
        """Split data into chunks for memory-efficient processing."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        if len(X) <= chunk_size:
            return [(X, y)]
        
        logger.info(f"📦 Splitting data into chunks of {chunk_size} samples")
        
        chunks = []
        for i in range(0, len(X), chunk_size):
            end_idx = min(i + chunk_size, len(X))
            chunks.append((X[i:end_idx], y[i:end_idx]))
        
        return chunks
    
    def monitor_training(self, stage: str) -> None:
        """Monitor memory during training stages."""
        self.log_memory_usage(stage)
        
        if self.should_cleanup():
            logger.warning(f"⚠️ High memory usage at {stage}, performing cleanup")
            self.cleanup()
    
    def get_memory_recommendations(self) -> Dict[str, Any]:
        """Get memory usage recommendations."""
        memory_info = self.get_memory_usage()
        
        recommendations = {
            'current_usage_gb': memory_info['rss_gb'],
            'system_usage_percent': memory_info['system_percent'],
            'recommendations': []
        }
        
        if memory_info['system_percent'] > 90:
            recommendations['recommendations'].append("Consider reducing batch size or using chunked processing")
        
        if memory_info['rss_gb'] > 50:
            recommendations['recommendations'].append("Consider using data capping or more aggressive cleanup")
        
        if memory_info['system_available_gb'] < 10:
            recommendations['recommendations'].append("System memory is low, consider closing other applications")
        
        return recommendations


# =============================================================================
# Standalone Phase Logging Functions (no MemoryManager instance needed)
# =============================================================================

def log_memory_phase(stage: str, level: str = "info") -> Dict[str, float]:
    """
    Log memory usage at a phase boundary (standalone function).

    Use this to track memory at key points in data processing pipelines
    to identify where spikes occur.

    Args:
        stage: Name of the current phase (e.g., "after_load", "after_concat", "after_to_pandas")
        level: Log level ("info", "debug", "warning")

    Returns:
        Dict with memory metrics for programmatic use

    Example:
        >>> from TRAINING.common.memory import log_memory_phase
        >>> log_memory_phase("after_streaming_concat")
        >>> df = lf.collect(streaming=True)
        >>> log_memory_phase("after_collect")
        >>> pdf = df.to_pandas()
        >>> log_memory_phase("after_to_pandas")  # This is where spike happens
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()

        metrics = {
            'rss_gb': memory_info.rss / 1024**3,
            'vms_gb': memory_info.vms / 1024**3,
            'system_used_gb': system_memory.used / 1024**3,
            'system_available_gb': system_memory.available / 1024**3,
            'system_percent': system_memory.percent
        }

        msg = (
            f"📊 Memory [{stage}]: "
            f"RSS={metrics['rss_gb']:.1f}GB, "
            f"System={metrics['system_percent']:.0f}% "
            f"({metrics['system_available_gb']:.0f}GB avail)"
        )

        log_fn = getattr(logger, level, logger.info)
        log_fn(msg)

        # Warn on high usage
        if metrics['rss_gb'] > 60:
            logger.warning(f"⚠️  High RSS at {stage}: {metrics['rss_gb']:.1f}GB")

        return metrics

    except Exception as e:
        logger.debug(f"Memory logging failed at {stage}: {e}")
        return {}


def log_memory_delta(stage: str, baseline_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Log memory change since a baseline measurement.

    Args:
        stage: Current phase name
        baseline_metrics: Metrics dict from a previous log_memory_phase() call

    Returns:
        Current metrics with delta information

    Example:
        >>> baseline = log_memory_phase("before_conversion")
        >>> df = polars_df.to_pandas()
        >>> log_memory_delta("after_conversion", baseline)
        # Output: 📊 Memory [after_conversion]: RSS=88.2GB (+44.1GB), System=78% (28GB avail)
    """
    current = log_memory_phase.__wrapped__(stage) if hasattr(log_memory_phase, '__wrapped__') else {}

    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()

        current = {
            'rss_gb': memory_info.rss / 1024**3,
            'system_percent': system_memory.percent,
            'system_available_gb': system_memory.available / 1024**3
        }

        if baseline_metrics:
            delta_rss = current['rss_gb'] - baseline_metrics.get('rss_gb', 0)
            delta_sign = '+' if delta_rss >= 0 else ''

            msg = (
                f"📊 Memory [{stage}]: "
                f"RSS={current['rss_gb']:.1f}GB ({delta_sign}{delta_rss:.1f}GB), "
                f"System={current['system_percent']:.0f}% "
                f"({current['system_available_gb']:.0f}GB avail)"
            )
        else:
            msg = (
                f"📊 Memory [{stage}]: "
                f"RSS={current['rss_gb']:.1f}GB, "
                f"System={current['system_percent']:.0f}%"
            )

        logger.info(msg)

        # Warn on large spike
        if baseline_metrics and delta_rss > 20:
            logger.warning(
                f"⚠️  Large memory spike at {stage}: +{delta_rss:.1f}GB "
                f"(likely transient duplication)"
            )

        return current

    except Exception as e:
        logger.debug(f"Memory delta logging failed at {stage}: {e}")
        return current
