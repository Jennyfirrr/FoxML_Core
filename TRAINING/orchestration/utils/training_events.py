"""
Training Event Emitter
======================

Emits training progress events for dashboard monitoring.

Events are sent via HTTP POST to the bridge API, with file fallback
for when the bridge is unavailable.
"""

import atexit
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Bridge API endpoint for training events
BRIDGE_URL = os.environ.get("FOXML_BRIDGE_URL", "http://127.0.0.1:8765")

# Fallback: event file for polling when bridge is unavailable
TRAINING_EVENT_FILE = Path(
    os.environ.get("FOXML_TRAINING_EVENT_FILE", "/tmp/foxml_training_events.jsonl")
)

# PID file for dashboard to detect running training
TRAINING_PID_FILE = Path(
    os.environ.get("FOXML_TRAINING_PID_FILE", "/tmp/foxml_training.pid")
)


class TrainingEventEmitter:
    """
    Emits training events for dashboard consumption.

    Tries HTTP POST to bridge first, falls back to file append.
    """

    def __init__(self, run_id: str):
        """
        Initialize emitter for a training run.

        Args:
            run_id: The training run identifier
        """
        self.run_id = run_id
        self._bridge_url = f"{BRIDGE_URL}/api/training/event"
        self._file_path = TRAINING_EVENT_FILE
        self._http_available = True
        self._session: Optional[Any] = None

    def _get_session(self) -> Any:
        """Get or create HTTP session (lazy import to avoid dependency issues)."""
        if self._session is None:
            try:
                import requests

                self._session = requests.Session()
                self._session.headers.update({"Content-Type": "application/json"})
            except ImportError:
                self._http_available = False
                logger.debug("requests not available, using file fallback")
        return self._session

    def _emit(self, event: dict[str, Any]) -> None:
        """
        Emit an event.

        Args:
            event: Event dictionary to emit
        """
        event["run_id"] = self.run_id
        event["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Try HTTP POST first
        if self._http_available:
            session = self._get_session()
            if session:
                try:
                    response = session.post(
                        self._bridge_url, json=event, timeout=1.0
                    )
                    if response.status_code == 200:
                        return
                except Exception as e:
                    logger.debug(f"HTTP POST failed: {e}")
                    # Don't disable HTTP - bridge might come back online

        # Fallback to file
        try:
            event_json = json.dumps(event) + "\n"
            with open(self._file_path, "a") as f:
                f.write(event_json)
        except Exception as e:
            logger.warning(f"Failed to write training event: {e}")

    def emit_progress(
        self,
        stage: str,
        progress_pct: float,
        current_target: Optional[str] = None,
        targets_complete: int = 0,
        targets_total: int = 0,
        message: Optional[str] = None,
    ) -> None:
        """
        Emit a progress update.

        Args:
            stage: Current stage (ranking, feature_selection, training)
            progress_pct: Overall progress percentage [0, 100]
            current_target: Target currently being processed
            targets_complete: Number of targets completed
            targets_total: Total number of targets
            message: Optional status message
        """
        self._emit(
            {
                "event_type": "progress",
                "stage": stage,
                "progress_pct": progress_pct,
                "current_target": current_target,
                "targets_complete": targets_complete,
                "targets_total": targets_total,
                "message": message,
            }
        )

    def emit_stage_change(
        self, previous_stage: Optional[str], new_stage: str
    ) -> None:
        """
        Emit a stage transition.

        Args:
            previous_stage: Previous stage (None if starting)
            new_stage: New stage being entered
        """
        self._emit(
            {
                "event_type": "stage_change",
                "previous_stage": previous_stage,
                "new_stage": new_stage,
            }
        )

    def emit_target_start(self, target: str, target_index: int, total_targets: int) -> None:
        """
        Emit target processing start.

        Args:
            target: Target name
            target_index: Index of this target (0-based)
            total_targets: Total number of targets
        """
        self._emit(
            {
                "event_type": "target_start",
                "target": target,
                "target_index": target_index,
                "total_targets": total_targets,
            }
        )

    def emit_target_complete(
        self,
        target: str,
        status: str,
        models_trained: int = 0,
        best_auc: Optional[float] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Emit target completion.

        Args:
            target: Target name
            status: Completion status (success, failed, skipped)
            models_trained: Number of models trained
            best_auc: Best AUC achieved (if available)
            duration_seconds: Time taken for this target
        """
        self._emit(
            {
                "event_type": "target_complete",
                "target": target,
                "status": status,
                "models_trained": models_trained,
                "best_auc": best_auc,
                "duration_seconds": duration_seconds,
            }
        )

    def emit_run_complete(
        self,
        status: str = "success",
        total_targets: int = 0,
        successful_targets: int = 0,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Emit run completion.

        Args:
            status: Final status (success, failed, cancelled)
            total_targets: Total targets processed
            successful_targets: Number of successful targets
            duration_seconds: Total run duration
        """
        self._emit(
            {
                "event_type": "run_complete",
                "status": status,
                "total_targets": total_targets,
                "successful_targets": successful_targets,
                "duration_seconds": duration_seconds,
            }
        )

    def emit_error(
        self,
        error_message: str,
        error_type: str = "unknown",
        recoverable: bool = True,
        target: Optional[str] = None,
    ) -> None:
        """
        Emit an error event.

        Args:
            error_message: Error description
            error_type: Error classification
            recoverable: Whether training can continue
            target: Target where error occurred (if applicable)
        """
        self._emit(
            {
                "event_type": "error",
                "error_message": error_message,
                "error_type": error_type,
                "recoverable": recoverable,
                "target": target,
            }
        )

    def close(self) -> None:
        """Close the event emitter."""
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None


# Global emitter instance
_emitter: Optional[TrainingEventEmitter] = None


def _write_pid_file(run_id: str) -> None:
    """Write PID file for dashboard process detection."""
    try:
        pid_data = {
            "pid": os.getpid(),
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        TRAINING_PID_FILE.write_text(json.dumps(pid_data))
        logger.debug(f"Wrote PID file: {TRAINING_PID_FILE}")
    except Exception as e:
        logger.warning(f"Failed to write PID file: {e}")


def _remove_pid_file() -> None:
    """Remove PID file on exit."""
    try:
        if TRAINING_PID_FILE.exists():
            TRAINING_PID_FILE.unlink()
            logger.debug(f"Removed PID file: {TRAINING_PID_FILE}")
    except Exception as e:
        logger.warning(f"Failed to remove PID file: {e}")


def init_training_events(run_id: str) -> TrainingEventEmitter:
    """
    Initialize the global training event emitter.

    Args:
        run_id: Training run identifier

    Returns:
        The event emitter instance
    """
    global _emitter
    if _emitter is not None:
        _emitter.close()
    _emitter = TrainingEventEmitter(run_id)

    # Write PID file for dashboard detection
    _write_pid_file(run_id)

    # Register cleanup on exit
    atexit.register(_remove_pid_file)

    return _emitter


def get_training_emitter() -> Optional[TrainingEventEmitter]:
    """Get the global training event emitter."""
    return _emitter


def close_training_events() -> None:
    """Close the global training event emitter."""
    global _emitter
    if _emitter is not None:
        _emitter.close()
        _emitter = None

    # Remove PID file
    _remove_pid_file()


# Convenience functions for common operations


def emit_progress(stage: str, progress_pct: float, **kwargs: Any) -> None:
    """Convenience function to emit progress."""
    if _emitter:
        _emitter.emit_progress(stage, progress_pct, **kwargs)


def emit_stage_change(previous_stage: Optional[str], new_stage: str) -> None:
    """Convenience function to emit stage change."""
    if _emitter:
        _emitter.emit_stage_change(previous_stage, new_stage)


def emit_target_start(target: str, target_index: int, total_targets: int) -> None:
    """Convenience function to emit target start."""
    if _emitter:
        _emitter.emit_target_start(target, target_index, total_targets)


def emit_target_complete(target: str, status: str, **kwargs: Any) -> None:
    """Convenience function to emit target completion."""
    if _emitter:
        _emitter.emit_target_complete(target, status, **kwargs)


def emit_run_complete(status: str = "success", **kwargs: Any) -> None:
    """Convenience function to emit run completion."""
    if _emitter:
        _emitter.emit_run_complete(status, **kwargs)


def emit_error(error_message: str, **kwargs: Any) -> None:
    """Convenience function to emit error."""
    if _emitter:
        _emitter.emit_error(error_message, **kwargs)
