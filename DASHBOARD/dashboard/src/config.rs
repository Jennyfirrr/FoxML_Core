//! Runtime configuration from environment variables
//!
//! Centralizes all configurable values that were previously hardcoded.
//! All functions fall back to sensible defaults when env vars are not set.

use std::path::PathBuf;

/// Bridge URL (host:port). Override with `FOXML_BRIDGE_URL`.
pub fn bridge_url() -> String {
    std::env::var("FOXML_BRIDGE_URL").unwrap_or_else(|_| "127.0.0.1:8765".to_string())
}

/// Temp directory for PID files, event logs, and auth tokens.
/// Override with `FOXML_TMP_DIR`.
pub fn tmp_dir() -> PathBuf {
    std::env::var("FOXML_TMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

/// Path to training events JSONL file.
pub fn training_events_file() -> PathBuf {
    tmp_dir().join("foxml_training_events.jsonl")
}

/// Path to training PID file.
pub fn training_pid_file() -> PathBuf {
    tmp_dir().join("foxml_training.pid")
}

/// Project root directory. Override with `FOXML_ROOT`.
pub fn project_root() -> PathBuf {
    std::env::var("FOXML_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// RESULTS directory.
pub fn results_dir() -> PathBuf {
    project_root().join("RESULTS")
}

/// CONFIG directory.
pub fn config_dir() -> PathBuf {
    project_root().join("CONFIG")
}
