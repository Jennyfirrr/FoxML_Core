//! Training pipeline executor - launch training independently of TUI
//!
//! Spawns training as a detached process that continues running even if
//! the TUI is closed. Progress is monitored via the events file.

use anyhow::{anyhow, Result};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Path to PID file for process detection
fn training_pid_file() -> PathBuf { crate::config::training_pid_file() }

/// Training executor - launches and monitors training pipeline
pub struct TrainingExecutor {
    experiment_config: String,
    output_dir: PathBuf,
    deterministic: bool,
    last_status_message: Option<String>,
}

impl TrainingExecutor {
    pub fn new() -> Self {
        Self {
            experiment_config: "production_baseline".to_string(),
            output_dir: crate::config::results_dir().join("prod"),
            deterministic: true,
            last_status_message: None,
        }
    }

    /// Start training pipeline as a detached background process
    pub fn start_training(&mut self) -> Result<()> {
        // Check if already running
        if self.is_running() {
            self.last_status_message = Some("Training already running".to_string());
            return Ok(());
        }

        // Create output directory if it doesn't exist
        fs::create_dir_all(&self.output_dir)?;

        // Build the command string
        let python_args = format!(
            "-m TRAINING.orchestration.intelligent_trainer --experiment-config {} --output-dir {}",
            self.experiment_config,
            self.output_dir.display()
        );

        // Log file path
        let log_file = self.output_dir.join("training.log");

        // Use nohup + setsid to fully detach the process
        // Output goes to log file, process continues after TUI exits
        let shell_cmd = if self.deterministic {
            format!(
                "nohup setsid bin/run_deterministic.sh python {} > {} 2>&1 &",
                python_args,
                log_file.display()
            )
        } else {
            format!(
                "nohup setsid python {} > {} 2>&1 &",
                python_args,
                log_file.display()
            )
        };

        // Spawn via shell to handle nohup/setsid properly
        let status = Command::new("sh")
            .arg("-c")
            .arg(&shell_cmd)
            .current_dir(std::env::current_dir()?)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;

        if status.success() {
            self.last_status_message = Some(format!(
                "Training started: {} → {}",
                self.experiment_config,
                self.output_dir.display()
            ));
            Ok(())
        } else {
            Err(anyhow!("Failed to start training process"))
        }
    }

    /// Stop training pipeline by sending SIGTERM to the process
    pub fn stop_training(&mut self) -> Result<()> {
        if let Some(pid) = self.get_running_pid() {
            // Send SIGTERM for graceful shutdown
            let status = Command::new("kill")
                .arg("-TERM")
                .arg(pid.to_string())
                .status()?;

            if status.success() {
                self.last_status_message = Some("Training stop signal sent".to_string());
            } else {
                // Try SIGKILL if SIGTERM didn't work
                Command::new("kill")
                    .arg("-KILL")
                    .arg(pid.to_string())
                    .status()?;
                self.last_status_message = Some("Training force stopped".to_string());
            }
        } else {
            self.last_status_message = Some("No training process found".to_string());
        }
        Ok(())
    }

    /// Check if training is currently running (via PID file)
    pub fn is_running(&self) -> bool {
        self.get_running_pid().is_some()
    }

    /// Get PID of running training process, if any
    fn get_running_pid(&self) -> Option<u32> {
        let content = fs::read_to_string(&training_pid_file()).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;
        let pid = json.get("pid")?.as_u64()? as u32;

        // Verify process is still alive
        let proc_path = PathBuf::from(format!("/proc/{}", pid));
        if proc_path.exists() {
            Some(pid)
        } else {
            None
        }
    }

    /// Get the run_id of the currently running training, if any
    pub fn get_running_run_id(&self) -> Option<String> {
        let content = fs::read_to_string(&training_pid_file()).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;

        // Verify process is still alive first
        let pid = json.get("pid")?.as_u64()? as u32;
        let proc_path = PathBuf::from(format!("/proc/{}", pid));
        if !proc_path.exists() {
            return None;
        }

        json.get("run_id")?.as_str().map(|s| s.to_string())
    }

    /// Get last status message
    pub fn get_status_message(&self) -> Option<&str> {
        self.last_status_message.as_deref()
    }

    /// Get output lines - now reads from log file instead of captured output
    pub fn get_output(&self) -> Vec<String> {
        let log_file = self.output_dir.join("training.log");
        if let Ok(content) = fs::read_to_string(&log_file) {
            // Return last 100 lines
            content.lines()
                .rev()
                .take(100)
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Set experiment config
    pub fn set_experiment_config(&mut self, config: String) {
        self.experiment_config = config;
    }

    /// Set output directory
    pub fn set_output_dir(&mut self, dir: PathBuf) {
        self.output_dir = dir;
    }

    /// Set deterministic mode
    pub fn set_deterministic(&mut self, deterministic: bool) {
        self.deterministic = deterministic;
    }

    /// Get current config
    pub fn get_experiment_config(&self) -> &str {
        &self.experiment_config
    }

    /// Get current output dir
    pub fn get_output_dir(&self) -> &PathBuf {
        &self.output_dir
    }
}

impl Default for TrainingExecutor {
    fn default() -> Self {
        Self::new()
    }
}
