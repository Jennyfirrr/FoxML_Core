//! Animation effects infrastructure
//!
//! Provides animation state management for view transitions and effects.
//! Uses tachyonfx for effect processing.

use std::time::{Duration, Instant};
use ratatui::prelude::*;

/// Simple fade state for transitions
#[derive(Clone, Debug)]
pub struct FadeState {
    start: Instant,
    duration: Duration,
    direction: FadeDirection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FadeDirection {
    In,
    Out,
}

impl FadeState {
    pub fn fade_in(duration_ms: u64) -> Self {
        Self {
            start: Instant::now(),
            duration: Duration::from_millis(duration_ms),
            direction: FadeDirection::In,
        }
    }

    pub fn fade_out(duration_ms: u64) -> Self {
        Self {
            start: Instant::now(),
            duration: Duration::from_millis(duration_ms),
            direction: FadeDirection::Out,
        }
    }

    pub fn progress(&self) -> f32 {
        let elapsed = self.start.elapsed();
        let progress = elapsed.as_secs_f32() / self.duration.as_secs_f32();
        progress.clamp(0.0, 1.0)
    }

    pub fn is_done(&self) -> bool {
        self.start.elapsed() >= self.duration
    }

    /// Get alpha value (0.0 to 1.0)
    pub fn alpha(&self) -> f32 {
        let p = self.progress();
        match self.direction {
            FadeDirection::In => ease_out_cubic(p),
            FadeDirection::Out => 1.0 - ease_in_cubic(p),
        }
    }
}

/// Cubic ease-out function
fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

/// Cubic ease-in function
fn ease_in_cubic(t: f32) -> f32 {
    t.powi(3)
}

/// Animation presets
pub struct Animations;

impl Animations {
    /// Create a fade-in state
    pub fn fade_in(duration_ms: u64) -> FadeState {
        FadeState::fade_in(duration_ms)
    }

    /// Create a fade-out state
    pub fn fade_out(duration_ms: u64) -> FadeState {
        FadeState::fade_out(duration_ms)
    }
}

/// Tracks active animations by name
pub struct AnimationManager {
    fades: Vec<(String, FadeState, Rect)>,
}

impl AnimationManager {
    pub fn new() -> Self {
        Self {
            fades: Vec::new(),
        }
    }

    /// Start a fade animation for a region
    pub fn start_fade(&mut self, name: impl Into<String>, fade: FadeState, area: Rect) {
        let name = name.into();
        // Remove any existing animation with the same name
        self.fades.retain(|(n, _, _)| n != &name);
        self.fades.push((name, fade, area));
    }

    /// Get fade alpha for a named region (returns 1.0 if no animation)
    pub fn get_alpha(&self, name: &str) -> f32 {
        self.fades
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, fade, _)| fade.alpha())
            .unwrap_or(1.0)
    }

    /// Remove completed animations
    pub fn cleanup(&mut self) {
        self.fades.retain(|(_, fade, _)| !fade.is_done());
    }

    /// Check if any animations are running
    pub fn is_animating(&self) -> bool {
        !self.fades.is_empty()
    }

    /// Clear all animations
    pub fn clear(&mut self) {
        self.fades.clear();
    }
}

impl Default for AnimationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply alpha to a color (simple dimming)
pub fn apply_alpha(color: Color, alpha: f32) -> Color {
    match color {
        Color::Rgb(r, g, b) => {
            let r = (r as f32 * alpha) as u8;
            let g = (g as f32 * alpha) as u8;
            let b = (b as f32 * alpha) as u8;
            Color::Rgb(r, g, b)
        }
        // For non-RGB colors, we can't easily apply alpha
        // Return the color unchanged
        other => other,
    }
}
