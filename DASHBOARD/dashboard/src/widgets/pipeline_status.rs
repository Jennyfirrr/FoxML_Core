//! Pipeline status widget - shows current trading pipeline stage

use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::themes::Theme;

/// Pipeline stages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    Idle,
    Prediction,
    Blending,
    Arbitration,
    Gating,
    Sizing,
    Risk,
    Execution,
}

impl PipelineStage {
    /// Parse stage from string (from API)
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "prediction" => Self::Prediction,
            "blending" => Self::Blending,
            "arbitration" => Self::Arbitration,
            "gating" => Self::Gating,
            "sizing" => Self::Sizing,
            "risk" => Self::Risk,
            "execution" => Self::Execution,
            _ => Self::Idle,
        }
    }

    /// Get stage index for ordering (1-7, 0 for idle)
    pub fn index(&self) -> usize {
        match self {
            Self::Idle => 0,
            Self::Prediction => 1,
            Self::Blending => 2,
            Self::Arbitration => 3,
            Self::Gating => 4,
            Self::Sizing => 5,
            Self::Risk => 6,
            Self::Execution => 7,
        }
    }

    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::Prediction => "Prediction",
            Self::Blending => "Blending",
            Self::Arbitration => "Arbitration",
            Self::Gating => "Gating",
            Self::Sizing => "Sizing",
            Self::Risk => "Risk",
            Self::Execution => "Execution",
        }
    }
}

/// Pipeline status widget
pub struct PipelineStatus {
    current_stage: PipelineStage,
    stage_start: std::time::Instant,
}

impl PipelineStatus {
    pub fn new() -> Self {
        Self {
            current_stage: PipelineStage::Idle,
            stage_start: std::time::Instant::now(),
        }
    }

    pub fn set_stage(&mut self, stage: PipelineStage) {
        if stage != self.current_stage {
            self.current_stage = stage;
            self.stage_start = std::time::Instant::now();
        }
    }

    pub fn set_stage_from_str(&mut self, stage: &str) {
        self.set_stage(PipelineStage::from_str(stage));
    }

    pub fn current_stage(&self) -> PipelineStage {
        self.current_stage
    }

    /// Render with theme support
    pub fn render_themed(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        let stages = [
            PipelineStage::Prediction,
            PipelineStage::Blending,
            PipelineStage::Arbitration,
            PipelineStage::Gating,
            PipelineStage::Sizing,
            PipelineStage::Risk,
        ];

        let lines: Vec<Line> = stages
            .iter()
            .map(|stage| {
                let (icon, color) = if *stage == self.current_stage {
                    ("▶", theme.accent)
                } else if self.is_stage_complete(*stage) {
                    ("✓", theme.success)
                } else {
                    ("○", theme.text_muted)
                };
                Line::from(vec![
                    Span::styled(format!("{} ", icon), Style::default().fg(color)),
                    Span::styled(stage.name(), Style::default().fg(color)),
                ])
            })
            .collect();

        // Render lines
        for (i, line) in lines.iter().enumerate() {
            if area.y + (i as u16) < area.bottom() {
                buf.set_line(area.x, area.y + (i as u16), line, area.width);
            }
        }

        // Render footer with current stage info
        let duration = self.stage_start.elapsed().as_secs_f64();
        let footer = if self.current_stage == PipelineStage::Idle {
            "Waiting for next cycle...".to_string()
        } else {
            format!("{} ({:.1}s)", self.current_stage.name(), duration)
        };

        let footer_y = area.y + (lines.len() as u16).min(area.height.saturating_sub(1));
        if footer_y < area.bottom() {
            buf.set_string(
                area.x,
                footer_y + 1,
                &footer,
                Style::default().fg(theme.text_secondary),
            );
        }
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title("Pipeline Status")
            .borders(Borders::ALL);

        let stages = vec![
            ("Prediction", PipelineStage::Prediction),
            ("Blending", PipelineStage::Blending),
            ("Arbitration", PipelineStage::Arbitration),
            ("Gating", PipelineStage::Gating),
            ("Sizing", PipelineStage::Sizing),
            ("Risk", PipelineStage::Risk),
        ];

        let lines: Vec<Line> = stages
            .iter()
            .map(|(name, stage)| {
                let (icon, color) = if *stage == self.current_stage {
                    ("→", Color::Yellow)
                } else if self.is_stage_complete(*stage) {
                    ("✓", Color::Green)
                } else {
                    (" ", Color::Gray)
                };
                Line::from(vec![
                    Span::styled(format!("{} ", icon), Style::default().fg(color)),
                    Span::styled(*name, Style::default().fg(color)),
                ])
            })
            .collect();

        let duration = self.stage_start.elapsed().as_secs_f64();
        let footer_text = format!("Current: {:?} ({:.1}s)", self.current_stage, duration);

        // Add footer as a separate line
        let mut all_lines = lines;
        all_lines.push(Line::from(""));
        all_lines.push(Line::from(footer_text));

        let paragraph = Paragraph::new(all_lines)
            .block(block)
            .wrap(Wrap { trim: true });
        paragraph.render(area, buf);
    }

    fn is_stage_complete(&self, stage: PipelineStage) -> bool {
        // Stages before current are complete
        stage.index() > 0 && stage.index() < self.current_stage.index()
    }
}

impl Default for PipelineStatus {
    fn default() -> Self {
        Self::new()
    }
}
