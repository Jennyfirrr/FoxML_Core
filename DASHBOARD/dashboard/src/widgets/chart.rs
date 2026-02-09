//! Chart widgets: Sparkline (block chars) and LineChart (braille dots)

use ratatui::prelude::*;
use std::collections::VecDeque;

use crate::themes::Theme;

// ─── Block-character sparkline ──────────────────────────────────────

/// Unicode block characters for sparkline (8 levels from empty to full)
const SPARK_CHARS: [char; 8] = ['\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}'];

/// Compact sparkline that renders in 1 row using block characters
pub struct Sparkline {
    data: VecDeque<f64>,
    max_points: usize,
    color: Color,
    label: Option<String>,
}

impl Sparkline {
    pub fn new(max_points: usize, color: Color) -> Self {
        Self {
            data: VecDeque::with_capacity(max_points),
            max_points,
            color,
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn push(&mut self, value: f64) {
        if self.data.len() >= self.max_points {
            self.data.pop_front();
        }
        self.data.push_back(value);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Render the sparkline into a single-row area
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        if area.height == 0 || area.width == 0 || self.data.is_empty() {
            return;
        }

        let label_width = if let Some(ref label) = self.label {
            let label_str = format!("{} ", label);
            let w = label_str.len().min(area.width as usize);
            for (i, ch) in label_str.chars().take(w).enumerate() {
                if let Some(cell) = buf.cell_mut((area.x + i as u16, area.y)) {
                    cell.set_char(ch);
                    cell.set_style(Style::default().fg(self.color));
                }
            }
            w as u16
        } else {
            0
        };

        let chart_width = (area.width - label_width) as usize;
        if chart_width == 0 {
            return;
        }

        // Determine visible data range
        let visible_count = chart_width.min(self.data.len());
        let start = self.data.len() - visible_count;
        let visible: Vec<f64> = self.data.iter().skip(start).copied().collect();

        // Auto-scale
        let min_val = visible.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = visible.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if (max_val - min_val).abs() < f64::EPSILON {
            1.0
        } else {
            max_val - min_val
        };

        // Render each data point as a block character
        for (i, &val) in visible.iter().enumerate() {
            let normalized = ((val - min_val) / range).clamp(0.0, 1.0);
            let idx = (normalized * 7.0).round() as usize;
            let ch = SPARK_CHARS[idx.min(7)];
            let x = area.x + label_width + i as u16;
            if x < area.right() {
                if let Some(cell) = buf.cell_mut((x, area.y)) {
                    cell.set_char(ch);
                    cell.set_style(Style::default().fg(self.color));
                }
            }
        }
    }
}

// ─── Braille-dot line chart ─────────────────────────────────────────

/// Braille dot positions (each cell is 2 columns x 4 rows)
/// Bit positions:
///   0x01 0x08
///   0x02 0x10
///   0x04 0x20
///   0x40 0x80
const BRAILLE_BASE: u32 = 0x2800;

/// A data series for the line chart
pub struct Series {
    pub data: VecDeque<f64>,
    pub color: Color,
    pub label: String,
    max_points: usize,
}

impl Series {
    pub fn new(label: impl Into<String>, color: Color, max_points: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(max_points),
            color,
            label: label.into(),
            max_points,
        }
    }

    pub fn push(&mut self, value: f64) {
        if self.data.len() >= self.max_points {
            self.data.pop_front();
        }
        self.data.push_back(value);
    }
}

/// Line chart using braille-dot rendering for high resolution
pub struct LineChart {
    pub series: Vec<Series>,
    pub y_label: String,
}

impl LineChart {
    pub fn new() -> Self {
        Self {
            series: Vec::new(),
            y_label: String::new(),
        }
    }

    pub fn with_y_label(mut self, label: impl Into<String>) -> Self {
        self.y_label = label.into();
        self
    }

    pub fn add_series(&mut self, series: Series) {
        self.series.push(series);
    }

    /// Render the line chart into the given area
    pub fn render(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        if area.height < 3 || area.width < 10 {
            return;
        }

        // Reserve space for Y-axis labels (8 chars) and bottom axis (1 row)
        let y_label_width: u16 = 8;
        let chart_x = area.x + y_label_width;
        let chart_width = area.width.saturating_sub(y_label_width);
        let chart_height = area.height.saturating_sub(1); // leave 1 row for x-axis

        if chart_width == 0 || chart_height == 0 {
            return;
        }

        // Collect all values for auto-scaling
        let all_values: Vec<f64> = self.series.iter()
            .flat_map(|s| s.data.iter())
            .copied()
            .collect();

        if all_values.is_empty() {
            let msg = "No data";
            let x = area.x + (area.width.saturating_sub(msg.len() as u16)) / 2;
            for (i, ch) in msg.chars().enumerate() {
                if let Some(cell) = buf.cell_mut((x + i as u16, area.y + area.height / 2)) {
                    cell.set_char(ch);
                    cell.set_style(Style::default().fg(theme.text_muted));
                }
            }
            return;
        }

        let min_val = all_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = all_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if (max_val - min_val).abs() < f64::EPSILON {
            1.0
        } else {
            max_val - min_val
        };

        // Braille grid resolution: each cell = 2 x-dots, 4 y-dots
        let grid_w = chart_width as usize * 2;
        let grid_h = chart_height as usize * 4;

        // Create braille grid (track color per cell for multi-series)
        let mut grid: Vec<Vec<u8>> = vec![vec![0u8; grid_w]; grid_h];
        // Track which series "owns" each cell for coloring
        let mut cell_color: Vec<Vec<Option<Color>>> =
            vec![vec![None; chart_width as usize]; chart_height as usize];

        for series in &self.series {
            let n = series.data.len();
            if n < 2 {
                continue;
            }

            // Map data points to braille grid coordinates
            let visible_count = (grid_w).min(n);
            let start = n.saturating_sub(visible_count);

            for i in start..n {
                let gx = ((i - start) as f64 / visible_count as f64 * (grid_w - 1) as f64).round() as usize;
                let normalized = ((series.data[i] - min_val) / range).clamp(0.0, 1.0);
                let gy = ((1.0 - normalized) * (grid_h - 1) as f64).round() as usize;

                let gx = gx.min(grid_w - 1);
                let gy = gy.min(grid_h - 1);
                grid[gy][gx] = 1;

                // Track cell ownership
                let cx = gx / 2;
                let cy = gy / 4;
                if cx < chart_width as usize && cy < chart_height as usize {
                    cell_color[cy][cx] = Some(series.color);
                }
            }

            // Connect points with lines (Bresenham between consecutive grid points)
            for i in start..(n - 1) {
                let x0 = ((i - start) as f64 / visible_count as f64 * (grid_w - 1) as f64).round() as i32;
                let x1 = ((i + 1 - start) as f64 / visible_count as f64 * (grid_w - 1) as f64).round() as i32;
                let y0 = ((1.0 - ((series.data[i] - min_val) / range).clamp(0.0, 1.0)) * (grid_h - 1) as f64).round() as i32;
                let y1 = ((1.0 - ((series.data[i + 1] - min_val) / range).clamp(0.0, 1.0)) * (grid_h - 1) as f64).round() as i32;

                // Simple Bresenham line
                let dx = (x1 - x0).abs();
                let dy = -(y1 - y0).abs();
                let sx = if x0 < x1 { 1 } else { -1 };
                let sy = if y0 < y1 { 1 } else { -1 };
                let mut err = dx + dy;
                let mut cx = x0;
                let mut cy = y0;

                loop {
                    let gx = (cx as usize).min(grid_w - 1);
                    let gy = (cy as usize).min(grid_h - 1);
                    grid[gy][gx] = 1;

                    let ccx = gx / 2;
                    let ccy = gy / 4;
                    if ccx < chart_width as usize && ccy < chart_height as usize {
                        cell_color[ccy][ccx] = Some(series.color);
                    }

                    if cx == x1 && cy == y1 {
                        break;
                    }
                    let e2 = 2 * err;
                    if e2 >= dy {
                        err += dy;
                        cx += sx;
                    }
                    if e2 <= dx {
                        err += dx;
                        cy += sy;
                    }
                }
            }
        }

        // Render braille characters
        for cy in 0..chart_height as usize {
            for cx in 0..chart_width as usize {
                let mut braille: u32 = 0;
                // Map 2x4 grid dots to braille bits
                let bit_map: [[u32; 2]; 4] = [
                    [0x01, 0x08],
                    [0x02, 0x10],
                    [0x04, 0x20],
                    [0x40, 0x80],
                ];
                for row in 0..4 {
                    for col in 0..2 {
                        let gy = cy * 4 + row;
                        let gx = cx * 2 + col;
                        if gy < grid_h && gx < grid_w && grid[gy][gx] != 0 {
                            braille |= bit_map[row][col];
                        }
                    }
                }

                if braille != 0 {
                    let ch = char::from_u32(BRAILLE_BASE + braille).unwrap_or(' ');
                    let screen_x = chart_x + cx as u16;
                    let screen_y = area.y + cy as u16;
                    if let Some(cell) = buf.cell_mut((screen_x, screen_y)) {
                        cell.set_char(ch);
                        let color = cell_color[cy][cx].unwrap_or(theme.accent);
                        cell.set_style(Style::default().fg(color));
                    }
                }
            }
        }

        // Y-axis labels (top, middle, bottom)
        let format_val = |v: f64| -> String {
            if v.abs() >= 1_000_000.0 {
                format!("{:.1}M", v / 1_000_000.0)
            } else if v.abs() >= 1_000.0 {
                format!("{:.1}K", v / 1_000.0)
            } else {
                format!("{:.1}", v)
            }
        };

        let labels = [
            (area.y, format_val(max_val)),
            (area.y + chart_height / 2, format_val((max_val + min_val) / 2.0)),
            (area.y + chart_height.saturating_sub(1), format_val(min_val)),
        ];

        for (y, label) in &labels {
            let padded = format!("{:>7}", label);
            for (i, ch) in padded.chars().enumerate() {
                let x = area.x + i as u16;
                if x < chart_x {
                    if let Some(cell) = buf.cell_mut((x, *y)) {
                        cell.set_char(ch);
                        cell.set_style(Style::default().fg(theme.text_muted));
                    }
                }
            }
        }

        // Legend (if multiple series)
        if self.series.len() > 1 {
            let legend_y = area.y + chart_height;
            let mut lx = chart_x;
            for series in &self.series {
                if lx + series.label.len() as u16 + 4 > area.right() {
                    break;
                }
                // Color dot
                if let Some(cell) = buf.cell_mut((lx, legend_y)) {
                    cell.set_char('\u{25CF}'); // filled circle
                    cell.set_style(Style::default().fg(series.color));
                }
                lx += 1;
                // Label
                for ch in series.label.chars() {
                    if lx < area.right() {
                        if let Some(cell) = buf.cell_mut((lx, legend_y)) {
                            cell.set_char(ch);
                            cell.set_style(Style::default().fg(theme.text_secondary));
                        }
                    }
                    lx += 1;
                }
                lx += 2; // spacing
            }
        }
    }
}
