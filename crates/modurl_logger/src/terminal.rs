use std::{
    collections::BTreeMap,
    fmt::Write as _,
    io::{self, Write as _},
    time::{Duration, Instant},
};

use candle_core::Tensor;
use console::Term;
use textplots::{Chart, LabelBuilder, LabelFormat, Plot, Shape};

use crate::{
    AggregationConfig, Logger, Result,
    aggregation::{Aggregator, CompletedTimestep},
};

const DEFAULT_CHART_WIDTH: u32 = 180;
const DEFAULT_CHART_HEIGHT: u32 = 30;
const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_millis(250);
const TARGET_CELL_ASPECT_RATIO: f32 = 4.0;

/// Collects named scalar metrics and plots them in the terminal.
#[derive(Debug)]
pub struct TerminalLogger {
    aggregator: Aggregator,
    series: BTreeMap<String, Vec<(usize, f32)>>,
    live_updates: bool,
    refresh_interval: Duration,
    last_refresh: Option<Instant>,
}

impl TerminalLogger {
    pub fn new(config: AggregationConfig) -> Self {
        Self {
            aggregator: Aggregator::new(config),
            series: BTreeMap::new(),
            live_updates: false,
            refresh_interval: DEFAULT_REFRESH_INTERVAL,
            last_refresh: None,
        }
    }

    /// Redraws all completed metric graphs whenever a timestep completes.
    /// Graphs automatically resize into a multi-column terminal grid.
    ///
    /// Live updates are skipped when standard output is not an interactive
    /// terminal. The final [`display`](Self::display) still prints normally.
    #[must_use]
    pub fn with_live_updates(mut self) -> Self {
        self.live_updates = true;
        self
    }

    /// Sets the minimum time between live terminal redraws.
    ///
    /// The default is 250 milliseconds. [`Duration::ZERO`] redraws after every
    /// completed timestep. The final [`display`](Self::display) is never
    /// throttled.
    #[must_use]
    pub fn with_refresh_interval(mut self, refresh_interval: Duration) -> Self {
        self.refresh_interval = refresh_interval;
        self
    }

    /// Completes the current timestep and makes its values available to
    /// [`series`](Self::series).
    ///
    /// A timestep normally completes when a larger timestep is logged. Call
    /// this after the last log entry because no later timestep will arrive to
    /// complete it.
    pub fn finish(&mut self) {
        if let Some(completed) = self.aggregator.finish() {
            self.store(completed);
        }
    }

    /// Returns the completed points for a metric.
    pub fn series(&self, metric: &str) -> Option<&[(usize, f32)]> {
        self.series.get(metric).map(Vec::as_slice)
    }

    /// Completes the current timestep and plots every metric.
    pub fn display(&mut self) {
        self.finish();
        if !self.live_updates || !self.redraw_dashboard() {
            self.plot_all_sequentially();
        }
    }

    fn store(&mut self, completed: CompletedTimestep) {
        let CompletedTimestep { timestep, metrics } = completed;
        for (name, value) in metrics {
            self.series.entry(name).or_default().push((timestep, value));
        }
    }

    fn update_live_display(&mut self) {
        let now = Instant::now();
        if self.refresh_is_due(now) && self.redraw_dashboard() {
            self.last_refresh = Some(Instant::now());
        }
    }

    fn refresh_is_due(&self, now: Instant) -> bool {
        self.last_refresh
            .is_none_or(|last_refresh| now.duration_since(last_refresh) >= self.refresh_interval)
    }

    fn redraw_dashboard(&self) -> bool {
        let terminal = Term::stdout();
        if !terminal.is_term() {
            return false;
        }

        let (terminal_rows, terminal_columns) = terminal.size();
        let dashboard = self.render_dashboard(terminal_rows as usize, terminal_columns as usize);
        if terminal.clear_screen().is_err() {
            return false;
        }
        print!("{dashboard}");
        let _ = io::stdout().flush();
        true
    }

    fn render_dashboard(&self, terminal_rows: usize, terminal_columns: usize) -> String {
        if self.series.is_empty() {
            return String::new();
        }

        let layout = GridLayout::new(self.series.len(), terminal_rows, terminal_columns);
        let charts = self
            .series
            .iter()
            .map(|(name, values)| render_chart(name, values, layout.cell_width, layout.cell_height))
            .collect::<Vec<_>>();
        render_grid(&charts, layout)
    }

    fn plot_all_sequentially(&self) {
        for (name, values) in &self.series {
            plot(name, values, DEFAULT_CHART_WIDTH, DEFAULT_CHART_HEIGHT);
        }
    }
}

impl Logger for TerminalLogger {
    fn log(&mut self, timestep: usize, metrics: &[(&str, &Tensor)]) -> Result<()> {
        if let Some(completed) = self.aggregator.log(timestep, metrics)? {
            self.store(completed);
            if self.live_updates {
                self.update_live_display();
            }
        }
        Ok(())
    }
}

impl Default for TerminalLogger {
    fn default() -> Self {
        Self::new(AggregationConfig::default())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GridLayout {
    columns: usize,
    rows: usize,
    cell_width: usize,
    cell_height: usize,
}

impl GridLayout {
    fn new(chart_count: usize, terminal_rows: usize, terminal_columns: usize) -> Self {
        let available_rows = terminal_rows.saturating_sub(1).max(1);
        let terminal_columns = terminal_columns.max(1);
        let columns = (1..=chart_count)
            .min_by(|left, right| {
                aspect_ratio_error(chart_count, *left, available_rows, terminal_columns).total_cmp(
                    &aspect_ratio_error(chart_count, *right, available_rows, terminal_columns),
                )
            })
            .unwrap_or(1);
        let rows = chart_count.div_ceil(columns);
        let cell_width = terminal_columns.saturating_sub(columns - 1) / columns;
        let cell_height = available_rows / rows;

        Self {
            columns,
            rows,
            cell_width: cell_width.max(1),
            cell_height: cell_height.max(1),
        }
    }
}

fn aspect_ratio_error(
    chart_count: usize,
    columns: usize,
    terminal_rows: usize,
    terminal_columns: usize,
) -> f32 {
    let rows = chart_count.div_ceil(columns);
    let cell_width = terminal_columns.saturating_sub(columns - 1) / columns;
    let cell_height = terminal_rows / rows;
    let aspect_ratio = cell_width as f32 / cell_height.max(1) as f32;
    (aspect_ratio - TARGET_CELL_ASPECT_RATIO).abs()
}

fn render_chart(
    name: &str,
    values: &[(usize, f32)],
    cell_width: usize,
    cell_height: usize,
) -> Vec<String> {
    let plot_character_width = cell_width.saturating_sub(12).max(16);
    let plot_character_height = cell_height.saturating_sub(3).max(1);
    let chart_width = (plot_character_width * 2) as u32;
    let chart_height = (plot_character_height * 4).max(4) as u32;

    let points = chart_points(values);
    let (x_min, x_max) = x_range(&points);
    let shape = Shape::Lines(&points);
    let mut chart = Chart::new(chart_width, chart_height, x_min, x_max);
    let chart = chart.lineplot(&shape);
    chart.axis();
    chart.figures();

    let mut lines = vec![format!("Graph for {name}:")];
    lines.extend(
        chart
            .y_label_format(y_axis_label_format())
            .to_string()
            .lines()
            .map(str::to_owned),
    );
    lines
        .into_iter()
        .take(cell_height)
        .map(|line| line.chars().take(cell_width).collect())
        .collect()
}

fn render_grid(charts: &[Vec<String>], layout: GridLayout) -> String {
    let mut output = String::new();
    for grid_row in 0..layout.rows {
        let start = grid_row * layout.columns;
        let end = (start + layout.columns).min(charts.len());
        let row = &charts[start..end];

        for line_index in 0..layout.cell_height {
            for (column, chart) in row.iter().enumerate() {
                let line = chart.get(line_index).map_or("", String::as_str);
                if column + 1 < row.len() {
                    let _ = write!(output, "{line:<width$} ", width = layout.cell_width);
                } else {
                    output.push_str(line);
                }
            }
            output.push('\n');
        }
    }
    output
}

fn plot(name: &str, values: &[(usize, f32)], width: u32, height: u32) {
    let points = chart_points(values);
    let (x_min, x_max) = x_range(&points);

    println!("Graph for {name}:");
    Chart::new(width, height, x_min, x_max)
        .lineplot(&Shape::Lines(&points))
        .y_label_format(y_axis_label_format())
        .display();
}

fn y_axis_label_format() -> LabelFormat {
    LabelFormat::Custom(Box::new(format_y_axis_value))
}

fn format_y_axis_value(value: f32) -> String {
    let magnitude = value.abs();
    if value == 0.0 {
        "0.00".to_owned()
    } else if !(1e-2..1e5).contains(&magnitude) {
        format!("{value:.2e}")
    } else {
        format!("{value:.2}")
    }
}

fn chart_points(values: &[(usize, f32)]) -> Vec<(f32, f32)> {
    values
        .iter()
        .map(|(timestep, value)| (*timestep as f32, *value))
        .collect()
}

fn x_range(points: &[(f32, f32)]) -> (f32, f32) {
    let first_timestep = points.first().map_or(0.0, |point| point.0);
    let last_timestep = points.last().map_or(1.0, |point| point.0);
    let x_max = if last_timestep > first_timestep {
        last_timestep
    } else {
        first_timestep + 1.0
    };
    (first_timestep, x_max)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{GridLayout, TerminalLogger, format_y_axis_value};

    #[test]
    fn layout_uses_multiple_columns_for_a_wide_terminal() {
        let layout = GridLayout::new(7, 40, 120);

        assert_eq!(layout.columns, 3);
        assert_eq!(layout.rows, 3);
    }

    #[test]
    fn layout_reserves_one_row_and_fits_the_terminal() {
        let layout = GridLayout::new(7, 24, 80);

        assert!(layout.columns * layout.cell_width + layout.columns - 1 <= 80);
        assert!(layout.rows * layout.cell_height <= 23);
    }

    #[test]
    fn rendered_dashboard_fits_the_terminal() {
        let mut logger = TerminalLogger::default();
        for index in 0..7 {
            logger
                .series
                .insert(format!("Metric {index}"), vec![(0, index as f32)]);
        }

        let dashboard = logger.render_dashboard(40, 120);

        assert!(dashboard.lines().count() <= 39);
        assert!(dashboard.lines().all(|line| line.chars().count() <= 120));
    }

    #[test]
    fn refresh_interval_is_configurable() {
        let mut logger =
            TerminalLogger::default().with_refresh_interval(Duration::from_millis(100));
        let last_refresh = Instant::now();
        logger.last_refresh = Some(last_refresh);

        assert_eq!(logger.refresh_interval, Duration::from_millis(100));
        assert!(!logger.refresh_is_due(last_refresh + Duration::from_millis(99)));
        assert!(logger.refresh_is_due(last_refresh + Duration::from_millis(100)));
    }

    #[test]
    fn zero_refresh_interval_is_unthrottled() {
        let mut logger = TerminalLogger::default().with_refresh_interval(Duration::ZERO);
        let last_refresh = Instant::now();
        logger.last_refresh = Some(last_refresh);

        assert!(logger.refresh_is_due(last_refresh));
    }

    #[test]
    fn y_axis_labels_use_scientific_notation_only_for_extreme_values() {
        assert_eq!(format_y_axis_value(0.000_003_2), "3.20e-6");
        assert_eq!(format_y_axis_value(0.02), "0.02");
        assert_eq!(format_y_axis_value(99_999.0), "99999.00");
        assert_eq!(format_y_axis_value(100_000.0), "1.00e5");
        assert_eq!(format_y_axis_value(-0.0), "0.00");
    }
}
