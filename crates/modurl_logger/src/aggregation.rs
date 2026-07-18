use std::collections::{HashMap, VecDeque};

use candle_core::{DType, Tensor};

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Reduction {
    Mean,
    Sum,
    Min,
    Max,
    Last,
}

/// How a metric is reduced within a timestep and smoothed across timesteps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Aggregation {
    reduction: Reduction,
    rolling_window: usize,
}

impl Aggregation {
    pub const fn mean() -> Self {
        Self::new(Reduction::Mean)
    }

    pub const fn sum() -> Self {
        Self::new(Reduction::Sum)
    }

    pub const fn min() -> Self {
        Self::new(Reduction::Min)
    }

    pub const fn max() -> Self {
        Self::new(Reduction::Max)
    }

    pub const fn last() -> Self {
        Self::new(Reduction::Last)
    }

    const fn new(reduction: Reduction) -> Self {
        Self {
            reduction,
            rolling_window: 1,
        }
    }

    /// Applies a rolling average after reducing each timestep.
    #[must_use]
    pub const fn with_rolling_window(mut self, rolling_window: usize) -> Self {
        assert!(
            rolling_window > 0,
            "rolling window must be greater than zero"
        );
        self.rolling_window = rolling_window;
        self
    }

    fn reduce(self, values: &[f32]) -> f32 {
        match self.reduction {
            Reduction::Mean => values.iter().sum::<f32>() / values.len() as f32,
            Reduction::Sum => values.iter().sum(),
            Reduction::Min => values.iter().copied().reduce(f32::min).unwrap(),
            Reduction::Max => values.iter().copied().reduce(f32::max).unwrap(),
            Reduction::Last => *values.last().unwrap(),
        }
    }
}

/// The default aggregation with optional overrides for named metrics.
#[derive(Clone, Debug)]
pub struct AggregationConfig {
    default: Aggregation,
    overrides: HashMap<String, Aggregation>,
}

impl AggregationConfig {
    pub fn new(default: Aggregation) -> Self {
        Self {
            default,
            overrides: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_override(mut self, metric: impl Into<String>, aggregation: Aggregation) -> Self {
        self.overrides.insert(metric.into(), aggregation);
        self
    }

    fn get(&self, metric: &str) -> Aggregation {
        self.overrides.get(metric).copied().unwrap_or(self.default)
    }
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self::new(Aggregation::mean())
    }
}

pub(crate) struct CompletedTimestep {
    pub(crate) timestep: usize,
    pub(crate) metrics: Vec<(String, f32)>,
}

/// Shared aggregation state owned by a logging backend.
///
/// Only the current timestep is incomplete. A larger timestep completes it.
#[derive(Debug)]
pub(crate) struct Aggregator {
    config: AggregationConfig,
    current_timestep: Option<usize>,
    current_values: HashMap<String, Vec<f32>>,
    // Recent finalized values for each metric, bounded by its rolling window.
    window_values: HashMap<String, VecDeque<f32>>,
}

impl Aggregator {
    pub(crate) fn new(config: AggregationConfig) -> Self {
        Self {
            config,
            current_timestep: None,
            current_values: HashMap::new(),
            window_values: HashMap::new(),
        }
    }

    pub(crate) fn log(
        &mut self,
        timestep: usize,
        metrics: &[(&str, &Tensor)],
    ) -> Result<Option<CompletedTimestep>> {
        if let Some(current) = self.current_timestep
            && timestep < current
        {
            return Err(Error::OutOfOrderTimestep {
                current,
                received: timestep,
            });
        }

        let metrics = read_metrics(metrics)?;
        let completed = if self
            .current_timestep
            .is_some_and(|current| timestep > current)
        {
            self.finish()
        } else {
            None
        };

        self.current_timestep = Some(timestep);
        for (name, value) in metrics {
            self.current_values.entry(name).or_default().push(value);
        }

        Ok(completed)
    }

    pub(crate) fn finish(&mut self) -> Option<CompletedTimestep> {
        let timestep = self.current_timestep?;
        self.current_timestep = None;
        let mut completed = Vec::with_capacity(self.current_values.len());

        for (name, values) in self.current_values.drain() {
            let aggregation = self.config.get(&name);
            let window_values = self.window_values.entry(name.clone()).or_default();
            window_values.push_back(aggregation.reduce(&values));
            while window_values.len() > aggregation.rolling_window {
                window_values.pop_front();
            }
            let value = window_values.iter().sum::<f32>() / window_values.len() as f32;
            completed.push((name, value));
        }

        Some(CompletedTimestep {
            timestep,
            metrics: completed,
        })
    }
}

fn read_metrics(metrics: &[(&str, &Tensor)]) -> Result<Vec<(String, f32)>> {
    metrics
        .iter()
        .map(|(name, tensor)| {
            if !tensor.dims().is_empty() {
                return Err(Error::NonScalar {
                    metric: (*name).to_owned(),
                    shape: tensor.dims().to_vec(),
                });
            }
            let value = tensor.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            Ok(((*name).to_owned(), value))
        })
        .collect()
}
