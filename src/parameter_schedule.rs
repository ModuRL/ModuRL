/// A minimal scalar parameter schedule trait.
pub trait ParameterSchedule {
    fn value(&self, t: f64) -> f64;
}

/// Tracks progress through a fixed training horizon for parameter schedules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleProgress {
    training_horizon: usize,
    elapsed_steps: usize,
}

impl ScheduleProgress {
    pub fn new(training_horizon: usize) -> Self {
        assert!(
            training_horizon > 0,
            "training horizon must be greater than zero"
        );
        Self {
            training_horizon,
            elapsed_steps: 0,
        }
    }

    pub fn set_elapsed_steps(&mut self, elapsed_steps: usize) {
        self.elapsed_steps = elapsed_steps;
    }

    pub fn elapsed_steps(&self) -> usize {
        self.elapsed_steps
    }

    pub fn advance_steps(&mut self, steps: usize) {
        self.elapsed_steps = self.elapsed_steps.saturating_add(steps);
    }

    pub fn get_elapsed_fraction(&self) -> f64 {
        (self.elapsed_steps as f64 / self.training_horizon as f64).clamp(0.0, 1.0)
    }

    pub fn parameter(&self, schedule: &dyn ParameterSchedule) -> f64 {
        schedule.value(self.get_elapsed_fraction())
    }
}

impl<F> ParameterSchedule for F
where
    F: Fn(f64) -> f64,
{
    fn value(&self, t: f64) -> f64 {
        self(t)
    }
}

/// A constant schedule that always returns the same value.
#[derive(Clone)]
pub struct ConstantSchedule {
    value: f64,
}

impl ConstantSchedule {
    pub fn new(value: f64) -> Self {
        Self { value }
    }
}

impl ParameterSchedule for ConstantSchedule {
    fn value(&self, _t: f64) -> f64 {
        self.value
    }
}

/// Linear schedule that interpolates between two values.
#[derive(Clone)]
pub struct LinearSchedule {
    start: f64,
    end: f64,
}

impl LinearSchedule {
    pub fn new(start: f64, end: f64) -> Self {
        Self { start, end }
    }
}

impl ParameterSchedule for LinearSchedule {
    fn value(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        self.start + (self.end - self.start) * t
    }
}

/// Exponential schedule that interpolates between two non-zero values of the same sign.
#[derive(Clone)]
pub struct ExponentialSchedule {
    start: f64,
    end: f64,
}

impl ExponentialSchedule {
    /// Creates a schedule that returns `start` at progress 0 and `end` at
    /// progress 1.
    ///
    /// Both values must be non-zero and have the same sign.
    pub fn new(start: f64, end: f64) -> Self {
        assert!(
            start != 0.0 && end != 0.0,
            "exponential endpoints must be non-zero"
        );
        assert!(
            start.is_sign_positive() == end.is_sign_positive(),
            "exponential endpoints must have the same sign"
        );
        Self { start, end }
    }
}

impl ParameterSchedule for ExponentialSchedule {
    fn value(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        let magnitude = self.start.abs() * (self.end.abs() / self.start.abs()).powf(t);
        self.start.signum() * magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::{ExponentialSchedule, LinearSchedule, ParameterSchedule, ScheduleProgress};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual {actual} differs from expected {expected}"
        );
    }

    #[test]
    fn linear_schedule_interpolates_between_start_and_end() {
        let schedule = LinearSchedule::new(1.0, 0.1);

        assert_close(schedule.value(0.0), 1.0);
        assert_close(schedule.value(0.5), 0.55);
        assert_close(schedule.value(1.0), 0.1);
    }

    #[test]
    fn linear_schedule_clamps_progress() {
        let schedule = LinearSchedule::new(1.0, 0.1);

        assert_close(schedule.value(-1.0), 1.0);
        assert_close(schedule.value(2.0), 0.1);
    }

    #[test]
    fn exponential_schedule_interpolates_between_positive_endpoints() {
        let schedule = ExponentialSchedule::new(1.0, 0.01);

        assert_close(schedule.value(0.0), 1.0);
        assert_close(schedule.value(0.5), 0.1);
        assert_close(schedule.value(1.0), 0.01);
    }

    #[test]
    fn exponential_schedule_clamps_progress() {
        let schedule = ExponentialSchedule::new(1.0, 0.1);

        assert_close(schedule.value(-1.0), 1.0);
        assert_close(schedule.value(2.0), 0.1);
    }

    #[test]
    fn schedule_progress_tracks_and_clamps_elapsed_fraction() {
        let mut progress = ScheduleProgress::new(100);
        assert_close(progress.get_elapsed_fraction(), 0.0);

        progress.set_elapsed_steps(25);
        assert_eq!(progress.elapsed_steps(), 25);
        assert_close(progress.get_elapsed_fraction(), 0.25);
        assert_close(progress.parameter(&LinearSchedule::new(1.0, 0.0)), 0.75);

        progress.advance_steps(25);
        assert_eq!(progress.elapsed_steps(), 50);
        assert_close(progress.get_elapsed_fraction(), 0.5);

        progress.set_elapsed_steps(150);
        assert_close(progress.get_elapsed_fraction(), 1.0);
    }
}
