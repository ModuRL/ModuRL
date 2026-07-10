/// A minimal scalar parameter schedule trait.
pub trait ParameterSchedule {
    fn value(&self, t: f64) -> f64;
}

impl<F> ParameterSchedule for F
where
    F: Fn(f64) -> f64,
{
    fn value(&self, t: f64) -> f64 {
        self(t)
    }
}

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

#[cfg(test)]
mod tests {
    use super::{LinearSchedule, ParameterSchedule};

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
}
