/// A minimal learning rate scheduler trait.
pub trait LrScheduler {
    fn get_lr(&self, t: f64) -> f64;
}

impl<F> LrScheduler for F
where
    F: Fn(f64) -> f64,
{
    fn get_lr(&self, t: f64) -> f64 {
        self(t)
    }
}
