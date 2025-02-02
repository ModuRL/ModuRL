use candle_core::{Device, Tensor};
use rand::{rng, Rng};

pub trait Space {
    fn sample(&self, device: &Device) -> Tensor;
    fn contains(&self, x: &Tensor) -> bool;
}

#[derive(Clone)]
pub struct Discrete {
    dims: Vec<usize>,
}

impl Space for Discrete {
    fn sample(&self, device: &Device) -> Tensor {
        let total_num_of_bools = self.dims.iter().product::<usize>();
        let mut bools = vec![];
        let mut rng = rng();
        for _ in 0..total_num_of_bools {
            bools.push(rng.random_bool(0.5) as u32 as f32);
        }
        Tensor::from_vec(bools, self.dims.clone(), &device).expect("Failed to create tensor.")
    }

    fn contains(&self, x: &Tensor) -> bool {
        // This is kinda weird, because if the shape is not equal,
        // Should we just say false, or should we return an error?
        if *x.shape() != candle_core::Shape::from_dims(self.dims.as_slice()) {
            return false;
        }
        true
    }
}

impl Discrete {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }
}

/// Box
/// A box space is a bounded, n-dimensional space.
/// The bounds are defined by two vectors: low and high.
// I don't like forcing the user to use f32, but this is what we'll do for now.
#[derive(Clone)]
pub struct BoxSpace {
    low: Tensor,
    high: Tensor,
}

impl Space for BoxSpace {
    fn sample(&self, device: &Device) -> Tensor {
        let mut rng = rng();
        let mut values = vec![];
        // flatten the low and high tensors
        let low = self.low.flatten_all().expect("Failed to flatten tensor.");
        let high = self.high.flatten_all().expect("Failed to flatten tensor.");
        for i in 0..low.shape().dim(0).expect("Failed to get dim.") {
            let low = low
                .get(i)
                .expect("Failed to get value.")
                .to_vec0::<f32>()
                .expect("Failed to convert to f32.");
            let high = high
                .get(i)
                .expect("Failed to get value.")
                .to_vec0::<f32>()
                .expect("Failed to convert to f32.");
            values.push(rng.random_range(low..high));
        }
        Tensor::from_vec(values, low.shape(), device).expect("Failed to create tensor.")
    }

    fn contains(&self, x: &Tensor) -> bool {
        // This is kinda weird, because if the shape is not equal,
        // Should we just say false, or should we return an error?
        if *x.shape() != *self.low.shape() {
            return false;
        }
        let low = self.low.flatten_all().expect("Failed to flatten tensor.");
        let high = self.high.flatten_all().expect("Failed to flatten tensor.");
        // So inefficient :*(
        for i in 0..low.shape().dim(0).expect("Failed to get dim.") {
            let low = low
                .get(i)
                .expect("Failed to get value.")
                .to_vec0::<f32>()
                .expect("Failed to convert to f32.");
            let high = high
                .get(i)
                .expect("Failed to get value.")
                .to_vec0::<f32>()
                .expect("Failed to convert to f32.");
            let value = x
                .get(i)
                .expect("Failed to get value.")
                .to_vec0::<f32>()
                .expect("Failed to convert to f32.");
            if value < low || value > high {
                return false;
            }
        }
        true
    }
}

impl BoxSpace {
    pub fn new(low: Tensor, high: Tensor) -> Self {
        assert!(low.shape() == high.shape());
        Self { low, high }
    }

    pub fn new_with_universal_bounds(
        shape: Vec<usize>,
        low: f32,
        high: f32,
        device: &Device,
    ) -> Self {
        let mut lows = vec![];
        let mut highs = vec![];
        for _ in 0..shape.iter().product::<usize>() {
            lows.push(low);
            highs.push(high);
        }
        let lows = Tensor::from_vec(lows, shape.clone(), device).expect("Failed to create tensor.");
        let highs =
            Tensor::from_vec(highs, shape.clone(), device).expect("Failed to create tensor.");
        Self {
            low: lows,
            high: highs,
        }
    }

    pub fn new_unbounded(shape: Vec<usize>, device: &Device) -> Self {
        Self::new_with_universal_bounds(shape, f32::NEG_INFINITY, f32::INFINITY, device)
    }
}
