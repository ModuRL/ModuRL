use candle_core::{D, Device, Tensor};

use crate::tensor_operations::gen_range_int_tensor;

pub trait Space {
    type Error;

    /// Returns a random sample from the space.
    fn sample(&self, device: &Device) -> Result<Tensor, Self::Error>;
    /// Returns true if the input tensor is within the space.
    fn contains(&self, x: &Tensor) -> bool;
    /// returns the shape of the space.
    /// This is the gonna be the shape of the tensor that is inputted into the from_neurons function.
    fn shape(&self) -> Vec<usize>;
    /// "Translates" the output of the neurons (of shape [batch_size, shape]) to the space.
    fn from_neurons(&self, neurons: &Tensor) -> Result<Tensor, Self::Error>;
}

#[derive(Clone)]
pub struct Discrete {
    possible_values: usize,
}

impl Space for Discrete {
    type Error = candle_core::Error;

    fn sample(&self, device: &Device) -> Result<Tensor, Self::Error> {
        let value = gen_range_int_tensor(0, self.possible_values as u32 - 1, device)?;
        Tensor::from_vec(vec![value], vec![], device)
    }

    fn contains(&self, x: &Tensor) -> bool {
        if x.dims() != Vec::<usize>::new() {
            return false;
        }
        let value = x.to_vec0::<u32>().expect("Failed to convert to u32.");
        if value < self.possible_values as u32 {
            return true;
        }
        false
    }

    fn shape(&self) -> Vec<usize> {
        if self.possible_values == 1 {
            vec![]
        } else {
            vec![self.possible_values]
        }
    }

    fn from_neurons(&self, neurons: &Tensor) -> Result<Tensor, Self::Error> {
        neurons.argmax(D::Minus1)
    }
}

impl Discrete {
    pub fn new(possible_values: usize) -> Self {
        Self { possible_values }
    }

    pub fn get_possible_values(&self) -> usize {
        self.possible_values
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
    type Error = candle_core::Error;

    fn sample(&self, device: &Device) -> Result<Tensor, Self::Error> {
        let mut values = vec![];

        // flatten the low and high tensors
        let low = self.low.flatten_all().expect("Failed to flatten tensor.");
        let high = self.high.flatten_all().expect("Failed to flatten tensor.");
        // we might be losing some precision here, but it's fine for now.
        let rng_bases = Tensor::rand(0.0, 1.0, low.shape(), device)?;
        for i in 0..low.shape().dim(0).expect("Failed to get dim.") {
            let mut low = low
                .get(i)
                .expect("Failed to get value.")
                .to_vec0::<f32>()
                .expect("Failed to convert to f32.");
            let mut high = high
                .get(i)
                .expect("Failed to get value.")
                .to_vec0::<f32>()
                .expect("Failed to convert to f32.");
            low = finitize(low);
            high = finitize(high);

            let rng_base = rng_bases
                .get(i)
                .expect("Failed to get value.")
                .to_vec0::<f32>()
                .expect("Failed to convert to f32.");
            let adjusted_range = high - low;
            let rand_num = low + adjusted_range * rng_base;

            values.push(rand_num);
        }
        Tensor::from_vec(values, low.shape(), device)
    }

    fn contains(&self, x: &Tensor) -> bool {
        // This is kinda weird, because if the shape is not equal,
        // Should we just say false, or should we return an error?
        if *x.shape() != *self.low.shape() {
            return false;
        }
        let low = self.low.flatten_all().expect("Failed to flatten tensor.");
        let high = self.high.flatten_all().expect("Failed to flatten tensor.");
        let x = x.flatten_all().expect("Failed to flatten tensor.");
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

    fn shape(&self) -> Vec<usize> {
        self.low.shape().clone().into_dims()
    }

    fn from_neurons(&self, neurons: &Tensor) -> Result<Tensor, Self::Error> {
        // Do not need to do anything here.
        Ok(neurons.clone())
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
        Self::new(lows, highs)
    }

    pub fn new_unbounded(shape: Vec<usize>, device: &Device) -> Self {
        Self::new_with_universal_bounds(shape, f32::NEG_INFINITY, f32::INFINITY, device)
    }
}

// A helper function to finitize the value.
// This way random_range can work with f32::INFINITY and f32::NEG_INFINITY.
// We just return f32::MAX / 2.0 and f32::MIN / 2.0 respectively.
// It's pretty dumb, but it lies and says min and max aren't finite otherwise.
fn finitize(value: f32) -> f32 {
    if value == f32::INFINITY {
        return f32::MAX / 2.0;
    }
    if value == f32::NEG_INFINITY {
        return f32::MIN / 2.0;
    }
    value
}
