use candle_core::{Device, Tensor};
use rand::{rng, Rng};

pub trait Space {
    /// Returns a random sample from the space.
    fn sample(&self, device: &Device) -> Tensor;
    /// Returns true if the input tensor is within the space.
    fn contains(&self, x: &Tensor) -> bool;
    /// returns the shape of the space.
    /// This is the gonna be the shape of the tensor that is inputted into the from_neurons function.
    fn shape(&self) -> Vec<usize>;
    /// "Translates" the output of the neurons (of shape [batch_size, shape]) to the space.
    fn from_neurons(&self, neurons: &Tensor) -> Tensor;
}

#[derive(Clone)]
pub struct Discrete {
    possible_values: usize,
    start_value: i32, // inclusive
}

impl Space for Discrete {
    fn sample(&self, device: &Device) -> Tensor {
        let mut rng = rng();
        let value = rng
            .random_range(self.start_value..(self.start_value + self.possible_values as i32))
            as u32;
        Tensor::from_vec(vec![value], vec![], device).expect("Failed to create tensor.")
    }

    fn contains(&self, x: &Tensor) -> bool {
        if x.dims() != Vec::<usize>::new() {
            return false;
        }
        let value = x.to_vec0::<u32>().expect("Failed to convert to u32.");
        value >= self.start_value as u32
            && value < (self.start_value + self.possible_values as i32) as u32
    }

    fn shape(&self) -> Vec<usize> {
        if self.possible_values == 1 {
            vec![]
        } else {
            vec![self.possible_values]
        }
    }

    fn from_neurons(&self, neurons: &Tensor) -> Tensor {
        let neurons = neurons.argmax(1).expect("Failed to get argmax.");
        neurons
    }
}

impl Discrete {
    pub fn new(possible_values: usize, start_value: i32) -> Self {
        Self {
            possible_values,
            start_value,
        }
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
    fn sample(&self, device: &Device) -> Tensor {
        let mut rng = rng();
        let mut values = vec![];
        // flatten the low and high tensors
        let low = self.low.flatten_all().expect("Failed to flatten tensor.");
        let high = self.high.flatten_all().expect("Failed to flatten tensor.");
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

    fn from_neurons(&self, neurons: &Tensor) -> Tensor {
        // Do not need to do anything here.
        neurons.clone()
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
