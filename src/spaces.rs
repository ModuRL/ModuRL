use ndarray::ArrayD;
use rand::{rng, Rng};

pub trait Space<T> {
    fn sample(&self) -> T;
    fn contains(&self, x: &T) -> bool;
}

pub struct Discrete {
    dims: Vec<usize>,
}

impl Space<ArrayD<bool>> for Discrete {
    fn sample(&self) -> ArrayD<bool> {
        let total_num_of_bools = self.dims.iter().product::<usize>();
        let mut bools = vec![];
        let mut rng = rng();
        for _ in 0..(total_num_of_bools / 64) {
            let chunk = rng.random::<u64>();
            for i in 0..64 {
                if bools.len() < total_num_of_bools {
                    bools.push((chunk >> i) & 1 == 1);
                }
            }
        }
        ArrayD::from_shape_vec(self.dims.clone(), bools).unwrap()
    }

    fn contains(&self, x: &ArrayD<bool>) -> bool {
        // This is kinda weird, because if the shape is not equal,
        // Should we just say false, or should we return an error?
        if x.shape() != self.dims {
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
pub struct Box {
    low: ArrayD<f32>,
    high: ArrayD<f32>,
}

impl Space<ArrayD<f32>> for Box {
    fn sample(&self) -> ArrayD<f32> {
        let mut rng = rng();
        let mut values = vec![];
        for i in 0..self.low.len() {
            values.push(rng.random_range(self.low[i]..self.high[i]));
        }
        ArrayD::from_shape_vec(self.low.shape(), values).unwrap()
    }

    fn contains(&self, x: &ArrayD<f32>) -> bool {
        if x.shape() != self.low.shape() {
            return false;
        }
        for i in 0..x.len() {
            if x[i] < self.low[i] || x[i] > self.high[i] {
                return false;
            }
        }
        true
    }
}

impl Box {
    pub fn new(low: ArrayD<f32>, high: ArrayD<f32>) -> Self {
        Self { low, high }
    }

    pub fn new_with_universal_bounds(shape: Vec<usize>) -> Self {
        let mut low = vec![];
        let mut high = vec![];
        for _ in 0..shape.iter().product::<usize>() {
            low.push(f32::NEG_INFINITY);
            high.push(f32::INFINITY);
        }
        Self {
            low: ArrayD::from_shape_vec(shape.clone(), low).unwrap(),
            high: ArrayD::from_shape_vec(shape, high).unwrap(),
        }
    }
}
