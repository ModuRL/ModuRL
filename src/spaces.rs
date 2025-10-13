use burn::{
    prelude::Backend,
    tensor::{DataError, Device, Distribution, Int, Tensor, TensorKind},
};

use crate::tensor_operations::gen_range_int_tensor;

pub trait Space<const OUTPUT_RANK: usize, const INPUT_RANK: usize> {
    type Error;
    type Backend: Backend;
    type OutputType: TensorKind<Self::Backend>;

    /// Returns a random sample from the space.
    fn sample(
        &self,
        device: &Device<Self::Backend>,
    ) -> Result<Tensor<Self::Backend, OUTPUT_RANK>, Self::Error>;
    /// Returns true if the input tensor is within the space.
    fn contains(&self, x: &Tensor<Self::Backend, OUTPUT_RANK>) -> Result<bool, DataError>;
    /// returns the shape of the space.
    /// This is the gonna be the shape of the tensor that is inputted into the from_neurons function.
    fn shape(&self) -> Vec<usize>;
    /// "Translates" the output of the neurons (of shape [batch_size, shape]) to the space.
    fn from_neurons(
        &self,
        neurons: &Tensor<Self::Backend, INPUT_RANK>,
    ) -> Result<Tensor<Self::Backend, OUTPUT_RANK, Self::OutputType>, Self::Error>;
}

#[derive(Clone)]
pub struct Discrete<B: Backend> {
    possible_values: usize,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Space<0, 0> for Discrete<B> {
    type Error = DataError;
    type Backend = B;
    type OutputType = Int;

    fn sample(&self, device: &Device<B>) -> Result<Tensor<B, 0>, Self::Error> {
        let value = gen_range_int_tensor::<B>(0, self.possible_values as u32 - 1, device)?;
        Ok(Tensor::from_data([value].as_slice(), device))
    }

    fn contains(&self, x: &Tensor<B, 0>) -> Result<bool, Self::Error> {
        let value = x.to_data().into_vec::<u32>();
        if value.is_err() {
            return Ok(false);
        }
        let value = value.unwrap()[0];
        if value < self.possible_values as u32 {
            return Ok(true);
        }
        Ok(false)
    }

    fn shape(&self) -> Vec<usize> {
        if self.possible_values == 1 {
            vec![]
        } else {
            vec![self.possible_values]
        }
    }

    fn from_neurons(
        &self,
        neurons: &Tensor<B, 0>,
    ) -> Result<Tensor<B, 0, Self::OutputType>, Self::Error> {
        // We just need to do an argmax here.
        let output = neurons.clone().argmax(0);
        Ok(output)
    }
}

impl<B> Discrete<B>
where
    B: Backend,
{
    pub fn new(possible_values: usize) -> Self {
        Self {
            possible_values,
            _backend: std::marker::PhantomData,
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
pub struct BoxSpace<B: Backend, const RANK: usize> {
    low: Tensor<B, RANK>,
    high: Tensor<B, RANK>,
    range: Tensor<B, RANK>,
}

impl<B, const RANK: usize> Space<RANK, RANK> for BoxSpace<B, RANK>
where
    B: Backend,
{
    type Error = DataError;
    type Backend = B;
    type OutputType = burn::tensor::Float;

    fn sample(&self, device: &Device<B>) -> Result<Tensor<B, RANK>, Self::Error> {
        // we might be losing some precision here, but it's fine for now.
        let rng_bases = Tensor::random(self.low.shape(), Distribution::Uniform(0.0, 1.0), device);
        let range = self.range.clone();
        let output = (rng_bases * range) + self.low.clone();
        Ok(output)
    }

    fn contains(&self, x: &Tensor<B, RANK>) -> Result<bool, Self::Error> {
        // This is kinda weird, because if the shape is not equal,
        // Should we just say false, or should we return an error?
        if x.shape() != self.low.shape() {
            return Ok(false);
        }
        let is_large_enough = (x.clone() - self.low.clone())
            .min()
            .to_data()
            .into_vec::<f32>()?[0]
            >= 0.0;
        let is_small_enough = (self.high.clone() - x.clone())
            .min()
            .to_data()
            .into_vec::<f32>()?[0]
            >= 0.0;
        Ok(is_large_enough && is_small_enough)
    }

    fn shape(&self) -> Vec<usize> {
        self.low.shape().clone().dims::<RANK>().to_vec()
    }

    fn from_neurons(&self, neurons: &Tensor<B, RANK>) -> Result<Tensor<B, RANK>, Self::Error> {
        // Do not need to do anything here.
        Ok(neurons.clone())
    }
}

#[derive(Debug)]
pub enum BoxSpaceError {
    LowHighShapeMismatch,
    HighLessThanLow,
    DataError(DataError),
}

impl<B: Backend, const RANK: usize> BoxSpace<B, RANK> {
    pub fn new(low: Tensor<B, RANK>, high: Tensor<B, RANK>) -> Result<Self, BoxSpaceError> {
        let low = low.clamp(f32::MIN / 2.0, f32::MAX / 2.0);
        let high = high.clamp(f32::MIN / 2.0, f32::MAX / 2.0);

        if low.shape() != high.shape() {
            return Err(BoxSpaceError::LowHighShapeMismatch);
        }
        let range = high.clone() - low.clone();

        if range
            .clone()
            .min()
            .to_data()
            .to_vec::<f32>()
            .map_err(BoxSpaceError::DataError)?[0]
            < 0.0
        {
            return Err(BoxSpaceError::HighLessThanLow);
        }
        Ok(Self { low, high, range })
    }

    pub fn new_with_uniform_bounds(
        shape: Vec<usize>,
        low: f32,
        high: f32,
        device: &Device<B>,
    ) -> Result<Self, BoxSpaceError> {
        let mut lows = vec![];
        let mut highs = vec![];
        for _ in 0..shape.iter().product::<usize>() {
            lows.push(low);
            highs.push(high);
        }
        let lows = Tensor::<B, RANK>::from_floats(lows.as_slice(), device);
        let highs = Tensor::<B, RANK>::from_floats(highs.as_slice(), device);
        Self::new(lows, highs)
    }

    pub fn new_unbounded(shape: Vec<usize>, device: &Device<B>) -> Result<Self, BoxSpaceError> {
        Self::new_with_uniform_bounds(shape, f32::NEG_INFINITY, f32::INFINITY, device)
    }
}

#[cfg(test)]
mod tests {
    use burn_ndarray::NdArray;

    use super::*;

    #[test]
    fn test_box_space() {
        type B = NdArray;
        let device = Device::<B>::Cpu;

        let low = Tensor::<B, 1>::from_floats([-1.0, -2.0, -3.0].as_slice(), &device);
        let high = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0].as_slice(), &device);
        let box_space = BoxSpace::<B, 1>::new(low, high).unwrap();

        for _ in 0..100 {
            let sample = box_space.sample(&device).unwrap();
            assert!(box_space.contains(&sample).unwrap());
        }

        let invalid_low = Tensor::<B, 1>::from_floats([1.0, 2.0].as_slice(), &device);
        let invalid_high = Tensor::<B, 1>::from_floats([0.0, 1.0].as_slice(), &device);
        assert!(BoxSpace::<B, 1>::new(invalid_low, invalid_high).is_err());

        let infinite_box = BoxSpace::<B, 1>::new_with_uniform_bounds(
            vec![3],
            f32::NEG_INFINITY,
            f32::INFINITY,
            &device,
        )
        .unwrap();

        for _ in 0..100 {
            let sample = infinite_box.sample(&device).unwrap();
            assert!(infinite_box.contains(&sample).unwrap());
        }
    }
}
