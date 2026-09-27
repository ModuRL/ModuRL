use candle_core::{D, Device, Tensor};

pub trait Space {
    type Error;

    /// Samples `[batch_size, ...action_shape]` using the requested device RNG.
    /// Discrete actions have shape `[batch_size]`; box actions have shape
    /// `[batch_size, ...self.shape()]`.
    fn sample_batch(&self, batch_size: usize, device: &Device) -> Result<Tensor, Self::Error>;

    /// Returns true if `x` has the concrete space's unbatched environment
    /// action shape and is within the space.
    fn contains(&self, x: &Tensor) -> bool;
    /// Returns the shape of one latent policy output consumed by
    /// [`Space::tensor_from_neurons`].
    fn shape(&self) -> Vec<usize>;
    /// Converts latent policy output shaped `[batch_size, ...self.shape()]`
    /// into batched environment actions. Concrete spaces document the output
    /// shape.
    ///
    /// A continuous [`BoxSpace`] clamps each component to its bounds. PPO keeps
    /// its original latent action separately for probability calculations.
    fn tensor_from_neurons(&self, neurons: &Tensor) -> Result<Tensor, Self::Error>;
}

#[derive(Clone)]
pub struct Discrete {
    possible_values: usize,
}

impl Space for Discrete {
    type Error = candle_core::Error;

    fn sample_batch(&self, batch_size: usize, device: &Device) -> Result<Tensor, Self::Error> {
        if self.possible_values == 0 || self.possible_values > (1 << 24) {
            return Err(candle_core::Error::Msg(
                "discrete sampling requires 1..=2^24 categories".into(),
            )
            .into());
        }
        // Preserve one scalar RNG draw per environment, including stream advancement.
        let draws = (0..batch_size)
            .map(|_| Tensor::rand(0.0_f32, self.possible_values as f32, (), device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Tensor::stack(&draws, 0)?
            .floor()?
            .to_dtype(candle_core::DType::U32)?)
    }

    /// Tests one scalar environment action `x` shaped `[]`.
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

    /// Returns latent policy shape `[action_count]` (or `[]` for a one-action
    /// space).
    fn shape(&self) -> Vec<usize> {
        if self.possible_values == 1 {
            vec![]
        } else {
            vec![self.possible_values]
        }
    }

    /// Converts logits `neurons` shaped `[batch_size, action_count]` into
    /// scalar action indices shaped `[batch_size]`.
    fn tensor_from_neurons(&self, neurons: &Tensor) -> Result<Tensor, Self::Error> {
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

/// A bounded, n-dimensional continuous space.
///
/// The `low` and `high` tensors define the inclusive component bounds.
/// [`Space::tensor_from_neurons`] clamps policy outputs to these bounds before
/// they are sent to an environment.
#[derive(Clone)]
pub struct BoxSpace {
    low: Tensor,
    high: Tensor,
}

impl Space for BoxSpace {
    type Error = candle_core::Error;

    fn sample_batch(&self, batch_size: usize, device: &Device) -> Result<Tensor, Self::Error> {
        let dtype = self.low.dtype();
        // Preserve F64 arithmetic and the existing finite substitutes for
        // infinite bounds, without scalar device-to-host reads.
        let prepare = |bounds: &Tensor| -> candle_core::Result<Tensor> {
            let bounds = bounds
                .to_device(device)?
                .to_dtype(candle_core::DType::F64)?;
            let positive = bounds.eq(f64::INFINITY)?;
            let negative = bounds.eq(f64::NEG_INFINITY)?;
            let limit = finitize(f64::INFINITY, dtype);
            let bounds =
                positive.where_cond(&Tensor::full(limit, bounds.shape(), device)?, &bounds)?;
            negative.where_cond(&Tensor::full(-limit, bounds.shape(), device)?, &bounds)
        };
        let low = prepare(&self.low)?;
        let high = prepare(&self.high)?;
        let mut shape = vec![batch_size];
        shape.extend(self.shape());
        // Preserve the original flattened RNG draw per environment. Combining
        // draws changes stream advancement on device backends.
        let draws = (0..batch_size)
            .map(|_| Tensor::rand(0.0_f64, 1.0_f64, self.low.elem_count(), device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let random = Tensor::stack(&draws, 0)?.reshape(shape.as_slice())?;
        Ok(random
            .broadcast_mul(&(&high - &low)?)?
            .broadcast_add(&low)?
            .to_dtype(dtype)?)
    }

    /// Tests one environment action `x` shaped `self.shape()`.
    fn contains(&self, x: &Tensor) -> bool {
        // This is kinda weird, because if the shape is not equal,
        // Should we just say false, or should we return an error?
        if *x.shape() != *self.low.shape() {
            return false;
        }
        let low = self
            .low
            .to_dtype(candle_core::DType::F64)
            .and_then(|tensor| tensor.flatten_all())
            .expect("Failed to flatten tensor.");
        let high = self
            .high
            .to_dtype(candle_core::DType::F64)
            .and_then(|tensor| tensor.flatten_all())
            .expect("Failed to flatten tensor.");
        let x = x
            .to_dtype(candle_core::DType::F64)
            .and_then(|tensor| tensor.flatten_all())
            .expect("Failed to flatten tensor.");
        // One read per tensor rather than one read per component.
        let low = low.to_vec1::<f64>().expect("Failed to read bounds.");
        let high = high.to_vec1::<f64>().expect("Failed to read bounds.");
        let values = x.to_vec1::<f64>().expect("Failed to read action.");
        values
            .iter()
            .zip(low)
            .zip(high)
            .all(|((&value, low), high)| !(value < low || value > high))
    }

    fn shape(&self) -> Vec<usize> {
        self.low.shape().clone().into_dims()
    }

    /// Clips `neurons` shaped `[batch_size, ...self.shape()]`, preserving that
    /// shape.
    fn tensor_from_neurons(&self, neurons: &Tensor) -> Result<Tensor, Self::Error> {
        // Continuous policies (for example, an unsquashed Gaussian) may
        // produce values outside the environment's bounds. Keep the latent
        // action unchanged for probability calculations, but clip the action
        // that is actually sent to the environment.
        let low = self
            .low
            .to_device(neurons.device())?
            .to_dtype(neurons.dtype())?;
        let high = self
            .high
            .to_device(neurons.device())?
            .to_dtype(neurons.dtype())?;
        neurons
            .broadcast_maximum(&low)
            .and_then(|values| values.broadcast_minimum(&high))
    }
}

impl BoxSpace {
    /// Creates a box whose `low` and `high` bounds have the same
    /// `action_shape`, which becomes the shape of one unbatched value.
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
fn finitize(value: f64, dtype: candle_core::DType) -> f64 {
    let finite_limit = if dtype == candle_core::DType::F64 {
        f64::MAX / 2.0
    } else {
        f64::from(f32::MAX / 2.0)
    };
    if value == f64::INFINITY {
        return finite_limit;
    }
    if value == f64::NEG_INFINITY {
        return -finite_limit;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_sampling_preserves_device_rng_draws_and_advancement() {
        let device = Device::cuda_if_available(0).unwrap();
        if device.is_cpu() {
            return;
        }
        let space = BoxSpace::new_with_universal_bounds(vec![3], 0.0, 1.0, &device);
        device.set_seed(123).unwrap();
        let expected = (0..5)
            .map(|_| Tensor::rand(0.0_f64, 1.0_f64, 3, &device).unwrap())
            .collect::<Vec<_>>();
        let expected = Tensor::stack(&expected, 0)
            .unwrap()
            .to_dtype(space.low.dtype())
            .unwrap();
        let next = Tensor::rand(0.0_f32, 1.0, 7, &device).unwrap();
        device.set_seed(123).unwrap();
        let actual = space.sample_batch(5, &device).unwrap();
        let actual_next = Tensor::rand(0.0_f32, 1.0, 7, &device).unwrap();
        assert_eq!(
            actual.to_vec2::<f32>().unwrap(),
            expected.to_vec2::<f32>().unwrap()
        );
        assert_eq!(
            actual_next.to_vec1::<f32>().unwrap(),
            next.to_vec1::<f32>().unwrap()
        );

        device.set_seed(456).unwrap();
        let expected = (0..5)
            .map(|_| crate::sampling::sample_u32_inclusive(0, 4, &device).unwrap())
            .collect::<Vec<_>>();
        let next = Tensor::rand(0.0_f32, 1.0, 7, &device).unwrap();
        device.set_seed(456).unwrap();
        let actual = Discrete::new(5).sample_batch(5, &device).unwrap();
        let actual_next = Tensor::rand(0.0_f32, 1.0, 7, &device).unwrap();
        assert_eq!(actual.to_vec1::<u32>().unwrap(), expected);
        assert_eq!(
            actual_next.to_vec1::<f32>().unwrap(),
            next.to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn batched_sampling_preserves_shapes_bounds_and_dtypes() {
        for device in [Device::Cpu, Device::cuda_if_available(0).unwrap()] {
            for dtype in [candle_core::DType::F32, candle_core::DType::F64] {
                let low = Tensor::new(&[[-2.0f64, 3.0], [7.0, -1.0]], &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let high = Tensor::new(&[[2.0f64, 3.0], [8.0, 0.0]], &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let space = BoxSpace::new(low, high);
                let samples = space.sample_batch(32, &device).unwrap();
                assert_eq!(samples.dims(), &[32, 2, 2]);
                assert_eq!(samples.dtype(), dtype);
                for row in 0..32 {
                    assert!(space.contains(&samples.get(row).unwrap()));
                }
                assert_eq!(
                    space
                        .sample_batch(1, &device)
                        .unwrap()
                        .squeeze(0)
                        .unwrap()
                        .dims(),
                    &[2, 2]
                );
                let unbounded = BoxSpace::new(
                    Tensor::new(&[f64::NEG_INFINITY], &device)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap(),
                    Tensor::new(&[f64::INFINITY], &device)
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap(),
                );
                assert!(
                    unbounded
                        .sample_batch(32, &device)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_dtype(candle_core::DType::F64)
                        .unwrap()
                        .to_vec1::<f64>()
                        .unwrap()
                        .iter()
                        .all(|value| value.is_finite())
                );
            }
            let actions = Discrete::new(5).sample_batch(128, &device).unwrap();
            assert_eq!(actions.dims(), &[128]);
            assert!(
                actions
                    .to_vec1::<u32>()
                    .unwrap()
                    .iter()
                    .all(|&value| value < 5)
            );
            assert!(Discrete::new(0).sample_batch(1, &device).is_err());
        }
    }

    #[test]
    fn box_space_clips_batched_policy_outputs_to_bounds() {
        let space = BoxSpace::new_with_universal_bounds(vec![3], -1.0, 1.0, &Device::Cpu);
        let neurons = Tensor::from_vec(
            vec![-2.0_f32, -0.5, 0.25, 0.75, 1.5, 3.0],
            (2, 3),
            &Device::Cpu,
        )
        .unwrap();

        let actions = space.tensor_from_neurons(&neurons).unwrap();

        assert_eq!(
            actions.to_vec2::<f32>().unwrap(),
            vec![vec![-1.0, -0.5, 0.25], vec![0.75, 1.0, 1.0]]
        );
    }

    #[test]
    fn box_space_clips_using_policy_output_dtype() {
        let space = BoxSpace::new_with_universal_bounds(vec![3], -1.0, 1.0, &Device::Cpu);
        let neurons = Tensor::from_vec(
            vec![-2.0_f64, -0.5, 0.25, 0.75, 1.5, 3.0],
            (2, 3),
            &Device::Cpu,
        )
        .unwrap();

        let actions = space.tensor_from_neurons(&neurons).unwrap();

        assert_eq!(actions.dtype(), candle_core::DType::F64);
        assert_eq!(
            actions.to_vec2::<f64>().unwrap(),
            vec![vec![-1.0, -0.5, 0.25], vec![0.75, 1.0, 1.0]]
        );
    }

    #[test]
    fn box_space_samples_are_f32() {
        let space = BoxSpace::new_with_universal_bounds(vec![3], -1.0, 1.0, &Device::Cpu);
        let sample = space
            .sample_batch(1, &Device::Cpu)
            .unwrap()
            .squeeze(0)
            .unwrap();

        assert_eq!(sample.dtype(), candle_core::DType::F32);
        assert!(space.contains(&sample));
    }

    #[test]
    fn box_space_preserves_f64_sampling_and_boundary_precision() {
        let low = Tensor::new(&[0.0_f64], &Device::Cpu).unwrap();
        let high = Tensor::new(&[1.0_f64], &Device::Cpu).unwrap();
        let space = BoxSpace::new(low, high);

        let sample = space
            .sample_batch(1, &Device::Cpu)
            .unwrap()
            .squeeze(0)
            .unwrap();
        let just_outside = Tensor::new(&[1.0_f64 + 1e-12], &Device::Cpu).unwrap();

        assert_eq!(sample.dtype(), candle_core::DType::F64);
        assert!(space.contains(&sample));
        assert!(!space.contains(&just_outside));
    }
}
