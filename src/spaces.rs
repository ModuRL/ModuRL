use burn::tensor::{DType, Device, Distribution, Float, Int, Tensor, kind::Basic};

/// Checks which rank-`O` observation batches are valid.
/// A single environment uses a batch of one.
pub trait ObservationSpace<const O: usize> {
    type Kind: Basic;
    /// Checks observations `[batch, ...self.shape()]` against the shape and bounds.
    fn contains(&self, values: &Tensor<O, Self::Kind>) -> bool;
    /// Returns the shape of one environment value, excluding the batch axis.
    fn shape(&self) -> Vec<usize>;
}

/// Checks valid environment actions and supplies random action batches.
/// A single environment uses a batch of one.
pub trait ActionSpace<const A: usize> {
    type Kind: Basic;
    type Error;

    /// Checks actions `[batch, ...self.shape()]` against the shape and bounds.
    fn contains(&self, values: &Tensor<A, Self::Kind>) -> bool;
    /// Returns the shape of one environment action, excluding the batch axis.
    fn shape(&self) -> Vec<usize>;

    /// Samples actions `[batch_size, ...self.shape()]` using the device RNG.
    fn sample_batch(
        &self,
        batch_size: usize,
        device: &Device,
    ) -> Result<Tensor<A, Self::Kind>, Self::Error>;
}

/// Converts original policy samples of rank `L` into environment actions of rank `A`.
pub trait ActionMap<const L: usize, const A: usize>: ActionSpace<A> {
    /// Returns the shape of one policy sample, excluding the batch axis.
    fn policy_shape(&self) -> Vec<usize>;
    /// Converts policy samples `[batch, ...self.policy_shape()]` of rank `L`
    /// into environment actions `[batch, ...self.shape()]` of rank `A`.
    /// Continuous actions are clamped; callers retain original samples separately
    /// for probability calculations.
    fn tensor_from_neurons(&self, neurons: Tensor<L>)
    -> Result<Tensor<A, Self::Kind>, Self::Error>;
}

#[derive(Debug, thiserror::Error)]
pub enum SpaceError {
    #[error("discrete sampling requires 1..=2^24 categories, got {0}")]
    InvalidCategoryCount(usize),
    #[error("invalid policy output shape: expected {expected:?}, got {actual:?}")]
    InvalidPolicyShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// Integer actions in `0..possible_values` with rank-2 category logits.
#[derive(Clone)]
pub struct Discrete {
    possible_values: usize,
}

impl Discrete {
    /// Checks that every index in `[batch]` is in `0..possible_values`.
    pub fn contains(&self, values: &Tensor<1, Int>) -> bool {
        values
            .clone()
            .greater_equal_scalar(0)
            .bool_and(values.clone().lower_elem(self.possible_values as i64))
            .all()
            .into_scalar::<bool>()
    }

    /// Discrete environment values are scalar indices.
    pub fn shape(&self) -> Vec<usize> {
        vec![]
    }
}

impl ObservationSpace<1> for Discrete {
    type Kind = Int;

    fn contains(&self, values: &Tensor<1, Int>) -> bool {
        self.contains(values)
    }
    fn shape(&self) -> Vec<usize> {
        self.shape()
    }
}

impl ActionSpace<1> for Discrete {
    type Kind = Int;
    type Error = SpaceError;

    fn contains(&self, values: &Tensor<1, Int>) -> bool {
        self.contains(values)
    }
    fn shape(&self) -> Vec<usize> {
        self.shape()
    }

    /// Samples integer action indices shaped `[batch_size]`.
    fn sample_batch(
        &self,
        batch_size: usize,
        device: &Device,
    ) -> Result<Tensor<1, Int>, Self::Error> {
        if self.possible_values == 0 || self.possible_values > (1 << 24) {
            return Err(SpaceError::InvalidCategoryCount(self.possible_values));
        }
        Ok(Tensor::random(
            [batch_size],
            Distribution::Uniform(0.0, self.possible_values as f64),
            (device, DType::I32),
        ))
    }
}

impl ActionMap<2, 1> for Discrete {
    /// Returns `[action_count]`, including `[1]` for a one-action space.
    fn policy_shape(&self) -> Vec<usize> {
        vec![self.possible_values]
    }

    /// Converts `[batch_size, action_count]` logits to `[batch_size]` indices.
    fn tensor_from_neurons(&self, neurons: Tensor<2>) -> Result<Tensor<1, Int>, Self::Error> {
        let [batch_size, categories] = neurons.dims();
        if categories != self.possible_values || categories == 0 {
            return Err(SpaceError::InvalidPolicyShape {
                expected: vec![batch_size, self.possible_values],
                actual: neurons.dims().to_vec(),
            });
        }
        Ok(neurons.argmax(1).squeeze_dim(1))
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

/// Inclusive component bounds for rank-`R` continuous batches. Bounds have a
/// leading batch axis of length one: vector bounds are `[1, features]`, and image
/// bounds are `[1, channels, height, width]`.
#[derive(Clone)]
pub struct BoxSpace<const R: usize = 2> {
    low: Tensor<R>,
    high: Tensor<R>,
}

impl<const R: usize> BoxSpace<R> {
    /// Returns true when every value has the expected action/observation shape
    /// and lies within bounds. Inputs are `[batch, ...self.shape()]`.
    pub fn contains(&self, values: &Tensor<R>) -> bool {
        if values.dims()[1..] != self.low.dims()[1..] {
            return false;
        }
        let dtype = if values.dtype() == DType::F64 || self.low.dtype() == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        let device = values.device();
        let low = self.low.clone().to_device(&device).cast(dtype);
        let high = self.high.clone().to_device(&device).cast(dtype);
        let values = values.clone().cast(dtype);
        values
            .clone()
            .greater_equal(low)
            .bool_and(values.lower_equal(high))
            .all()
            .into_scalar::<bool>()
    }

    /// Returns the action/observation shape, excluding the leading batch axis.
    pub fn shape(&self) -> Vec<usize> {
        self.low.dims()[1..].to_vec()
    }
}

impl<const R: usize> ObservationSpace<R> for BoxSpace<R> {
    type Kind = Float;

    fn contains(&self, values: &Tensor<R>) -> bool {
        self.contains(values)
    }
    fn shape(&self) -> Vec<usize> {
        self.shape()
    }
}

impl<const R: usize> ActionSpace<R> for BoxSpace<R> {
    type Kind = Float;
    type Error = SpaceError;

    fn contains(&self, values: &Tensor<R>) -> bool {
        self.contains(values)
    }
    fn shape(&self) -> Vec<usize> {
        self.shape()
    }

    /// Samples `[batch_size, ...self.shape()]` in the bounds' dtype.
    /// Infinite endpoints use finite substitutes of half the dtype's maximum.
    fn sample_batch(&self, batch_size: usize, device: &Device) -> Result<Tensor<R>, Self::Error> {
        let dtype = self.low.dtype();
        let prepare = |bounds: &Tensor<R>| {
            let bounds = bounds.clone().to_device(device);
            let positive = bounds.clone().equal_elem(f64::INFINITY);
            let negative = bounds.clone().equal_elem(f64::NEG_INFINITY);
            let limit = finite_limit(dtype);
            bounds
                .mask_fill(positive, limit)
                .mask_fill(negative, -limit)
        };
        let low = prepare(&self.low);
        let high = prepare(&self.high);
        let mut shape = self.low.dims();
        shape[0] = batch_size;
        let random = Tensor::<R>::random(shape, Distribution::Uniform(0.0, 1.0), (device, dtype));
        // A weighted average of the bounds avoids overflowing high - low.
        let samples: Tensor<R> = low.clone() * (1.0 - random.clone()) + high.clone() * random;
        Ok(samples.max_pair(low).min_pair(high))
    }
}

impl<const R: usize> ActionMap<R, R> for BoxSpace<R> {
    fn policy_shape(&self) -> Vec<usize> {
        self.shape()
    }

    /// Clips `[batch_size, ...self.shape()]` in the policy output's dtype.
    fn tensor_from_neurons(&self, neurons: Tensor<R>) -> Result<Tensor<R>, Self::Error> {
        let actual = neurons.dims();
        if actual[1..] != self.low.dims()[1..] {
            let mut expected = vec![actual[0]];
            expected.extend_from_slice(&self.low.dims()[1..]);
            return Err(SpaceError::InvalidPolicyShape {
                expected,
                actual: actual.to_vec(),
            });
        }
        let device = neurons.device();
        let low = self.low.clone().to_device(&device).cast(neurons.dtype());
        let high = self.high.clone().to_device(&device).cast(neurons.dtype());
        Ok(neurons.max_pair(low).min_pair(high))
    }
}

impl<const R: usize> BoxSpace<R> {
    /// Creates bounds `low` and `high`, both rank `R` and shaped
    /// `[1, ...action_shape]`, with matching dtypes and devices.
    pub fn new(low: Tensor<R>, high: Tensor<R>) -> Self {
        assert!(R > 0, "box bounds require a batch axis");
        assert_eq!(
            low.dims()[0],
            1,
            "box bounds require a singleton batch axis"
        );
        assert_eq!(
            low.dims(),
            high.dims(),
            "box bounds must have matching shapes"
        );
        assert_eq!(
            low.dtype(),
            high.dtype(),
            "box bounds must have matching dtypes"
        );
        assert_eq!(
            low.device(),
            high.device(),
            "box bounds must share a device"
        );
        Self { low, high }
    }

    /// Creates `f32` bounds of the requested shape.
    pub fn new_with_universal_bounds(
        shape: [usize; R],
        low: f32,
        high: f32,
        device: &Device,
    ) -> Self {
        Self::new(
            Tensor::full(shape, low, (device, DType::F32)),
            Tensor::full(shape, high, (device, DType::F32)),
        )
    }

    pub fn new_unbounded(shape: [usize; R], device: &Device) -> Self {
        Self::new_with_universal_bounds(shape, f32::NEG_INFINITY, f32::INFINITY, device)
    }
}

fn finite_limit(dtype: DType) -> f64 {
    match dtype {
        DType::F64 => f64::MAX / 2.0,
        DType::F16 => 65_504.0 / 2.0,
        DType::BF16 => 3.389_531_389_251_535_5e38 / 2.0,
        _ => f64::from(f32::MAX / 2.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Flex shares its RNG across devices; seeded checks require exclusive access.
    static RNG_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn discrete_samples_indices_and_decodes_logits() {
        let _guard = RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let space = Discrete::new(3);
        let actions = space.sample_batch(128, &device).unwrap();
        assert_eq!(actions.dims(), [128]);
        assert_eq!(actions.dtype(), DType::I32);
        assert!(
            actions
                .into_data()
                .try_to_vec::<i32>()
                .unwrap()
                .iter()
                .all(|&value| (0..3).contains(&value))
        );
        let neurons = Tensor::from_floats([[1.0, 3.0, 2.0], [5.0, 0.0, 1.0]], &device);
        assert_eq!(
            space
                .tensor_from_neurons(neurons)
                .unwrap()
                .into_data()
                .try_to_vec::<i32>()
                .unwrap(),
            vec![1, 0]
        );
        assert!(space.contains(&Tensor::from_data([2i32], &device)));
        assert!(!space.contains(&Tensor::from_data([-1i32], &device)));
        assert!(!space.contains(&Tensor::from_data([3i32], &device)));
        assert!(space.contains(&Tensor::from_data([0i32, 1], &device)));
        assert!(!space.contains(&Tensor::from_data([0i32, 3], &device)));
        assert!(Discrete::new(0).sample_batch(1, &device).is_err());
        assert!(
            Discrete::new((1 << 24) + 1)
                .sample_batch(1, &device)
                .is_err()
        );
        assert!(
            space
                .tensor_from_neurons(Tensor::zeros([1, 2], &device))
                .is_err()
        );
        let single = Discrete::new(1);
        assert!(single.shape().is_empty());
        assert_eq!(single.policy_shape(), vec![1]);
        assert_eq!(
            single
                .tensor_from_neurons(Tensor::from_floats([[5.0], [-2.0]], &device))
                .unwrap()
                .into_data()
                .try_to_vec::<i32>()
                .unwrap(),
            vec![0, 0]
        );
    }

    #[test]
    fn box_sampling_preserves_bounds_ranks_and_dtypes() {
        let _guard = RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        for dtype in [DType::F32, DType::F64] {
            let low = Tensor::<3>::from_data([[[-2.0f64, 3.0], [7.0, -1.0]]], &device).cast(dtype);
            let high = Tensor::<3>::from_data([[[2.0f64, 3.0], [8.0, 0.0]]], &device).cast(dtype);
            let space = BoxSpace::<3>::new(low, high);
            let samples = space.sample_batch(32, &device).unwrap();
            assert_eq!(samples.dims(), [32, 2, 2]);
            assert_eq!(samples.dtype(), dtype);
            for row in 0..32 {
                assert!(space.contains(&samples.clone().slice([row..row + 1, 0..2, 0..2])));
            }
            assert!(space.contains(&samples));
            assert_eq!(space.shape(), vec![2, 2]);
            let unbounded = BoxSpace::<2>::new(
                Tensor::<2>::from_data([[f64::NEG_INFINITY]], &device).cast(dtype),
                Tensor::<2>::from_data([[f64::INFINITY]], &device).cast(dtype),
            );
            assert!(
                unbounded
                    .sample_batch(32, &device)
                    .unwrap()
                    .cast(DType::F64)
                    .into_data()
                    .try_to_vec::<f64>()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
    }

    #[test]
    fn box_clamps_in_policy_dtype_and_checks_shape() {
        let device = Device::flex();
        let space = BoxSpace::<2>::new_with_universal_bounds([1, 3], -1.0, 1.0, &device);
        for dtype in [DType::F32, DType::F64] {
            let neurons =
                Tensor::from_floats([[-2.0, -0.5, 0.25], [0.75, 1.5, 3.0]], &device).cast(dtype);
            let actions = space.tensor_from_neurons(neurons).unwrap();
            assert_eq!(actions.dims(), [2, 3]);
            assert_eq!(actions.dtype(), dtype);
            assert_eq!(
                actions
                    .cast(DType::F64)
                    .into_data()
                    .try_to_vec::<f64>()
                    .unwrap(),
                vec![-1.0, -0.5, 0.25, 0.75, 1.0, 1.0]
            );
        }
        assert!(
            space
                .tensor_from_neurons(Tensor::zeros([2, 1], &device))
                .is_err()
        );
    }

    #[test]
    fn box_membership_preserves_f64_precision_and_rejects_nan() {
        let device = Device::flex();
        let space = BoxSpace::<2>::new(
            Tensor::from_data([[0.0f64]], (&device, DType::F64)),
            Tensor::from_data([[1.0f64]], (&device, DType::F64)),
        );
        assert!(space.contains(&Tensor::from_data([[1.0f64]], (&device, DType::F64))));
        assert!(!space.contains(&Tensor::from_data(
            [[1.0f64 + 1e-12]],
            (&device, DType::F64)
        )));
        assert!(!space.contains(&Tensor::from_data([[f64::NAN]], (&device, DType::F64))));
        assert!(!space.contains(&Tensor::from_data([[0.0f64, 0.0]], &device)));
        assert!(!space.contains(&Tensor::from_data(
            [[0.0f64], [1.0 + 1e-12]],
            (&device, DType::F64)
        )));
    }

    #[test]
    fn box_sampling_and_clipping_work_through_a_trait_object() {
        let _guard = RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let space: Box<dyn ActionMap<2, 2, Error = SpaceError, Kind = Float>> = Box::new(
            BoxSpace::<2>::new_with_universal_bounds([1, 2], -1.0, 1.0, &device),
        );
        assert_eq!(space.sample_batch(4, &device).unwrap().dims(), [4, 2]);
        let actions = space
            .tensor_from_neurons(Tensor::from_floats([[-2.0, 2.0]], &device))
            .unwrap();
        assert_eq!(
            actions.into_data().try_to_vec::<f32>().unwrap(),
            vec![-1.0, 1.0]
        );
    }

    #[test]
    fn reseeding_reproduces_space_samples() {
        let _guard = RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let space = BoxSpace::<2>::new_unbounded([1, 3], &device);
        let discrete = Discrete::new(5);
        let draw = || {
            device.seed(123);
            let continuous = space.sample_batch(16, &device).unwrap().into_data();
            let categorical = discrete.sample_batch(16, &device).unwrap().into_data();
            (continuous, categorical)
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn empty_batches_preserve_rank() {
        let _guard = RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        assert_eq!(
            Discrete::new(3).sample_batch(0, &device).unwrap().dims(),
            [0]
        );
        let space = BoxSpace::<2>::new_with_universal_bounds([1, 3], -1.0, 1.0, &device);
        assert_eq!(space.sample_batch(0, &device).unwrap().dims(), [0, 3]);
    }

    #[test]
    fn observation_and_action_roles_use_independent_ranks() {
        let _guard = RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let observations: Box<dyn ObservationSpace<4, Kind = Float>> = Box::new(
            BoxSpace::<4>::new_with_universal_bounds([1, 2, 3, 4], 0.0, 255.0, &device),
        );
        let actions: Box<dyn ActionSpace<1, Kind = Int, Error = SpaceError>> =
            Box::new(Discrete::new(3));
        assert_eq!(observations.shape(), vec![2, 3, 4]);
        assert!(observations.contains(&Tensor::zeros([5, 2, 3, 4], &device)));
        assert!(!observations.contains(&Tensor::zeros([5, 2, 4, 3], &device)));
        assert!(!observations.contains(&Tensor::full([5, 2, 3, 4], 256.0, &device)));
        assert_eq!(actions.sample_batch(5, &device).unwrap().dims(), [5]);
        assert!(actions.contains(&Tensor::from_data([0i32, 1, 2], &device)));
    }
}
