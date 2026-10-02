use burn::tensor::{DType, Device, Distribution, Int, Tensor, TensorReadError};

const MAX_EXACT_F32_INTEGER: u32 = 1 << 24;

/// Invalid ranges or failures reading samples from Burn's device RNG.
#[derive(Debug, thiserror::Error)]
pub enum SamplingError {
    #[error("invalid inclusive sampling range {start}..={end}")]
    InvalidInclusiveRange { start: u32, end: u32 },
    #[error("inclusive sampling range end {end} exceeds the maximum {maximum}")]
    InclusiveRangeTooLarge { end: u32, maximum: u32 },
    #[error("reading tensor samples failed: {0}")]
    TensorRead(#[from] TensorReadError),
}

/// Samples one integer uniformly from the inclusive range `start..=end`.
///
/// Sampling uses an `Int` tensor and Burn's device RNG. [`Device::seed`]
/// controls reproducibility; some backends share their RNG across devices.
/// The maximum supported endpoint is `2^24 - 1`.
pub fn sample_u32_inclusive(start: u32, end: u32, device: &Device) -> Result<u32, SamplingError> {
    if start > end {
        return Err(SamplingError::InvalidInclusiveRange { start, end });
    }
    if end >= MAX_EXACT_F32_INTEGER {
        return Err(SamplingError::InclusiveRangeTooLarge {
            end,
            maximum: MAX_EXACT_F32_INTEGER - 1,
        });
    }

    Ok(Tensor::<1, Int>::random(
        [1],
        Distribution::Uniform(f64::from(start), f64::from(end) + 1.0),
        (device, DType::I32),
    )
    .try_into_scalar::<u32>()?)
}

/// Shuffles `values` using random numbers produced by Burn on `device`.
///
/// [`Device::seed`] controls minibatch ordering in single-threaded use.
/// Some backends share their RNG across devices. Random values are sampled
/// as `f32` regardless of the device's default float dtype.
pub fn shuffle_with_device_rng<T>(values: &mut [T], device: &Device) -> Result<(), SamplingError> {
    if values.len() < 2 {
        return Ok(());
    }

    let random_values = Tensor::<1>::random(
        [values.len()],
        Distribution::Uniform(0.0, 1.0),
        (device, DType::F32),
    )
    .try_into_data_as::<f32>()?
    .try_to_vec::<f32>()
    .map_err(TensorReadError::from)?;
    for index in (1..values.len()).rev() {
        let swap_index = (random_values[index] * (index as f32 + 1.0)).floor() as usize;
        values.swap(index, swap_index.min(index));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Flex shares its RNG across devices, so seeded checks must not interleave
    // with random draws from the other tests in this module.
    static RNG_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn sample_is_inclusive() {
        let _guard = RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        device.seed(42);
        let samples: Vec<_> = (0..256)
            .map(|_| sample_u32_inclusive(3, 7, &device).unwrap())
            .collect();
        assert!(samples.iter().all(|value| (3..=7).contains(value)));
        assert!(samples.contains(&3));
        assert!(samples.contains(&7));
        assert_eq!(sample_u32_inclusive(5, 5, &device).unwrap(), 5);
        assert_eq!(sample_u32_inclusive(0, 0, &device).unwrap(), 0);
        let maximum = MAX_EXACT_F32_INTEGER - 1;
        assert_eq!(
            sample_u32_inclusive(maximum, maximum, &device).unwrap(),
            maximum
        );
    }

    #[test]
    fn rejects_invalid_ranges() {
        let device = Device::flex();
        assert!(matches!(
            sample_u32_inclusive(2, 1, &device),
            Err(SamplingError::InvalidInclusiveRange { start: 2, end: 1 })
        ));
        assert!(matches!(
            sample_u32_inclusive(0, MAX_EXACT_F32_INTEGER, &device),
            Err(SamplingError::InclusiveRangeTooLarge {
                end: MAX_EXACT_F32_INTEGER,
                maximum: 16_777_215,
            })
        ));
    }

    #[test]
    fn shuffle_preserves_a_permutation() {
        let _guard = RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let expected: Vec<u32> = (0..100).collect();
        let mut shuffled = expected.clone();
        shuffle_with_device_rng(&mut shuffled, &device).unwrap();
        shuffled.sort_unstable();
        assert_eq!(shuffled, expected);
    }

    #[test]
    fn reseeding_reproduces_samples_and_shuffle() {
        let _guard = RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let draw = || {
            device.seed(123);
            let samples: Vec<_> = (0..32)
                .map(|_| sample_u32_inclusive(0, 100, &device).unwrap())
                .collect();
            let mut values: Vec<_> = (0..100).collect();
            shuffle_with_device_rng(&mut values, &device).unwrap();
            (samples, values)
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn trivial_shuffles_leave_values_unchanged() {
        let device = Device::flex();
        let mut empty: [u32; 0] = [];
        shuffle_with_device_rng(&mut empty, &device).unwrap();
        let mut singleton = [7];
        shuffle_with_device_rng(&mut singleton, &device).unwrap();
        assert_eq!(singleton, [7]);
    }
}
