use candle_core::{DType, Device, Tensor};

const MAX_EXACT_F32_INTEGER: u32 = 1 << 24;

/// Samples one integer uniformly from the inclusive range `start..=end`.
///
/// Sampling uses Candle's device RNG alongside tensor initialization and
/// action-space sampling. On backends that support [`Device::set_seed`], that
/// seed also controls this sample. Values above `2^24 - 1` are rejected because
/// the underlying `f32` uniform sampler cannot represent every larger integer
/// exactly.
pub fn sample_u32_inclusive(
    start: u32,
    end: u32,
    device: &Device,
) -> Result<u32, candle_core::Error> {
    if start > end {
        candle_core::bail!("invalid inclusive sampling range {start}..={end}")
    }
    if end >= MAX_EXACT_F32_INTEGER {
        candle_core::bail!("inclusive sampling range must end below {MAX_EXACT_F32_INTEGER}")
    }

    Tensor::rand(start as f32, (end + 1) as f32, (), device)?
        .floor()?
        .to_dtype(DType::U32)?
        .to_scalar::<u32>()
}

/// Shuffles `values` using random numbers produced by Candle on `device`.
///
/// Keeping the RNG on Candle means [`Device::set_seed`] controls minibatch
/// ordering on backends where Candle supports device seeding.
pub fn shuffle_with_device_rng<T>(
    values: &mut [T],
    device: &Device,
) -> Result<(), candle_core::Error> {
    if values.len() < 2 {
        return Ok(());
    }

    let random_values = Tensor::rand(0.0_f32, 1.0_f32, values.len(), device)?.to_vec1::<f32>()?;
    for index in (1..values.len()).rev() {
        let swap_index = (random_values[index] * (index as f32 + 1.0)).floor() as usize;
        values.swap(index, swap_index.min(index));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_is_inclusive() {
        let device = Device::Cpu;
        let sample = sample_u32_inclusive(3, 7, &device).unwrap();

        assert!((3..=7).contains(&sample));
        assert_eq!(sample_u32_inclusive(5, 5, &device).unwrap(), 5);
    }

    #[test]
    fn rejects_invalid_ranges() {
        let device = Device::Cpu;
        assert!(sample_u32_inclusive(2, 1, &device).is_err());
        assert!(sample_u32_inclusive(0, MAX_EXACT_F32_INTEGER, &device).is_err());
    }

    #[test]
    fn shuffle_preserves_a_permutation() {
        let device = Device::Cpu;
        let expected: Vec<u32> = (0..100).collect();
        let mut shuffled = expected.clone();
        shuffle_with_device_rng(&mut shuffled, &device).unwrap();
        shuffled.sort_unstable();
        assert_eq!(shuffled, expected);
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[test]
    fn shuffle_is_device_seeded() {
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0).unwrap();
        #[cfg(feature = "metal")]
        let device = Device::new_metal(0).unwrap();
        let mut first: Vec<u32> = (0..100).collect();
        let mut second = first.clone();

        device.set_seed(42).unwrap();
        shuffle_with_device_rng(&mut first, &device).unwrap();
        device.set_seed(42).unwrap();
        shuffle_with_device_rng(&mut second, &device).unwrap();

        assert_eq!(first, second);
    }
}
