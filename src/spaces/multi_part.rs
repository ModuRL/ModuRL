//! Typed observation fields transported through the existing flat tensor interface.
//!
//! A named application struct needs no string keys or runtime schema. Convert it
//! to and from the tuple accepted by [`MultiPartSpace::pack`] and returned by
//! [`MultiPartSpace::unpack`]:
//!
//! ```
//! use burn::tensor::{Int, Tensor};
//! struct RobotObs { image: Tensor<4>, joints: Tensor<2>, mode: Tensor<2, Int> }
//! impl From<RobotObs> for (Tensor<4>, Tensor<2>, Tensor<2, Int>) {
//!     fn from(obs: RobotObs) -> Self { (obs.image, obs.joints, obs.mode) }
//! }
//! impl From<(Tensor<4>, Tensor<2>, Tensor<2, Int>)> for RobotObs {
//!     fn from((image, joints, mode): (Tensor<4>, Tensor<2>, Tensor<2, Int>)) -> Self {
//!         Self { image, joints, mode }
//!     }
//! }
//! ```

use super::{BoxSpace, Discrete, ObservationSpace, SpaceError};
use crate::tensor_rank::TensorBatchTypes;
use burn::tensor::{DType, Float, Int, Tensor};

/// One typed field of an observation. Flattening preserves floating-point dtype.
pub trait ObservationPart {
    /// Native batched tensor, including its rank and kind.
    type Batch;
    /// Number of scalars per environment.
    fn flat_len(&self) -> usize;
    /// Shape without the leading batch axis.
    fn shape(&self) -> Vec<usize>;
    /// Shape of a native batch, including its batch axis.
    fn batch_shape(&self, batch: &Self::Batch) -> Vec<usize>;
    /// Reshapes a correctly shaped batch to `[batch, flat_len]`.
    fn flatten(&self, batch: Self::Batch) -> Tensor<2>;
    /// Restores `[batch, flat_len]` to its native rank and kind.
    fn unflatten(&self, batch: Tensor<2>) -> Self::Batch;
    /// Checks the native shape and bounds.
    fn contains(&self, batch: &Self::Batch) -> bool;
}

impl<const R: usize> ObservationPart for BoxSpace<R> {
    type Batch = Tensor<R>;
    fn flat_len(&self) -> usize {
        self.shape().iter().product()
    }
    fn shape(&self) -> Vec<usize> {
        self.shape()
    }
    fn batch_shape(&self, batch: &Self::Batch) -> Vec<usize> {
        batch.dims().to_vec()
    }
    fn flatten(&self, batch: Self::Batch) -> Tensor<2> {
        let count = batch.dims()[0];
        batch.reshape([count, self.flat_len()])
    }
    /// Restores `[batch, flat_len]` to `[batch, ...shape]`.
    fn unflatten(&self, batch: Tensor<2>) -> Self::Batch {
        let mut shape = self.low().dims();
        shape[0] = batch.dims()[0];
        batch.reshape(shape)
    }
    fn contains(&self, batch: &Self::Batch) -> bool {
        self.contains(batch)
    }
}

impl ObservationPart for Discrete {
    type Batch = Tensor<2, Int>;
    fn flat_len(&self) -> usize {
        1
    }
    fn shape(&self) -> Vec<usize> {
        self.shape()
    }
    fn batch_shape(&self, batch: &Self::Batch) -> Vec<usize> {
        batch.dims().to_vec()
    }
    fn flatten(&self, batch: Self::Batch) -> Tensor<2> {
        batch.float()
    }
    /// Converts `[batch, 1]` floating indices to native integers.
    fn unflatten(&self, batch: Tensor<2>) -> Self::Batch {
        batch.int()
    }
    fn contains(&self, batch: &Self::Batch) -> bool {
        self.contains(batch)
    }
}

/// A typed layout of observation spaces, implemented for tuples of 2–8 fields.
pub trait ObservationParts: Clone {
    /// Tuple of native batched tensors.
    type Parts;
    /// Total number of scalars per environment.
    fn flat_len(&self) -> usize;
    /// Validates native shapes and combines the fields along axis one.
    fn pack(&self, parts: Self::Parts) -> Result<Tensor<2>, SpaceError>;
    /// Validates `[batch, flat_len]` and restores each field.
    fn unpack(&self, batch: Tensor<2>) -> Result<Self::Parts, SpaceError>;
    /// Checks `[batch, flat_len]`, category integrality, and every field's bounds.
    fn contains(&self, batch: &Tensor<2>) -> bool;
}

/// Flat transport for a tuple of typed observation spaces.
///
/// Existing models may continue to consume the packed tensor directly. Use a
/// multipart encoder when fields need distinct feature extractors.
#[derive(Clone)]
pub struct MultiPartSpace<P> {
    parts: P,
}

impl<P> std::fmt::Debug for MultiPartSpace<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiPartSpace").finish_non_exhaustive()
    }
}

impl<P: ObservationParts> MultiPartSpace<P> {
    /// Creates a layout from a tuple of spaces.
    pub fn new(parts: P) -> Self {
        Self { parts }
    }
    /// Returns the typed sub-spaces in transport order.
    pub fn parts(&self) -> &P {
        &self.parts
    }
    /// Total packed feature count.
    pub fn flat_len(&self) -> usize {
        self.parts.flat_len()
    }
    /// Packs native batches, promoting all fields to F64 if any field is F64.
    /// Bounds are not checked; use [`ObservationSpace::contains`] separately.
    /// Empty batches are rejected.
    pub fn pack(&self, parts: P::Parts) -> Result<Tensor<2>, SpaceError> {
        self.parts.pack(parts)
    }
    /// Restores `[batch, flat_len]` to native ranks and kinds.
    /// Floating fields retain the packed dtype. Empty batches are rejected.
    pub fn unpack(&self, batch: Tensor<2>) -> Result<P::Parts, SpaceError> {
        self.parts.unpack(batch)
    }
}

impl<P: ObservationParts> TensorBatchTypes for MultiPartSpace<P> {
    type Kind = Float;
    type Item = Tensor<1>;
    type Batch = Tensor<2>;
}
impl<P: ObservationParts> ObservationSpace<2> for MultiPartSpace<P> {
    fn shape(&self) -> Vec<usize> {
        vec![self.flat_len()]
    }
    /// Checks `[batch, flat_len]` against all field shapes and bounds.
    fn contains(&self, batch: &Tensor<2>) -> bool {
        self.parts.contains(batch)
    }
}

macro_rules! impl_parts {
    ($( $P:ident : $i:tt ),+) => {
        impl<$($P: ObservationPart + Clone),+> ObservationParts for ($($P,)+) {
            type Parts = ($($P::Batch,)+);
            fn flat_len(&self) -> usize { 0 $(+ self.$i.flat_len())+ }
            fn pack(&self, parts: Self::Parts) -> Result<Tensor<2>, SpaceError> {
                $(let actual = self.$i.batch_shape(&parts.$i);
                if actual[0] == 0 { return Err(SpaceError::EmptyObservationBatch); }
                if actual[1..] != self.$i.shape() {
                    return Err(SpaceError::InvalidPartShape { expected: self.$i.shape(), actual });
                })+
                let fields = vec![$(self.$i.flatten(parts.$i)),+];
                let count = fields[0].dims()[0];
                let device = fields[0].device();
                if fields.iter().any(|f| f.dims()[0] != count || f.device() != device) {
                    return Err(SpaceError::IncompatibleParts);
                }
                let dtype = if fields.iter().any(|f| f.dtype() == DType::F64) { DType::F64 } else { DType::F32 };
                Ok(Tensor::cat(fields.into_iter().map(|f| f.cast(dtype)).collect(), 1))
            }
            /// Splits `[batch, flat_len]` into native field batches.
            fn unpack(&self, batch: Tensor<2>) -> Result<Self::Parts, SpaceError> {
                let [count, width] = batch.dims();
                if count == 0 { return Err(SpaceError::EmptyObservationBatch); }
                if width != self.flat_len() {
                    return Err(SpaceError::InvalidPartShape { expected: vec![count, self.flat_len()], actual: batch.dims().to_vec() });
                }
                let mut offset = 0;
                let result = ($({
                    let end = offset + self.$i.flat_len();
                    let field = self.$i.unflatten(batch.clone().slice([0..count, offset..end]));
                    offset = end;
                    field
                },)+);
                let _ = offset;
                Ok(result)
            }
            /// Validates `[batch, flat_len]` and every field's bounds.
            fn contains(&self, batch: &Tensor<2>) -> bool {
                let Ok(parts) = self.unpack(batch.clone()) else { return false; };
                if !($(self.$i.contains(&parts.$i))&&+) { return false; }
                // A cast to integer must not silently accept fractional categories or NaNs.
                let Ok(restored) = self.pack(parts) else { return false; };
                restored.equal(batch.clone()).all().into_scalar::<bool>()
            }
        }
    };
}
impl_parts!(A:0, B:1);
impl_parts!(A:0, B:1, C:2);
impl_parts!(A:0, B:1, C:2, D:3);
impl_parts!(A:0, B:1, C:2, D:3, E:4);
impl_parts!(A:0, B:1, C:2, D:3, E:4, F:5);
impl_parts!(A:0, B:1, C:2, D:3, E:4, F:5, G:6);
impl_parts!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn native_round_trip_and_validation() {
        let device = Device::flex();
        for dtype in [DType::F32, DType::F64] {
            let space = MultiPartSpace::new((
                BoxSpace::new_unbounded([1, 2, 3, 4], &device),
                BoxSpace::new_with_universal_bounds([1, 3], -1.0, 1.0, &device),
                Discrete::new(1 << 24).unwrap(),
            ));
            let image = Tensor::ones([2, 2, 3, 4], (&device, dtype));
            let joints = Tensor::zeros([2, 3], (&device, dtype));
            let mode = Tensor::from_data([[0i32], [(1 << 24) - 1]], &device);
            let flat = space
                .pack((image.clone(), joints.clone(), mode.clone()))
                .unwrap();
            assert_eq!(flat.dims(), [2, 28]);
            assert_eq!(flat.dtype(), dtype);
            assert!(ObservationSpace::contains(&space, &flat));
            let restored = space.unpack(flat).unwrap();
            assert_eq!(restored.0.into_data(), image.into_data());
            assert_eq!(restored.1.into_data(), joints.into_data());
            assert_eq!(
                restored.2.into_data().try_to_vec::<i32>().unwrap(),
                mode.into_data().try_to_vec::<i32>().unwrap()
            );
            assert!(!ObservationSpace::contains(
                &space,
                &Tensor::zeros([2, 27], (&device, dtype))
            ));
            assert!(
                space
                    .unpack(Tensor::zeros([2, 27], (&device, dtype)))
                    .is_err()
            );
            let invalid = space
                .pack((
                    Tensor::zeros([2, 2, 3, 4], (&device, dtype)),
                    Tensor::full([2, 3], 2.0, (&device, dtype)),
                    Tensor::zeros([2, 1], &device),
                ))
                .unwrap();
            assert!(!ObservationSpace::contains(&space, &invalid));
            let fractional = Tensor::zeros([2, 28], (&device, dtype))
                .slice_assign([0..2, 27..28], Tensor::full([2, 1], 0.5, (&device, dtype)));
            assert!(!ObservationSpace::contains(&space, &fractional));
            assert!(matches!(
                space.pack((
                    Tensor::zeros([2, 2, 3, 4], &device),
                    Tensor::zeros([1, 3], &device),
                    Tensor::zeros([2, 1], &device)
                )),
                Err(SpaceError::IncompatibleParts)
            ));
            assert!(matches!(
                space.pack((
                    Tensor::zeros([2, 2, 4, 3], &device),
                    Tensor::zeros([2, 3], &device),
                    Tensor::zeros([2, 1], &device)
                )),
                Err(SpaceError::InvalidPartShape { .. })
            ));
        }
    }

    #[test]
    fn empty_batches_and_mixed_precision() {
        let device = Device::flex();
        let space = MultiPartSpace::new((
            BoxSpace::new_unbounded([1, 2], &device),
            Discrete::new(3).unwrap(),
        ));
        assert!(matches!(
            space.pack((
                Tensor::zeros([0, 2], &device),
                Tensor::zeros([0, 1], &device)
            )),
            Err(SpaceError::EmptyObservationBatch)
        ));
        assert!(matches!(
            space.unpack(Tensor::zeros([0, 3], &device)),
            Err(SpaceError::EmptyObservationBatch)
        ));
        assert!(!ObservationSpace::contains(
            &space,
            &Tensor::zeros([0, 3], &device)
        ));
        let flat = space
            .pack((
                Tensor::full([1, 2], 1.000000000000001f64, (&device, DType::F64)),
                Tensor::from_data([[2i32]], &device),
            ))
            .unwrap();
        assert_eq!(flat.dtype(), DType::F64);
        assert_eq!(
            space
                .unpack(flat)
                .unwrap()
                .0
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            vec![1.000000000000001f64; 2]
        );
    }
}
