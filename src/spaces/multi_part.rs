//! Typed observation fields transported through the existing flat tensor interface.
//!
//! ModuRL's typed answer to Gymnasium/SB3 `Dict` observations.
//!
//! Float fields must share one dtype, including F16/BF16; mismatches are errors.
//! Discrete fields use that shared float dtype, or F32 for all-discrete layouts.
//! The category-count limit depends on the transport dtype: 2048 for F16,
//! 256 for BF16, and 2^24 for F32/F64. Plain [`Discrete`] fields require and restore
//! I32; [`DiscretePart`] configures another native integer dtype. Pack/unpack
//! reject category counts exceeding either dtype's capacity. Packing validates
//! shapes and dtypes, not bounds; use [`ObservationSpace::contains`] for bounds.
//! Unpacking rejects non-integral discrete columns instead of truncating them.
//! Configure observation normalization with
//! `with_passthrough_columns(space.discrete_columns())`, or normalize continuous
//! fields before packing. Frame-stacked inputs must fold their frame axis into
//! the batch axis before a multipart encoder, then restore or concatenate frames.
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
use burn::tensor::{DType, Float, FloatDType, Int, IntDType, Tensor};

/// One typed field of an observation. Flattening preserves floating-point dtype.
/// Discrete fields flatten to F32 unless [`Self::flatten_as`] selects a shared dtype.
pub trait ObservationPart {
    /// Native batched tensor, including its rank and kind.
    type Batch;
    /// Dtype of a native batch, before transport conversion.
    fn batch_dtype(&self, batch: &Self::Batch) -> DType;
    /// Configured integer dtype for discrete fields; `None` for float fields.
    fn int_dtype(&self) -> Option<IntDType> {
        None
    }
    /// Number of categories for discrete fields; `None` for continuous fields.
    fn categories(&self) -> Option<usize> {
        None
    }
    /// Number of scalars per environment.
    fn flat_len(&self) -> usize;
    /// Shape without the leading batch axis.
    fn shape(&self) -> Vec<usize>;
    /// Shape of a native batch, including its batch axis.
    fn batch_shape(&self, batch: &Self::Batch) -> Vec<usize>;
    /// Reshapes a correctly shaped batch to `[batch, flat_len]`.
    fn flatten(&self, batch: Self::Batch) -> Tensor<2>;
    /// Packs `[batch, ...shape]` in the validated shared dtype.
    /// Float fields retain their dtype; integer fields convert directly to `dtype`.
    fn flatten_as(&self, batch: Self::Batch, _dtype: DType) -> Tensor<2> {
        self.flatten(batch)
    }
    /// Restores `[batch, flat_len]` to its native rank and kind.
    fn unflatten(&self, batch: Tensor<2>) -> Self::Batch;
    /// Checks the native shape and bounds.
    fn contains(&self, batch: &Self::Batch) -> bool;
}

impl<const R: usize> ObservationPart for BoxSpace<R> {
    type Batch = Tensor<R>;
    fn batch_dtype(&self, batch: &Self::Batch) -> DType {
        batch.dtype()
    }
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

/// A discrete observation field with an explicit native integer dtype.
/// Plain [`Discrete`] fields use I32, matching `Discrete::sample_batch`.
/// Packing and bounds checks reject other input dtypes; unpacking restores this dtype.
#[derive(Clone, Debug)]
pub struct DiscretePart {
    space: Discrete,
    dtype: IntDType,
}

impl DiscretePart {
    /// Configures the native integer dtype. Precision is validated during pack/unpack.
    pub fn new(space: Discrete, dtype: IntDType) -> Self {
        Self { space, dtype }
    }
    /// Returns the underlying category space.
    pub fn space(&self) -> &Discrete {
        &self.space
    }
    /// Returns the integer dtype restored by unpacking.
    pub fn dtype(&self) -> IntDType {
        self.dtype
    }
}

macro_rules! impl_discrete_part {
    ($ty:ty, $space:expr, $dtype:expr) => {
        impl ObservationPart for $ty {
            type Batch = Tensor<2, Int>;
            fn batch_dtype(&self, batch: &Self::Batch) -> DType {
                batch.dtype()
            }
            fn int_dtype(&self) -> Option<IntDType> {
                Some(($dtype)(self))
            }
            fn categories(&self) -> Option<usize> {
                Some(($space)(self).get_possible_values())
            }
            fn flat_len(&self) -> usize {
                1
            }
            fn shape(&self) -> Vec<usize> {
                vec![1]
            }
            fn batch_shape(&self, batch: &Self::Batch) -> Vec<usize> {
                batch.dims().to_vec()
            }
            fn flatten(&self, batch: Self::Batch) -> Tensor<2> {
                batch.cast(FloatDType::F32)
            }
            fn flatten_as(&self, batch: Self::Batch, dtype: DType) -> Tensor<2> {
                batch.cast(FloatDType::from(dtype))
            }
            /// Converts `[batch, 1]` floating indices to the configured integer dtype.
            fn unflatten(&self, batch: Tensor<2>) -> Self::Batch {
                batch.cast(($dtype)(self))
            }
            fn contains(&self, batch: &Self::Batch) -> bool {
                if batch.dtype() != DType::from(($dtype)(self)) {
                    return false;
                }
                let values = match batch.dtype() {
                    DType::I8 | DType::U8 | DType::I16 | DType::U16 => {
                        batch.clone().cast(IntDType::I32)
                    }
                    _ => batch.clone(),
                };
                ($space)(self).contains(&values)
            }
        }
    };
}
fn discrete_space(space: &Discrete) -> &Discrete {
    space
}
fn discrete_dtype(_: &Discrete) -> IntDType {
    IntDType::I32
}
impl_discrete_part!(Discrete, discrete_space, discrete_dtype);
impl_discrete_part!(DiscretePart, DiscretePart::space, DiscretePart::dtype);

fn validate_precision<P: ObservationPart>(
    space: &P,
    dtype: DType,
    part: usize,
) -> Result<(), SpaceError> {
    let Some(categories) = space.categories() else {
        return Ok(());
    };
    let max_categories = match dtype {
        DType::F16 => 2048,
        DType::BF16 => 256,
        DType::I8 => 128,
        DType::U8 => 256,
        DType::I16 => 32768,
        DType::U16 => 65536,
        _ => 1 << 24,
    };
    if categories > max_categories {
        return Err(SpaceError::DiscretePartPrecision {
            part,
            categories,
            dtype,
            max_categories,
        });
    }
    Ok(())
}

/// Checks that discrete `[batch, width]` values survive their native integer conversion.
fn discrete_round_trip_matches<P: ObservationPart>(space: &P, batch: &Tensor<2>) -> bool {
    space.categories().is_none()
        || space
            .flatten_as(space.unflatten(batch.clone()), batch.dtype())
            .equal(batch.clone())
            .bool_and(batch.clone().is_finite())
            .all()
            .into_scalar::<bool>()
}

/// A typed layout of observation spaces, implemented for tuples of 2–8 fields.
pub trait ObservationParts: Clone {
    /// Tuple of native batched tensors.
    type Parts;
    /// Total number of scalars per environment.
    fn flat_len(&self) -> usize;
    /// Packed feature indices belonging to discrete fields, in transport order.
    fn discrete_columns(&self) -> Vec<usize>;
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
    /// Returns the packed feature columns that observation normalization must skip.
    /// Pass these to `NormalizeObservationGym::with_passthrough_columns`.
    pub fn discrete_columns(&self) -> Vec<usize> {
        self.parts.discrete_columns()
    }
    /// Packs native batches without casting Float fields, which must share one dtype.
    /// Discrete fields use that dtype, or F32 for an all-discrete layout.
    /// Discrete category limits depend on the float dtype (F16: 2048, BF16: 256,
    /// F32/F64: 2^24) and the configured integer dtype.
    /// Bounds are not checked; use [`ObservationSpace::contains`] separately.
    /// Empty batches are rejected.
    pub fn pack(&self, parts: P::Parts) -> Result<Tensor<2>, SpaceError> {
        self.parts.pack(parts)
    }
    /// Restores `[batch, flat_len]` to native ranks and kinds.
    /// Floating fields retain the packed dtype; discrete fields restore I32 or
    /// the dtype configured by [`DiscretePart`]. Empty batches are rejected.
    /// Discrete columns must contain finite integral values that survive an integer round-trip.
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
            fn discrete_columns(&self) -> Vec<usize> {
                let mut columns = Vec::new();
                let mut offset = 0;
                $(let end = offset + self.$i.flat_len();
                if self.$i.categories().is_some() { columns.extend(offset..end); }
                offset = end;)+
                let _ = offset;
                columns
            }
            fn pack(&self, parts: Self::Parts) -> Result<Tensor<2>, SpaceError> {
                let mut dtype = None;
                $(let actual = self.$i.batch_shape(&parts.$i);
                if actual[0] == 0 { return Err(SpaceError::EmptyObservationBatch); }
                if actual[1..] != self.$i.shape() {
                    let mut expected = vec![actual[0]];
                    expected.extend(self.$i.shape());
                    return Err(SpaceError::InvalidPartShape { expected, actual });
                }
                let actual_dtype = self.$i.batch_dtype(&parts.$i);
                if let Some(int_dtype) = self.$i.int_dtype() {
                    let expected = DType::from(int_dtype);
                    if actual_dtype != expected { return Err(SpaceError::PartDTypeMismatch { part: $i, expected, actual: actual_dtype }); }
                    validate_precision(&self.$i, expected, $i)?;
                } else {
                    if let Some(expected) = dtype {
                        if actual_dtype != expected { return Err(SpaceError::PartDTypeMismatch { part: $i, expected, actual: actual_dtype }); }
                    }
                    dtype = Some(actual_dtype);
                })+
                let dtype = dtype.unwrap_or(DType::F32);
                $(validate_precision(&self.$i, dtype, $i)?;)+
                let fields = vec![$(self.$i.flatten_as(parts.$i, dtype)),+];
                let count = fields[0].dims()[0];
                let device = fields[0].device();
                if fields.iter().any(|f| f.dims()[0] != count || f.device() != device) {
                    return Err(SpaceError::IncompatibleParts);
                }
                Ok(Tensor::cat(fields, 1))
            }
            /// Splits `[batch, flat_len]` into native field batches.
            fn unpack(&self, batch: Tensor<2>) -> Result<Self::Parts, SpaceError> {
                let [count, width] = batch.dims();
                if count == 0 { return Err(SpaceError::EmptyObservationBatch); }
                if width != self.flat_len() {
                    return Err(SpaceError::InvalidPartShape { expected: vec![count, self.flat_len()], actual: batch.dims().to_vec() });
                }
                $(validate_precision(&self.$i, batch.dtype(), $i)?;
                if let Some(dtype) = self.$i.int_dtype() { validate_precision(&self.$i, dtype.into(), $i)?; })+
                let mut offset = 0;
                let result = ($({
                    let end = offset + self.$i.flat_len();
                    let flat = batch.clone().slice([0..count, offset..end]);
                    if !discrete_round_trip_matches(&self.$i, &flat) {
                        return Err(SpaceError::InvalidDiscretePartValue { part: $i });
                    }
                    let field = self.$i.unflatten(flat);
                    offset = end;
                    field
                },)+);
                let _ = offset;
                Ok(result)
            }
            /// Validates `[batch, flat_len]` and every field's bounds.
            fn contains(&self, batch: &Tensor<2>) -> bool {
                let Ok(parts) = self.unpack(batch.clone()) else { return false; };
                $(self.$i.contains(&parts.$i))&&+
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
    fn empty_batches_and_native_precision() {
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

    #[test]
    fn mixed_float_dtypes_and_full_shape_errors() {
        let device = Device::flex();
        let space = MultiPartSpace::new((
            Discrete::new(2).unwrap(),
            BoxSpace::new_unbounded([1, 2], &device),
            BoxSpace::new_unbounded([1, 3], &device),
        ));
        for (expected, actual) in [
            (DType::F32, DType::F64),
            (DType::F64, DType::F32),
            (DType::F16, DType::BF16),
        ] {
            assert!(
                matches!(space.pack((Tensor::zeros([2, 1], &device), Tensor::zeros([2, 2], (&device, expected)), Tensor::zeros([2, 3], (&device, actual)))), Err(SpaceError::PartDTypeMismatch { part: 2, expected: e, actual: a }) if e == expected && a == actual)
            );
        }
        let error = space
            .pack((
                Tensor::zeros([2, 1], &device),
                Tensor::zeros([2, 4], &device),
                Tensor::zeros([2, 3], &device),
            ))
            .unwrap_err();
        assert!(
            matches!(error, SpaceError::InvalidPartShape { expected, actual } if expected == [2, 2] && actual == [2, 4])
        );
        assert!(
            matches!(space.unpack(Tensor::zeros([2, 7], &device)), Err(SpaceError::InvalidPartShape { expected, actual }) if expected == [2, 6] && actual == [2, 7])
        );
    }

    #[test]
    fn float_dtype_preservation_and_discrete_precision_limits() {
        let device = Device::flex();
        for (dtype, limit) in [
            (DType::F16, 2048),
            (DType::BF16, 256),
            (DType::F32, 1 << 24),
            (DType::F64, 1 << 24),
        ] {
            let make_space = |categories| {
                MultiPartSpace::new((
                    BoxSpace::new_unbounded([1, 2], &device),
                    BoxSpace::new_unbounded([1, 1], &device),
                    Discrete::new(categories).unwrap(),
                ))
            };
            let space = make_space(limit);
            let flat = space
                .pack((
                    Tensor::ones([2, 2], (&device, dtype)),
                    Tensor::ones([2, 1], (&device, dtype)),
                    Tensor::full([2, 1], (limit - 1) as i32, (&device, DType::I32)),
                ))
                .unwrap();
            assert_eq!(flat.dtype(), dtype);
            assert!(ObservationSpace::contains(&space, &flat));
            let fields = space.unpack(flat).unwrap();
            assert_eq!(fields.0.dtype(), dtype);
            assert_eq!(fields.1.dtype(), dtype);
            assert_eq!(fields.2.dtype(), DType::I32);
            assert_eq!(
                fields.2.into_data().try_to_vec::<i32>().unwrap(),
                vec![(limit - 1) as i32; 2]
            );
            if limit < 1 << 24 {
                let bad = make_space(limit + 1);
                assert!(
                    matches!(bad.pack((fields.0, fields.1, Tensor::zeros([2, 1], &device))), Err(SpaceError::DiscretePartPrecision { part: 2, categories, dtype: d, max_categories }) if categories == limit + 1 && d == dtype && max_categories == limit)
                );
                let flat = Tensor::zeros([2, 4], (&device, dtype));
                assert!(matches!(
                    bad.unpack(flat.clone()),
                    Err(SpaceError::DiscretePartPrecision { .. })
                ));
                assert!(!ObservationSpace::contains(&bad, &flat));
            }
        }
    }

    #[test]
    fn configured_integer_dtypes_round_trip_and_validate() {
        let device = Device::flex();
        for dtype in [IntDType::I32, IntDType::I64, IntDType::U32] {
            let field = DiscretePart::new(Discrete::new(1 << 24).unwrap(), dtype);
            let space = MultiPartSpace::new((Discrete::new(2).unwrap(), field.clone()));
            let native =
                Tensor::<2, Int>::full([2, 1], (1 << 24) - 1, (&device, DType::from(dtype)));
            let flat = space
                .pack((Tensor::zeros([2, 1], (&device, DType::I32)), native.clone()))
                .unwrap();
            assert_eq!(flat.dtype(), DType::F32);
            assert!(ObservationSpace::contains(&space, &flat));
            let restored = space.unpack(flat).unwrap();
            assert_eq!(restored.0.dtype(), DType::I32);
            assert_eq!(restored.1.dtype(), DType::from(dtype));
            assert_eq!(restored.1.into_data(), native.into_data());
            let wrong_dtype = if dtype == IntDType::I32 {
                DType::I64
            } else {
                DType::I32
            };
            let wrong = Tensor::zeros([2, 1], (&device, wrong_dtype));
            assert!(!ObservationPart::contains(&field, &wrong));
            assert!(
                matches!(space.pack((Tensor::zeros([2, 1], &device), wrong)), Err(SpaceError::PartDTypeMismatch { part: 1, expected, actual }) if expected == DType::from(dtype) && actual == wrong_dtype)
            );
        }
        let plain = Discrete::new(3).unwrap();
        assert!(!ObservationPart::contains(
            &plain,
            &Tensor::zeros([1, 1], (&device, DType::I64))
        ));
        let narrow = MultiPartSpace::new((
            DiscretePart::new(Discrete::new(129).unwrap(), IntDType::I8),
            Discrete::new(2).unwrap(),
        ));
        assert!(matches!(
            narrow.pack((
                Tensor::zeros([1, 1], (&device, DType::I8)),
                Tensor::zeros([1, 1], &device)
            )),
            Err(SpaceError::DiscretePartPrecision {
                part: 0,
                dtype: DType::I8,
                ..
            })
        ));
    }

    #[test]
    fn unpack_rejects_fractional_and_nonfinite_discrete_values() {
        let device = Device::flex();
        let space = MultiPartSpace::new((
            BoxSpace::new_unbounded([1, 1], &device),
            Discrete::new(4).unwrap(),
        ));
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            for value in [0.009, 1.5, f64::NAN, f64::INFINITY] {
                let packed = Tensor::from_data([[0.5f64, value]], (&device, dtype));
                assert!(matches!(
                    space.unpack(packed.clone()),
                    Err(SpaceError::InvalidDiscretePartValue { part: 1 })
                ));
                assert!(!ObservationSpace::contains(&space, &packed));
            }
            let restored = space
                .unpack(Tensor::from_data([[0.5f64, 2.0]], (&device, dtype)))
                .unwrap();
            assert_eq!(restored.1.into_scalar::<i32>(), 2);
            let out_of_bounds = Tensor::from_data([[0.5f64, 4.0]], (&device, dtype));
            assert!(space.unpack(out_of_bounds.clone()).is_ok());
            assert!(!ObservationSpace::contains(&space, &out_of_bounds));
        }
    }

    #[test]
    fn discrete_columns_follow_flat_transport_offsets() {
        let device = Device::flex();
        let space = MultiPartSpace::new((
            BoxSpace::new_unbounded([1, 2], &device),
            Discrete::new(3).unwrap(),
            BoxSpace::new_unbounded([1, 1, 2, 2], &device),
            DiscretePart::new(Discrete::new(2).unwrap(), IntDType::U32),
        ));
        assert_eq!(space.discrete_columns(), vec![2, 7]);
        assert!(
            MultiPartSpace::new((
                BoxSpace::new_unbounded([1, 2], &device),
                BoxSpace::new_unbounded([1, 3], &device)
            ))
            .discrete_columns()
            .is_empty()
        );
        assert_eq!(
            MultiPartSpace::new((Discrete::new(2).unwrap(), Discrete::new(3).unwrap()))
                .discrete_columns(),
            vec![0, 1]
        );
    }

    #[test]
    fn narrow_integer_bounds_include_full_capacity_endpoints() {
        let device = Device::flex();
        for (dtype, categories) in [
            (IntDType::I8, 128),
            (IntDType::U8, 256),
            (IntDType::I16, 32768),
            (IntDType::U16, 65536),
        ] {
            let part = DiscretePart::new(Discrete::new(categories).unwrap(), dtype);
            let indices = Tensor::<2, Int>::from_data([[0i32], [(categories - 1) as i32]], &device)
                .cast(dtype);
            assert!(ObservationPart::contains(&part, &indices));
            assert!(!ObservationPart::contains(
                &part,
                &indices.clone().cast(IntDType::I32)
            ));
            let space =
                MultiPartSpace::new((BoxSpace::new_unbounded([1, 1], &device), part.clone()));
            let packed = space
                .pack((Tensor::zeros([2, 1], &device), indices.clone()))
                .unwrap();
            assert!(ObservationSpace::contains(&space, &packed));
            assert_eq!(
                space.unpack(packed).unwrap().1.into_data(),
                indices.into_data()
            );
            if matches!(dtype, IntDType::I8 | IntDType::I16) {
                let negative = Tensor::<2, Int>::from_data([[-1i32]], &device).cast(dtype);
                assert!(!ObservationPart::contains(&part, &negative));
            }
        }
    }
}
