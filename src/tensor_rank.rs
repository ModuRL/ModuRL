//! Native tensor types with one more or one fewer axis.
//!
//! These mappings preserve the tensor kind. They do not create or convert tensor values.
//! `NextRank` supports ranks 1 through 1024; `PrevRank` supports ranks 2 through 1025.
//! Rank 1 has no predecessor because Burn represents scalars with shape `[1]`.
//! Rank 1025 has a predecessor but no successor mapping here.
//!
//! In a generic implementation, constrain `Next` or `Prev` to `Tensor<B, K>` to use native Burn methods.
//! The implementation can infer `B` without adding it to the component's public type parameters.

use burn::tensor::{Tensor, kind::Basic};

/// Supplies the native tensor type with one additional axis and the same kind.
pub trait NextRank: Clone + std::fmt::Debug {
    /// The rank-D + 1 tensor type. Axis sizes, dtype, and device belong to tensor values.
    type Next: Clone + std::fmt::Debug;
}

/// Supplies the native tensor type with one fewer axis and the same kind.
pub trait PrevRank: Clone + std::fmt::Debug {
    /// The rank-D - 1 tensor type. Removing an axis requires an appropriate Burn operation.
    type Prev: NextRank<Next = Self>;
}

// Split the range in two at each step to keep macro recursion shallow.
macro_rules! impl_ranks {
    ($rank:expr;) => {
        impl<K: Basic> NextRank for Tensor<{ $rank }, K> {
            type Next = Tensor<{ $rank + 1 }, K>;
        }

        impl<K: Basic> PrevRank for Tensor<{ $rank + 1 }, K> {
            type Prev = Tensor<{ $rank }, K>;
        }
    };
    ($start:expr; $offset:literal $(, $rest:literal)*) => {
        impl_ranks!($start; $($rest),*);
        impl_ranks!($start + $offset; $($rest),*);
    };
}

impl_ranks!(1; 512, 256, 128, 64, 32, 16, 8, 4, 2, 1);

#[cfg(test)]
mod tests {
    use burn::nn::{Initializer, Linear, LinearConfig};
    use burn::tensor::{
        DType, Device, Tensor,
        kind::{Basic, Bool, Int},
    };

    use super::{NextRank, PrevRank};

    // Store the native associated tensor without a public output-rank parameter.
    struct Batch<T: NextRank> {
        observations: T::Next,
    }

    impl<const D: usize, const B: usize, K: Basic> Batch<Tensor<D, K>>
    where
        Tensor<D, K>: NextRank<Next = Tensor<B, K>>,
    {
        /// Stacks matching rank-D observations into `[batch_size, ...observation_shape]`, with B = D + 1.
        /// Preserves kind, dtype, device, and gradients. Burn requires matching input shapes and devices.
        fn new(observations: Vec<Tensor<D, K>>) -> Self {
            const {
                assert!(
                    B == D + 1,
                    "batch rank must equal observation rank plus one"
                )
            };
            Self {
                observations: Tensor::stack(observations, 0),
            }
        }

        /// Returns the stored rank-B tensor, preserving every axis, dtype, device, and gradient connection.
        fn into_tensor(self) -> Tensor<B, K> {
            self.observations
        }
    }

    impl<const D: usize, const B: usize> Batch<Tensor<D>>
    where
        Tensor<D>: NextRank<Next = Tensor<B>>,
    {
        /// Selects the first batch item, forwards its features, and sums squared outputs into shape `[1]`.
        /// The final axis must match the model's input features. Preserves gradient connections.
        fn loss(self, model: &Linear) -> Tensor<1> {
            let selected = self.observations.slice_dim(0, 0..1);
            let output = model.forward(selected);
            (output.clone() * output).sum()
        }
    }

    /// Removes a leading size-one axis from `[1, ...item_shape]`, returning rank P = D - 1.
    /// Preserves kind, dtype, device, and gradients. Burn requires axis 0 to have size one.
    fn remove_batch<const D: usize, const P: usize, K: Basic>(
        tensor: Tensor<D, K>,
    ) -> <Tensor<D, K> as PrevRank>::Prev
    where
        Tensor<D, K>: PrevRank<Prev = Tensor<P, K>>,
    {
        const { assert!(D == P + 1, "input rank must equal output rank plus one") };
        tensor.squeeze_dim(0)
    }

    #[test]
    fn inferred_rank_returns_a_native_tensor() {
        let device = Device::flex();
        let first = Tensor::<1>::from_data([1.0f32, 2.0], &device);
        let second = Tensor::<1>::from_data([3.0f32, 4.0], &device);
        let batch: Batch<Tensor<1>> = Batch::new(vec![first, second]);
        let tensor: Tensor<2> = batch.into_tensor();
        assert_eq!(tensor.dims(), [2, 2]);
        assert_eq!(
            tensor.into_data().try_to_vec::<f32>().unwrap(),
            [1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn predecessor_returns_a_native_tensor() {
        let device = Device::flex();
        let input = Tensor::<2>::from_data([[1.0f32, 2.0]], &device);
        let output: Tensor<1> = remove_batch(input);
        assert_eq!(output.dims(), [2]);
        assert_eq!(output.into_data().try_to_vec::<f32>().unwrap(), [1.0, 2.0]);
    }

    #[test]
    fn mappings_preserve_integer_and_boolean_kinds() {
        let device = Device::flex();
        let integers = Tensor::<1, Int>::from_data([2i32, 3], &device);
        let integers: Tensor<2, Int> = Batch::new(vec![integers]).into_tensor();
        let integers: Tensor<1, Int> = remove_batch(integers);
        assert_eq!(integers.into_data().try_to_vec::<i32>().unwrap(), [2, 3]);
        let booleans = Tensor::<1, Bool>::from_data([true, false], &device);
        let booleans: Tensor<2, Bool> = Batch::new(vec![booleans]).into_tensor();
        let booleans: Tensor<1, Bool> = remove_batch(booleans);
        assert_eq!(
            booleans.into_data().try_to_vec::<bool>().unwrap(),
            [true, false]
        );
    }

    #[test]
    fn mapping_preserves_f64_and_device() {
        let device = Device::flex();
        let input = Tensor::<1>::from_data([1.25f64, 2.5], &device).cast(DType::F64);
        let output: Tensor<2> = Batch::new(vec![input]).into_tensor();
        assert_eq!(output.dtype(), DType::F64);
        assert_eq!(output.device(), device);
        assert_eq!(output.into_data().try_to_vec::<f64>().unwrap(), [1.25, 2.5]);
    }

    #[test]
    fn generic_native_model_operations_preserve_gradients() {
        let device = Device::flex().autodiff();
        let input = Tensor::<2>::from_data([[1.0f32, 2.0]], &device).require_grad();
        let model = LinearConfig::new(2, 1)
            .with_bias(false)
            .with_initializer(Initializer::Ones)
            .init(&device);
        let batch: Batch<Tensor<2>> = Batch::new(vec![input.clone()]);
        let loss = batch.loss(&model);
        assert_eq!(loss.clone().into_scalar::<f32>(), 9.0);
        let gradients = loss.backward();
        assert_eq!(
            input
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [6.0, 6.0]
        );
    }

    #[test]
    fn boundary_mappings_and_chaining_have_exact_native_types() {
        fn next<T: NextRank<Next = N>, N>() {}

        fn prev<T: PrevRank<Prev = P>, P>() {}

        fn round_trip<T: NextRank>()
        where
            T::Next: PrevRank<Prev = T>,
        {
        }

        next::<Tensor<1>, Tensor<2>>();
        next::<Tensor<1024>, Tensor<1025>>();
        prev::<Tensor<2>, Tensor<1>>();
        prev::<Tensor<1024, Bool>, Tensor<1023, Bool>>();
        prev::<Tensor<1025>, Tensor<1024>>();
        round_trip::<Tensor<1>>();
        round_trip::<Tensor<512, Int>>();
        round_trip::<Tensor<1023>>();
        round_trip::<Tensor<1024>>();
    }
}
