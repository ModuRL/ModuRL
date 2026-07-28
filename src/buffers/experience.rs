use candle_core::Tensor;

pub trait Experience: Clone {
    type Batch;
    type Error;

    fn batch(experiences: &[Self]) -> Result<Self::Batch, Self::Error>;
}

/// Stacks one tensor field shaped `[...]` from each experience into `[batch, ...]`.
pub(crate) fn stack_tensor_field<T>(
    experiences: &[T],
    select: fn(&T) -> Tensor,
) -> Result<Tensor, candle_core::Error> {
    Tensor::stack(&experiences.iter().map(select).collect::<Vec<_>>(), 0)
}

/// Converts one `[item_count]` boolean field from each experience into an
/// `f32` mask shaped `[batch, item_count]`.
pub(crate) fn stack_bool_field<T>(
    experiences: &[T],
    select: for<'a> fn(&'a T) -> &'a [bool],
    device: &candle_core::Device,
) -> Result<Tensor, candle_core::Error> {
    let item_count = select(
        experiences
            .first()
            .expect("cannot stack a boolean field from an empty experience batch"),
    )
    .len();
    let values = experiences
        .iter()
        .flat_map(|experience| {
            select(experience)
                .iter()
                .map(|&value| if value { 1.0f32 } else { 0.0 })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (experiences.len(), item_count), device)
}
