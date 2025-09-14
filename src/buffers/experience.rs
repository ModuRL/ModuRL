use candle_core::Tensor;

pub trait Experience {
    type Error;
    fn get_elements(&self) -> Result<Vec<Tensor>, Self::Error>;
}
