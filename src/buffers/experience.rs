use candle_core::Tensor;

pub trait Experience {
    fn get_elements(&self) -> Vec<Tensor>;
}
