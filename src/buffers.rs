pub mod experience_replay;
pub mod rollout_buffer;
use candle_core::Tensor;
pub mod experience;
use experience::Experience;

pub struct ExperienceBatch<T>
where
    T: Experience,
{
    elements: Vec<Tensor>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ExperienceBatch<T>
where
    T: Experience,
{
    pub fn new(experiences: Vec<T>) -> Self {
        let mut tensor_elements = vec![];
        for experience in experiences.iter() {
            let mut experience_elements = experience.get_elements();
            for (i, element) in experience_elements.iter_mut().enumerate() {
                *element = element.unsqueeze(0).unwrap();
                if tensor_elements.len() <= i {
                    tensor_elements.push(element.clone());
                } else {
                    tensor_elements[i] = Tensor::cat(&[&tensor_elements[i], &element], 0).unwrap();
                }
            }
        }
        Self {
            elements: tensor_elements,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn get_elements(&self) -> Vec<Tensor> {
        self.elements.clone()
    }
}
