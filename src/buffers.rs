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
    pub fn new(experiences: Vec<T>) -> Result<Self, T::Error> {
        //let mut tensor_elements = vec![];
        let mut elements = vec![];
        for experience in experiences.iter() {
            let mut experience_elements = experience.get_elements()?;
            for (i, element) in experience_elements.iter_mut().enumerate() {
                if elements.len() <= i {
                    elements.push(vec![element.clone()]);
                } else {
                    elements[i].push(element.clone());
                }
            }
        }

        let mut tensor_elements = vec![];
        for element_group in elements.iter_mut() {
            let stacked = Tensor::stack(element_group, 0).unwrap();
            tensor_elements.push(stacked);
        }

        Ok(Self {
            elements: tensor_elements,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn get_elements(&self) -> Vec<Tensor> {
        self.elements.clone()
    }
}
