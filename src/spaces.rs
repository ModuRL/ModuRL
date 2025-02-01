use ndarray::{ArrayD, IxDyn, IxDynImpl, StrideShape};
use rand::{rng, Rng};

trait Space<T> {
    fn sample(&self) -> T;
    fn contains(&self, x: &T) -> bool;
}

struct Discrete {
    dims: Vec<usize>,
}

impl Space<ArrayD<bool>> for Discrete {
    fn sample(&self) -> ArrayD<bool> {
        let mut total_num_of_bools = 1;
        for dim in &self.dims {
            total_num_of_bools *= dim;
        }
        let mut bools = vec![];
        let mut rng = rng();
        for _ in 0..(total_num_of_bools / 64) {
            let chunk = rng.random::<u64>();
            for i in 0..64 {
                if bools.len() < total_num_of_bools {
                    bools.push((chunk >> i) & 1 == 1);
                }
            }
        }
        ArrayD::from_shape_vec(self.dims.clone(), bools).unwrap()
    }

    fn contains(&self, x: &ArrayD<bool>) -> bool {
        // This is kinda weird, because if the shape is not equal,
        // Should we just say false, or should we return an error?
        if x.shape() != self.dims {
            return false;
        }
        true
    }
}
