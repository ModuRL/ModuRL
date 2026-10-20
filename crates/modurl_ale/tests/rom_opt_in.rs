use burn::tensor::{DType, Device, Int, Tensor};
use modurl::{
    gym::Gym,
    spaces::ObservationSpace,
    tensor_rank::{NextRank, PrevRank},
};
use modurl_ale::{
    AtariGym, AtariGymError, AtariObservationMode, AtariObservationSpace, GrayscaleScreen, Ram,
    RgbScreen,
};
use std::path::PathBuf;

fn check_mode<M, const D: usize, const B: usize>(path: PathBuf)
where
    M: AtariObservationMode<
            Space = AtariObservationSpace<B>,
            Item = Tensor<D, Int>,
            Batch = Tensor<B, Int>,
        >,
    Tensor<D, Int>: NextRank<Next = Tensor<B, Int>>,
    Tensor<B, Int>: PrevRank<Prev = Tensor<D, Int>>,
{
    let mut env = AtariGym::<M>::builder()
        .rom_path(path)
        .random_seed(42)
        .repeat_action_probability(0.0)
        .build()
        .unwrap();
    env.set_frame_skip(4);
    let reset = env.reset().unwrap();
    assert_eq!(
        reset.observation.dims().as_slice(),
        env.observation_space().shape()
    );
    assert_eq!(reset.observation.dtype(), DType::U8);
    assert_eq!(reset.observation.device(), Device::flex());
    assert!(matches!(
        env.step(Tensor::from_data([-1i32], &Device::flex())),
        Err(AtariGymError::ActionIndex { .. })
    ));
    let step = env
        .step(Tensor::from_data([0i32], &Device::flex()))
        .unwrap();
    assert_eq!(step.observation.dtype(), DType::U8);
    assert_eq!(step.observation.dims(), reset.observation.dims());
    assert!(step.info.frame_number > reset.info.frame_number);
}

#[test]
fn user_supplied_rom_can_reset_and_step_all_observation_modes() {
    let Some(path) = std::env::var_os("MODURL_ALE_TEST_ROM") else {
        eprintln!("skipping: MODURL_ALE_TEST_ROM is not set");
        return;
    };
    let path = PathBuf::from(path);
    check_mode::<Ram, 1, 2>(path.clone());
    check_mode::<RgbScreen, 3, 4>(path.clone());
    check_mode::<GrayscaleScreen, 2, 3>(path);
}
