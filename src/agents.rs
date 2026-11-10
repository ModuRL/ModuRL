use crate::tensor_rank::TensorBatchTypes;
use crate::{
    gym::{MultiGym, MultiGymStepInfo},
    spaces::{ActionSpace, ObservationSpace},
};
use burn::tensor::{Tensor, kind::Basic};

pub mod a2c;
pub mod action_value_critic;
pub mod deterministic_actor_critic;

pub mod ppo;
pub mod q_learning;
mod replay_device_strategy;
pub mod sac;

pub(crate) use replay_device_strategy::ReplayStorageConfig;

#[derive(Clone)]
pub(crate) struct SlotValues<T> {
    dense: Vec<T>,
    sparse: std::collections::BTreeMap<usize, T>,
    zero: T,
}

impl<T: Default + Clone> SlotValues<T> {
    fn new(count: usize) -> Self {
        Self {
            dense: vec![T::default(); count],
            sparse: Default::default(),
            zero: T::default(),
        }
    }

    fn clear(&mut self) {
        self.dense.clear();
        self.sparse.clear();
    }

    fn len(&self) -> usize {
        self.dense.len()
    }
}

impl<T> std::ops::Index<usize> for SlotValues<T> {
    type Output = T;
    fn index(&self, slot: usize) -> &T {
        self.dense
            .get(slot)
            .or_else(|| self.sparse.get(&slot))
            .unwrap_or(&self.zero)
    }
}

impl<T: Default> std::ops::IndexMut<usize> for SlotValues<T> {
    fn index_mut(&mut self, slot: usize) -> &mut T {
        if slot == self.dense.len() {
            self.dense
                .push(self.sparse.remove(&slot).unwrap_or_default());
        }
        if slot < self.dense.len() {
            &mut self.dense[slot]
        } else {
            self.sparse.entry(slot).or_default()
        }
    }
}

/// Training actions selected by [`Agent::act_for_training`] and converted for the environment on demand.
pub trait ToEnvironmentActions {
    type Actions;
    type Error;

    /// Returns the action batch passed to `MultiGym::step`, shape `[num_envs, ...action_shape]`.
    fn to_environment_actions(&self) -> Result<Self::Actions, Self::Error>;
}

impl<const D: usize, K: Basic> ToEnvironmentActions for Tensor<D, K> {
    type Actions = Self;
    type Error = std::convert::Infallible;

    fn to_environment_actions(&self) -> Result<Self, Self::Error> {
        Ok(self.clone())
    }
}

/// Selects native environment actions and trains through batched environment interaction.
/// Ranks `O` and `A` include the batch axis; `PrevRank` determines each unbatched terminal observation type.
/// Scalar observations use `[batch_size, 1]` and terminal observations use `[1]`.
/// Associated spaces determine the native observation and action kinds.
pub trait Agent<I = (), const O: usize = 2, const A: usize = 2> {
    type Error: From<<Self::TrainingActions as ToEnvironmentActions>::Error>;
    /// Environment type accepted by `learn`; built-in agents use `dyn MultiGym + 'e` with their gym error type.
    type Env<'e>: ?Sized
        + MultiGym<
            I,
            O,
            A,
            ObservationSpace = Self::ObservationSpace,
            ActionSpace = Self::ActionSpace,
        >;

    type ObservationSpace: ObservationSpace<O>;
    type ActionSpace: ActionSpace<A>;

    /// Extra input `act_for_training` and `act` need beyond observations, such as an environment to search; `()` when observations suffice.
    type ActContext<'c>: ?Sized;

    /// Training actions used by `learn_from_timesteps`.
    type TrainingActions: ToEnvironmentActions<
        Actions = <Self::ActionSpace as TensorBatchTypes>::Batch,
    >;

    /// Selects training actions from observations shaped `[batch_size, ...observation_shape]`.
    fn act_for_training(
        &mut self,
        observations: &<Self::ObservationSpace as TensorBatchTypes>::Batch,
        context: &mut Self::ActContext<'_>,
    ) -> Result<Self::TrainingActions, Self::Error>;

    /// Selects environment actions from observations shaped `[batch_size, ...observation_shape]`.
    fn act(
        &mut self,
        observations: &<Self::ObservationSpace as TensorBatchTypes>::Batch,
        context: &mut Self::ActContext<'_>,
    ) -> Result<<Self::ActionSpace as TensorBatchTypes>::Batch, Self::Error> {
        Ok(self
            .act_for_training(observations, context)?
            .to_environment_actions()?)
    }

    /// Trains with rank-`O` observation batches and rank-`A` action batches, each with `num_envs` rows.
    /// Terminal observations have rank `O - 1`; scalar terminal observations use `[1]`.
    /// The environment must use the associated space types and meet the agent's dtype and device requirements.
    fn learn(&mut self, env: &mut Self::Env<'_>, num_timesteps: usize) -> Result<(), Self::Error>;

    /// Stores compact observations and training actions with one row per slot in `slots`.
    /// Training actions must come from `act_for_training` on those observations.
    /// Slot IDs must be unique in each call and match the row count; violations return errors.
    /// Empty batches are valid no-ops.
    /// Each slot is a continuous stream: pass its rows in time order, starting a new episode
    /// only after its done or truncated row. Slots may disappear between calls and have any ID.
    /// Schedules need the builder's `training_horizon` because this method has no total timestep count.
    fn learn_from_timesteps(
        &mut self,
        observations: &<Self::ObservationSpace as TensorBatchTypes>::Batch,
        training_actions: Self::TrainingActions,
        step: MultiGymStepInfo<I, <Self::ObservationSpace as TensorBatchTypes>::Item>,
        slots: &[usize],
    ) -> Result<(), Self::Error>;
}

#[cfg(test)]
pub(crate) mod test_support {
    use burn::tensor::{DType, Device, Int, Tensor};
    use std::{cell::Cell, convert::Infallible};

    use crate::{
        gym::{Gym, ResetInfo, StepInfo},
        spaces::{BoxSpace, Discrete},
    };

    pub(crate) trait SelectRows: Sized {
        fn select_rows(self, rows: &[usize]) -> Self;
    }

    impl<const D: usize, K: burn::tensor::kind::Basic> SelectRows for Tensor<D, K> {
        fn select_rows(self, rows: &[usize]) -> Self {
            if rows.iter().copied().eq(0..self.dims()[0]) {
                return self;
            }
            let indices = Tensor::<1, Int>::from_data(
                burn::tensor::TensorData::new(
                    rows.iter().map(|&row| row as i64).collect::<Vec<_>>(),
                    [rows.len()],
                ),
                (&self.device(), DType::I64),
            );
            self.select(0, indices)
        }
    }

    pub(crate) fn select_step<I: Clone, T: crate::tensor_rank::NextRank + Clone>(
        step: crate::gym::MultiGymStepInfo<I, T>,
        rows: &[usize],
    ) -> crate::gym::MultiGymStepInfo<I, T>
    where
        T::Next: SelectRows,
    {
        crate::gym::MultiGymStepInfo {
            observations: step.observations.select_rows(rows),
            rewards: step.rewards.select_rows(rows),
            infos: rows.iter().map(|&row| step.infos[row].clone()).collect(),
            dones: rows.iter().map(|&row| step.dones[row]).collect(),
            truncateds: rows.iter().map(|&row| step.truncateds[row]).collect(),
            terminal_observations: rows
                .iter()
                .map(|&row| step.terminal_observations[row].clone())
                .collect(),
        }
    }

    pub(crate) fn learn_selected_rows<G, I: Clone, const O: usize, const A: usize>(
        agent: &mut G,
        observations: &<G::ObservationSpace as crate::tensor_rank::TensorBatchTypes>::Batch,
        actions: G::TrainingActions,
        step: crate::gym::MultiGymStepInfo<
            I,
            <G::ObservationSpace as crate::tensor_rank::TensorBatchTypes>::Item,
        >,
        slots: &[usize],
    ) -> Result<(), G::Error>
    where
        G: super::Agent<I, O, A>,
        <G::ObservationSpace as crate::tensor_rank::TensorBatchTypes>::Batch: Clone + SelectRows,
        <G::ObservationSpace as crate::tensor_rank::TensorBatchTypes>::Item: Clone,
        G::TrainingActions: SelectRows,
    {
        let observations = observations.clone().select_rows(slots);
        agent.learn_from_timesteps(
            &observations,
            actions.select_rows(slots),
            select_step(step, slots),
            slots,
        )
    }

    pub(crate) struct FixedEnv {
        device: Device,
    }

    impl FixedEnv {
        pub(crate) fn new(device: Device) -> Self {
            Self { device }
        }
    }

    impl Gym for FixedEnv {
        type Error = Infallible;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;

        /// Accepts one scalar Int action `[1]` and returns unbatched F32 observations `[4]` on the fixture device.
        fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
            Ok(StepInfo {
                observation: Tensor::zeros([4], (&self.device, DType::F32)),
                reward: 1.0,
                done: false,
                truncated: false,
                info: (),
            })
        }

        /// Returns unbatched F32 observations `[4]` on the fixture device.
        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            Ok(ResetInfo {
                observation: Tensor::zeros([4], (&self.device, DType::F32)),
                info: (),
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new(
                Tensor::zeros([1, 4], (&self.device, DType::F32)),
                Tensor::ones([1, 4], (&self.device, DType::F32)),
            )
        }

        fn action_space(&self) -> Self::ActionSpace {
            Discrete::new(2).unwrap()
        }
    }

    pub(crate) struct BorrowingFixedEnv<'a> {
        inner: FixedEnv,
        calls: &'a Cell<usize>,
    }

    impl<'a> BorrowingFixedEnv<'a> {
        pub(crate) fn new(device: Device, calls: &'a Cell<usize>) -> Self {
            Self {
                inner: FixedEnv::new(device),
                calls,
            }
        }
    }

    impl Gym for BorrowingFixedEnv<'_> {
        type Error = Infallible;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;

        /// Accepts one scalar Int action `[1]` and returns unbatched F32 observations `[4]` on the fixture device.
        fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
            self.calls.set(self.calls.get() + 1);
            self.inner.step(action)
        }

        /// Returns unbatched F32 observations `[4]` on the fixture device.
        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            self.inner.reset()
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            self.inner.observation_space()
        }

        fn action_space(&self) -> Self::ActionSpace {
            self.inner.action_space()
        }
    }

    pub(crate) struct FixedContinuousEnv {
        device: Device,
    }

    impl FixedContinuousEnv {
        pub(crate) fn new(device: Device) -> Self {
            Self { device }
        }
    }

    impl Gym<(), 2, 2> for FixedContinuousEnv {
        type Error = Infallible;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = BoxSpace<2>;

        /// Accepts one Float action vector `[1]` and returns unbatched F32 observations `[4]` on the fixture device.
        fn step(&mut self, _action: Tensor<1>) -> Result<StepInfo, Self::Error> {
            Ok(StepInfo {
                observation: Tensor::zeros([4], (&self.device, DType::F32)),
                reward: 1.0,
                done: false,
                truncated: false,
                info: (),
            })
        }

        /// Returns unbatched F32 observations `[4]` on the fixture device.
        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            Ok(ResetInfo {
                observation: Tensor::zeros([4], (&self.device, DType::F32)),
                info: (),
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new_with_universal_bounds([1, 4], -1.0, 1.0, &self.device)
        }

        fn action_space(&self) -> Self::ActionSpace {
            BoxSpace::new_with_universal_bounds([1, 1], -1.0, 1.0, &self.device)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{BorrowingFixedEnv, FixedContinuousEnv, FixedEnv};
    use super::{Agent, ToEnvironmentActions};
    use crate::{
        gym::{Gym, MultiGym, MultiGymStepInfo, VectorizedGymError, VectorizedGymWrapper},
        spaces::{BoxSpace, Discrete},
    };
    use burn::tensor::{DType, Device, Int, Tensor};
    use std::convert::Infallible;

    struct FixedAgent;

    impl Agent for FixedAgent {
        type Error = Box<dyn std::error::Error>;
        type Env<'e> = dyn MultiGym<
                (),
                2,
                2,
                Error = VectorizedGymError<Infallible>,
                ObservationSpace = BoxSpace<2>,
                ActionSpace = Discrete,
            > + 'e;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;
        type ActContext<'c> = ();
        type TrainingActions = Tensor<2, Int>;

        /// Returns zero Int actions `[batch_size, 1]` for Float observations `[batch_size, features]` on the observation device.
        fn act_for_training(
            &mut self,
            observations: &Tensor<2>,
            _context: &mut Self::ActContext<'_>,
        ) -> Result<Tensor<2, Int>, Self::Error> {
            Ok(Tensor::zeros(
                [observations.dims()[0], 1],
                &observations.device(),
            ))
        }

        /// Steps observation batches `[num_envs, 4]` with scalar action batches `[num_envs, 1]`.
        fn learn(
            &mut self,
            env: &mut Self::Env<'_>,
            num_timesteps: usize,
        ) -> Result<(), Self::Error> {
            let mut observations = env.reset()?;
            for _ in (0..num_timesteps).step_by(env.num_envs()) {
                observations = env.step(self.act(&observations, &mut ())?)?.observations;
            }
            Ok(())
        }

        /// Accepts observations and step data with matching environment batch shapes.
        fn learn_from_timesteps(
            &mut self,
            _observations: &Tensor<2>,
            _training_actions: Tensor<2, Int>,
            _step: MultiGymStepInfo<(), Tensor<1>>,
            _slots: &[usize],
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn agent_accepts_native_tensors_and_environment_batches() {
        let device = Device::flex();
        let mut agent = FixedAgent;
        let mut single = VectorizedGymWrapper::from(FixedEnv::new(device.clone()));
        let mut multiple = VectorizedGymWrapper::new(vec![
            FixedEnv::new(device.clone()),
            FixedEnv::new(device.clone()),
        ])
        .unwrap();
        let observations = multiple.reset().unwrap();
        let training_actions = (|| {
            let training_actions = agent.act_for_training(&observations, &mut ())?;
            Ok::<_, Box<dyn std::error::Error>>(training_actions.to_environment_actions()?)
        })()
        .unwrap();
        let actions = agent.act(&observations, &mut ()).unwrap();
        assert_eq!(
            training_actions.into_data().try_to_vec::<i32>().unwrap(),
            vec![0, 0]
        );
        assert_eq!(actions.dims(), [2, 1]);
        assert_eq!(actions.device(), device);
        assert_eq!(actions.into_data().try_to_vec::<i32>().unwrap(), vec![0, 0]);
        agent.learn(&mut single, 1).unwrap();
        agent.learn(&mut multiple, 2).unwrap();
    }

    struct ContextAgent;

    impl Agent for ContextAgent {
        type Error = Infallible;
        type Env<'e> = dyn MultiGym<
                (),
                2,
                2,
                Error = Infallible,
                ObservationSpace = BoxSpace<2>,
                ActionSpace = Discrete,
            > + 'e;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;
        type ActContext<'c> = [i64];
        type TrainingActions = Tensor<2, Int>;

        /// Returns context-provided Int actions `[batch_size, 1]` for Float observations `[batch_size, features]`.
        fn act_for_training(
            &mut self,
            observations: &Tensor<2>,
            context: &mut [i64],
        ) -> Result<Tensor<2, Int>, Self::Error> {
            Ok(Tensor::from_data(
                burn::tensor::TensorData::new(context.to_vec(), [observations.dims()[0], 1]),
                (&observations.device(), DType::I64),
            ))
        }

        /// Accepts any compatible batched environment without stepping it.
        fn learn(
            &mut self,
            _env: &mut Self::Env<'_>,
            _num_timesteps: usize,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        /// Accepts observations `[num_envs, 4]` and matching caller-collected step data.
        fn learn_from_timesteps(
            &mut self,
            _observations: &Tensor<2>,
            _training_actions: Tensor<2, Int>,
            _step: MultiGymStepInfo<(), Tensor<1>>,
            _slots: &[usize],
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn act_uses_caller_context() {
        let device = Device::flex();
        let observations = Tensor::zeros([2, 4], (&device, DType::F32));
        let mut agent = ContextAgent;

        let actions = agent.act(&observations, &mut [1i64, 0][..]).unwrap();

        assert_eq!(actions.dims(), [2, 1]);
        assert_eq!(actions.into_data().try_to_vec::<i64>().unwrap(), vec![1, 0]);
    }

    #[test]
    fn learn_accepts_environments_borrowing_locals() {
        let calls = std::cell::Cell::new(0usize);
        let mut env = VectorizedGymWrapper::from(BorrowingFixedEnv::new(Device::flex(), &calls));
        FixedAgent.learn(&mut env, 2).unwrap();
        assert!(calls.get() > 0);
    }

    #[test]
    fn fixed_environments_keep_single_and_batched_layouts() {
        let device = Device::flex();
        let mut discrete = FixedEnv::new(device.clone());
        assert_eq!(discrete.reset().unwrap().observation.dims(), [4]);
        assert_eq!(discrete.observation_space().shape(), vec![4]);
        assert_eq!(discrete.action_space().shape(), vec![1]);
        let mut discrete = VectorizedGymWrapper::from(discrete);
        assert_eq!(discrete.reset().unwrap().dims(), [1, 4]);
        let step = discrete
            .step(Tensor::<2, Int>::from_data([[0i32]], &device))
            .unwrap();
        assert_eq!(step.observations.dims(), [1, 4]);
        assert_eq!(step.observations.dtype(), DType::F32);
        assert_eq!(step.rewards.into_scalar::<f32>(), 1.0);
        assert_eq!(step.dones, vec![false]);
        assert_eq!(step.truncateds, vec![false]);

        let mut continuous = FixedContinuousEnv::new(device.clone());
        assert_eq!(continuous.reset().unwrap().observation.dims(), [4]);
        assert_eq!(continuous.action_space().shape(), vec![1]);
        let mut continuous = VectorizedGymWrapper::from(continuous);
        assert_eq!(continuous.reset().unwrap().dims(), [1, 4]);
        let step = continuous
            .step(Tensor::from_data([[0.0f32]], &device))
            .unwrap();
        assert_eq!(step.observations.dims(), [1, 4]);
        assert_eq!(step.rewards.into_scalar::<f32>(), 1.0);
        assert_eq!(step.dones, vec![false]);
        assert_eq!(step.truncateds, vec![false]);
    }
}
