use candle_core::Error;
use candle_nn::Optimizer;

use crate::spaces;

pub struct PPOActor<O>
where
    O: Optimizer,
{
    clipped: bool,
    gamma: f32,
    gae_lambda: f32,
    clip_range: Option<f32>,
    clip_range_vf: Option<f32>,
    normalize_advantage: bool,
    vf_coef: f32,
    ent_coef: f32,
    max_grad_norm: f32,
    optimizer: O,
    observation_space: Box<dyn spaces::Space>,
    action_space: Box<dyn spaces::Space>,
    critic_network: Box<dyn candle_core::Module>,
    actor_network: Box<dyn candle_core::Module>,
}

pub struct PPOActorBuilder<O>
where
    O: Optimizer,
{
    // Neccesary hyperparameters
    optimizer: O,
    observation_space: Box<dyn spaces::Space>,
    action_space: Box<dyn spaces::Space>,
    critic_network: Box<dyn candle_core::Module>,
    actor_network: Box<dyn candle_core::Module>,
    // Has default values so optional
    clipped: Option<bool>,
    gamma: Option<f32>,
    gae_lambda: Option<f32>,
    clip_range: Option<f32>,
    clip_range_vf: Option<f32>,
    normalize_advantage: Option<bool>,
    vf_coef: Option<f32>,
    ent_coef: Option<f32>,
    max_grad_norm: Option<f32>,
}

impl<O> PPOActorBuilder<O>
where
    O: Optimizer,
{
    pub fn new(
        observation_space: Box<dyn spaces::Space>,
        action_space: Box<dyn spaces::Space>,
        actor_network: Box<dyn candle_core::Module>,
        critic_network: Box<dyn candle_core::Module>,
        optimizer: O,
    ) -> Self {
        Self {
            optimizer,
            observation_space,
            action_space,
            actor_network,
            critic_network,
            clipped: None,
            gamma: None,
            gae_lambda: None,
            clip_range: None,
            clip_range_vf: None,
            normalize_advantage: None,
            vf_coef: None,
            ent_coef: None,
            max_grad_norm: None,
        }
    }

    pub fn clipped(mut self, clipped: bool) -> Self {
        self.clipped = Some(clipped);
        self
    }

    pub fn gamma(mut self, gamma: f32) -> Self {
        self.gamma = Some(gamma);
        self
    }

    pub fn gae_lambda(mut self, gae_lambda: f32) -> Self {
        self.gae_lambda = Some(gae_lambda);
        self
    }

    pub fn clip_range(mut self, clip_range: f32) -> Self {
        self.clip_range = Some(clip_range);
        self
    }

    pub fn clip_range_vf(mut self, clip_range_vf: f32) -> Self {
        self.clip_range_vf = Some(clip_range_vf);
        self
    }

    pub fn normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.normalize_advantage = Some(normalize_advantage);
        self
    }

    pub fn vf_coef(mut self, vf_coef: f32) -> Self {
        self.vf_coef = Some(vf_coef);
        self
    }

    pub fn ent_coef(mut self, ent_coef: f32) -> Self {
        self.ent_coef = Some(ent_coef);
        self
    }

    pub fn max_grad_norm(mut self, max_grad_norm: f32) -> Self {
        self.max_grad_norm = Some(max_grad_norm);
        self
    }

    pub fn build(self) -> PPOActor<O> {
        PPOActor {
            clipped: self.clipped.unwrap_or(true),
            gamma: self.gamma.unwrap_or(0.99),
            gae_lambda: self.gae_lambda.unwrap_or(0.95),
            clip_range: self.clip_range,
            clip_range_vf: self.clip_range_vf,
            normalize_advantage: self.normalize_advantage.unwrap_or(true),
            vf_coef: self.vf_coef.unwrap_or(0.5),
            ent_coef: self.ent_coef.unwrap_or(0.01),
            max_grad_norm: self.max_grad_norm.unwrap_or(0.5),
            optimizer: self.optimizer,
            observation_space: self.observation_space,
            action_space: self.action_space,
            critic_network: self.critic_network,
            actor_network: self.actor_network,
        }
    }
}

impl<O> PPOActor<O>
where
    O: Optimizer,
{
    fn optimize(&mut self) -> Result<(), Error> {
        todo!()
    }
}
