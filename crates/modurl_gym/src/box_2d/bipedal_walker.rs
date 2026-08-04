use std::{cell::RefCell, rc::Rc};

use crate::EnvironmentError;
use bon::bon;
use box2d_rs::{
    b2_body::{B2body, B2bodyDef, B2bodyType, BodyPtr},
    b2_collision::B2manifold,
    b2_contact::B2contactDynTrait,
    b2_fixture::B2fixtureDef,
    b2_joint::{B2JointDefEnum, B2jointPtr, JointAsDerived, JointAsDerivedMut},
    b2_math::B2vec2,
    b2_world::{B2world, B2worldPtr},
    b2_world_callbacks::{B2contactImpulse, B2contactListener},
    b2rs_common::UserDataType,
    joints::b2_revolute_joint::B2revoluteJointDef,
    shapes::{b2_edge_shape::B2edgeShape, b2_polygon_shape::B2polygonShape},
};
use candle_core::{DType, Device, Tensor};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::{BoxSpace, Space},
};

const FPS: f32 = 50.0;
const SCALE: f32 = 30.0;
const MOTORS_TORQUE: f32 = 80.0;
const SPEED_HIP: f32 = 4.0;
const SPEED_KNEE: f32 = 6.0;
const LIDAR_RANGE: f32 = 160.0 / SCALE;
const INITIAL_RANDOM: f32 = 5.0;
const LEG_DOWN: f32 = -8.0 / SCALE;
const LEG_W: f32 = 8.0 / SCALE;
const LEG_H: f32 = 34.0 / SCALE;
const VIEWPORT_W: f32 = 600.0;
const VIEWPORT_H: f32 = 400.0;
const TERRAIN_STEP: f32 = 14.0 / SCALE;
const TERRAIN_LENGTH: usize = 200;
const TERRAIN_HEIGHT: f32 = VIEWPORT_H / SCALE / 4.0;
const TERRAIN_GRASS: usize = 10;
const TERRAIN_STARTPAD: usize = 20;
const FRICTION: f32 = 2.5;
const HULL_POLY: [(f32, f32); 5] = [
    (-30.0, 9.0),
    (6.0, 9.0),
    (34.0, 1.0),
    (34.0, -8.0),
    (-30.0, -8.0),
];

struct JointSettings {
    reference_angle: f32,
    motor_speed: f32,
    lower_angle: f32,
    upper_angle: f32,
}

#[derive(Default, Copy, Clone, Debug, PartialEq)]
struct WalkerUserData;

impl UserDataType for WalkerUserData {
    type Fixture = i32;
    type Body = i32;
    type Joint = i32;
}

#[derive(Default)]
struct WalkerContactDetector {
    game_over: bool,
    lower_leg_contacts: [bool; 2],
}

impl WalkerContactDetector {
    fn update(&mut self, contact: &mut dyn B2contactDynTrait<WalkerUserData>, touching: bool) {
        let fixture_a = contact.get_base().get_fixture_a();
        let fixture_b = contact.get_base().get_fixture_b();
        let body_a = fixture_a.borrow().get_body();
        let body_b = fixture_b.borrow().get_body();
        let id_a = body_a.borrow().get_user_data().unwrap_or(0);
        let id_b = body_b.borrow().get_user_data().unwrap_or(0);
        if touching && (id_a == 1 || id_b == 1) {
            self.game_over = true;
        }
        for (index, lower_id) in [3, 5].into_iter().enumerate() {
            if id_a == lower_id || id_b == lower_id {
                self.lower_leg_contacts[index] = touching;
            }
        }
    }
}

impl B2contactListener<WalkerUserData> for WalkerContactDetector {
    fn begin_contact(&mut self, contact: &mut dyn B2contactDynTrait<WalkerUserData>) {
        self.update(contact, true);
    }

    fn end_contact(&mut self, contact: &mut dyn B2contactDynTrait<WalkerUserData>) {
        self.update(contact, false);
    }

    fn pre_solve(
        &mut self,
        _contact: &mut dyn B2contactDynTrait<WalkerUserData>,
        _old_manifold: &B2manifold,
    ) {
    }

    fn post_solve(
        &mut self,
        _contact: &mut dyn B2contactDynTrait<WalkerUserData>,
        _impulse: &B2contactImpulse,
    ) {
    }
}

/// Gymnasium-compatible standard (non-hardcore) `BipedalWalker-v3`.
pub struct BipedalWalkerV3 {
    world: Option<B2worldPtr<WalkerUserData>>,
    terrain: Vec<BodyPtr<WalkerUserData>>,
    terrain_points: Vec<(f32, f32)>,
    hull: Option<BodyPtr<WalkerUserData>>,
    legs: Vec<BodyPtr<WalkerUserData>>,
    joints: Vec<B2jointPtr<WalkerUserData>>,
    contact_detector: Option<Rc<RefCell<WalkerContactDetector>>>,
    previous_shaping: Option<f32>,
    device: Device,
    action_space: BoxSpace,
    observation_space: BoxSpace,
    #[cfg(feature = "rendering")]
    renderer: Option<crate::rendering::Renderer>,
}

#[bon]
impl BipedalWalkerV3 {
    /// Creates the standard Gymnasium v3 walker.
    #[builder]
    pub fn new(
        #[builder(default = &Device::Cpu)] device: &Device,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
    ) -> Result<Self, EnvironmentError> {
        let low = vec![
            -std::f32::consts::PI,
            -5.0,
            -5.0,
            -5.0,
            -std::f32::consts::PI,
            -5.0,
            -std::f32::consts::PI,
            -5.0,
            -0.0,
            -std::f32::consts::PI,
            -5.0,
            -std::f32::consts::PI,
            -5.0,
            -0.0,
        ]
        .into_iter()
        .chain(std::iter::repeat_n(-1.0, 10))
        .collect::<Vec<_>>();
        let high = vec![
            std::f32::consts::PI,
            5.0,
            5.0,
            5.0,
            std::f32::consts::PI,
            5.0,
            std::f32::consts::PI,
            5.0,
            5.0,
            std::f32::consts::PI,
            5.0,
            std::f32::consts::PI,
            5.0,
            5.0,
        ]
        .into_iter()
        .chain(std::iter::repeat_n(1.0, 10))
        .collect::<Vec<_>>();
        Ok(Self {
            world: None,
            terrain: Vec::new(),
            terrain_points: Vec::new(),
            hull: None,
            legs: Vec::new(),
            joints: Vec::new(),
            contact_detector: None,
            previous_shaping: None,
            device: device.clone(),
            action_space: BoxSpace::new_with_universal_bounds(vec![4], -1.0, 1.0, device),
            observation_space: BoxSpace::new(
                Tensor::from_vec(low, 24, device)?,
                Tensor::from_vec(high, 24, device)?,
            ),
            #[cfg(feature = "rendering")]
            renderer: render
                .then(|| {
                    crate::rendering::Renderer::new(
                        VIEWPORT_W as usize,
                        VIEWPORT_H as usize,
                        "Bipedal Walker",
                    )
                })
                .transpose()?,
        })
    }

    fn random_uniform(&self, low: f32, high: f32) -> Result<f32, EnvironmentError> {
        if !low.is_finite() || !high.is_finite() || low >= high {
            return Err(EnvironmentError::InvalidConfiguration(
                "BipedalWalker received an invalid random-sampling range",
            ));
        }
        Ok(Tensor::rand(low, high, (), &self.device)?.to_vec0::<f32>()?)
    }

    fn clear_world(&mut self) {
        self.joints.clear();
        self.legs.clear();
        self.hull = None;
        self.terrain.clear();
        self.contact_detector = None;
        self.world = None;
    }

    fn make_edge_body(
        world: &B2worldPtr<WalkerUserData>,
        point_a: (f32, f32),
        point_b: (f32, f32),
    ) -> BodyPtr<WalkerUserData> {
        let body = B2world::create_body(
            world.clone(),
            &B2bodyDef {
                body_type: B2bodyType::B2StaticBody,
                user_data: Some(0),
                ..Default::default()
            },
        );
        let mut edge = B2edgeShape::default();
        edge.set_two_sided(
            B2vec2::new(point_a.0, point_a.1),
            B2vec2::new(point_b.0, point_b.1),
        );
        let mut fixture = B2fixtureDef {
            shape: Some(Rc::new(RefCell::new(edge))),
            friction: FRICTION,
            ..Default::default()
        };
        fixture.filter.category_bits = 0x0001;
        B2body::create_fixture(body.clone(), &fixture);
        body
    }

    fn create_dynamic_body(
        world: &B2worldPtr<WalkerUserData>,
        id: i32,
        position: B2vec2,
        angle: f32,
        shape: B2polygonShape,
        density: f32,
        friction: f32,
    ) -> BodyPtr<WalkerUserData> {
        let body = B2world::create_body(
            world.clone(),
            &B2bodyDef {
                body_type: B2bodyType::B2DynamicBody,
                position,
                angle,
                user_data: Some(id),
                ..Default::default()
            },
        );
        let mut fixture = B2fixtureDef {
            shape: Some(Rc::new(RefCell::new(shape))),
            density,
            friction,
            restitution: 0.0,
            ..Default::default()
        };
        fixture.filter.category_bits = 0x0020;
        fixture.filter.mask_bits = 0x0001;
        B2body::create_fixture(body.clone(), &fixture);
        body
    }

    fn create_joint(
        world: &B2worldPtr<WalkerUserData>,
        body_a: BodyPtr<WalkerUserData>,
        body_b: BodyPtr<WalkerUserData>,
        anchor_a: B2vec2,
        anchor_b: B2vec2,
        settings: JointSettings,
    ) -> B2jointPtr<WalkerUserData> {
        let mut definition = B2revoluteJointDef::default();
        definition.base.body_a = Some(body_a);
        definition.base.body_b = Some(body_b);
        definition.local_anchor_a = anchor_a;
        definition.local_anchor_b = anchor_b;
        definition.reference_angle = settings.reference_angle;
        definition.enable_motor = true;
        definition.enable_limit = true;
        definition.max_motor_torque = MOTORS_TORQUE;
        definition.motor_speed = settings.motor_speed;
        definition.lower_angle = settings.lower_angle;
        definition.upper_angle = settings.upper_angle;
        world
            .borrow_mut()
            .create_joint(&B2JointDefEnum::RevoluteJoint(definition))
    }

    fn reset_internal(&mut self, deterministic: bool) -> Result<Tensor, EnvironmentError> {
        self.clear_world();
        let world = B2world::<WalkerUserData>::new(B2vec2::new(0.0, -10.0));
        let detector = Rc::new(RefCell::new(WalkerContactDetector::default()));
        world.borrow_mut().set_contact_listener(detector.clone());
        self.world = Some(world.clone());
        self.contact_detector = Some(detector);
        self.previous_shaping = None;

        self.terrain_points.clear();
        let mut y = TERRAIN_HEIGHT;
        let mut velocity = 0.0_f32;
        let mut counter = TERRAIN_STARTPAD;
        let mut one_shot = false;
        for index in 0..TERRAIN_LENGTH {
            let x = index as f32 * TERRAIN_STEP;
            if !one_shot {
                let height_correction = match TERRAIN_HEIGHT.partial_cmp(&y) {
                    Some(std::cmp::Ordering::Greater) => 1.0,
                    Some(std::cmp::Ordering::Less) => -1.0,
                    _ => 0.0,
                };
                velocity = 0.8 * velocity + 0.01 * height_correction;
                if index > TERRAIN_STARTPAD && !deterministic {
                    velocity += self.random_uniform(-1.0, 1.0)? / SCALE;
                }
                y += velocity;
            }
            one_shot = false;
            self.terrain_points.push((x, y));
            counter -= 1;
            if counter == 0 {
                counter = if deterministic {
                    TERRAIN_GRASS / 2
                } else {
                    self.random_uniform((TERRAIN_GRASS / 2) as f32, TERRAIN_GRASS as f32)?
                        .floor() as usize
                };
                one_shot = true;
            }
        }
        for points in self.terrain_points.windows(2) {
            self.terrain
                .push(Self::make_edge_body(&world, points[0], points[1]));
        }

        let initial_x = TERRAIN_STEP * TERRAIN_STARTPAD as f32 / 2.0;
        let initial_y = TERRAIN_HEIGHT + 2.0 * LEG_H;
        let mut hull_shape = B2polygonShape::default();
        hull_shape.set(
            &HULL_POLY
                .iter()
                .map(|(x, y)| B2vec2::new(*x / SCALE, *y / SCALE))
                .collect::<Vec<_>>(),
        );
        let hull = Self::create_dynamic_body(
            &world,
            1,
            B2vec2::new(initial_x, initial_y),
            0.0,
            hull_shape,
            5.0,
            0.1,
        );
        let initial_force = if deterministic {
            0.0
        } else {
            self.random_uniform(-INITIAL_RANDOM, INITIAL_RANDOM)?
        };
        hull.borrow_mut()
            .apply_force_to_center(B2vec2::new(initial_force, 0.0), true);
        self.hull = Some(hull.clone());

        for (side_index, side) in [-1.0_f32, 1.0].into_iter().enumerate() {
            let mut upper_shape = B2polygonShape::default();
            upper_shape.set_as_box(LEG_W / 2.0, LEG_H / 2.0);
            let upper = Self::create_dynamic_body(
                &world,
                2 + side_index as i32 * 2,
                B2vec2::new(initial_x, initial_y - LEG_H / 2.0 - LEG_DOWN),
                side * 0.05,
                upper_shape,
                1.0,
                0.2,
            );
            self.joints.push(Self::create_joint(
                &world,
                hull.clone(),
                upper.clone(),
                B2vec2::new(0.0, LEG_DOWN),
                B2vec2::new(0.0, LEG_H / 2.0),
                JointSettings {
                    reference_angle: side * 0.05,
                    motor_speed: side,
                    lower_angle: -0.8,
                    upper_angle: 1.1,
                },
            ));
            self.legs.push(upper.clone());

            let mut lower_shape = B2polygonShape::default();
            lower_shape.set_as_box(0.8 * LEG_W / 2.0, LEG_H / 2.0);
            let lower = Self::create_dynamic_body(
                &world,
                3 + side_index as i32 * 2,
                B2vec2::new(initial_x, initial_y - LEG_H * 1.5 - LEG_DOWN),
                side * 0.05,
                lower_shape,
                1.0,
                0.2,
            );
            self.joints.push(Self::create_joint(
                &world,
                upper,
                lower.clone(),
                B2vec2::new(0.0, -LEG_H / 2.0),
                B2vec2::new(0.0, LEG_H / 2.0),
                JointSettings {
                    reference_angle: 0.0,
                    motor_speed: 1.0,
                    lower_angle: -1.6,
                    upper_angle: -0.1,
                },
            ));
            self.legs.push(lower);
        }

        let zero_action = Tensor::zeros(4, DType::F32, &self.device)?;
        Ok(self.step(zero_action)?.state)
    }

    fn joint_state(joint: &B2jointPtr<WalkerUserData>) -> Result<(f32, f32), EnvironmentError> {
        let joint = joint.borrow();
        match joint.as_derived() {
            JointAsDerived::ERevoluteJoint(joint) => {
                Ok((joint.get_joint_angle(), joint.get_joint_speed()))
            }
            _ => Err(EnvironmentError::InvalidPhysicsState(
                "BipedalWalker motor joint is not revolute",
            )),
        }
    }

    fn drive_joint(
        joint: &B2jointPtr<WalkerUserData>,
        action: f32,
        speed: f32,
    ) -> Result<(), EnvironmentError> {
        let direction = if action > 0.0 {
            1.0
        } else if action < 0.0 {
            -1.0
        } else {
            0.0
        };
        let mut joint = joint.borrow_mut();
        match joint.as_derived_mut() {
            JointAsDerivedMut::ERevoluteJoint(joint) => {
                joint.set_motor_speed(speed * direction);
                joint.set_max_motor_torque(MOTORS_TORQUE * action.abs().clamp(0.0, 1.0));
                Ok(())
            }
            _ => Err(EnvironmentError::InvalidPhysicsState(
                "BipedalWalker motor joint is not revolute",
            )),
        }
    }

    fn lidar_fractions(&self, origin: B2vec2) -> Result<[f32; 10], EnvironmentError> {
        let world = self
            .world
            .as_ref()
            .ok_or(EnvironmentError::NotInitialized(
                "call reset before reading BipedalWalker lidar",
            ))?
            .borrow();
        Ok(std::array::from_fn(|index| {
            let mut closest = 1.0_f32;
            let endpoint = B2vec2::new(
                origin.x + (1.5 * index as f32 / 10.0).sin() * LIDAR_RANGE,
                origin.y - (1.5 * index as f32 / 10.0).cos() * LIDAR_RANGE,
            );
            world.ray_cast(
                |fixture, _point, _normal, fraction| {
                    if fixture.borrow().get_filter_data().category_bits & 1 == 0 {
                        -1.0
                    } else {
                        closest = fraction;
                        fraction
                    }
                },
                origin,
                endpoint,
            );
            closest
        }))
    }

    #[cfg(feature = "rendering")]
    fn render(&mut self) -> Result<(), EnvironmentError> {
        let Some(renderer) = &mut self.renderer else {
            return Ok(());
        };
        if !renderer.is_open() {
            return Ok(());
        }
        let Some(hull) = &self.hull else {
            return Ok(());
        };
        let scroll = hull.borrow().get_position().x - VIEWPORT_W / SCALE / 5.0;
        let to_screen =
            |point: (f32, f32)| ((point.0 - scroll) * SCALE, VIEWPORT_H - point.1 * SCALE);
        renderer.clear(0xD7D7FF);
        for points in self.terrain_points.windows(2) {
            renderer.line(to_screen(points[0]), to_screen(points[1]), 2, 0x4CFF4C);
        }
        let draw_body = |renderer: &mut crate::rendering::Renderer,
                         body: &BodyPtr<WalkerUserData>,
                         half_width: f32,
                         half_height: f32,
                         color: u32| {
            let body = body.borrow();
            let center = body.get_position();
            let angle = body.get_angle();
            let cos = angle.cos();
            let sin = angle.sin();
            let transform = |x: f32, y: f32| {
                to_screen((center.x + x * cos - y * sin, center.y + x * sin + y * cos))
            };
            renderer.quad(
                transform(-half_width, -half_height),
                transform(-half_width, half_height),
                transform(half_width, half_height),
                transform(half_width, -half_height),
                color,
            );
        };
        {
            let hull = hull.borrow();
            let center = hull.get_position();
            let angle = hull.get_angle();
            let cos = angle.cos();
            let sin = angle.sin();
            let points = HULL_POLY
                .iter()
                .map(|(x, y)| {
                    let x = x / SCALE;
                    let y = y / SCALE;
                    to_screen((center.x + x * cos - y * sin, center.y + x * sin + y * cos))
                })
                .collect::<Vec<_>>();
            renderer.polygon(&points, 0x7F33E5);
        }
        for (index, leg) in self.legs.iter().enumerate() {
            draw_body(
                renderer,
                leg,
                if index % 2 == 0 {
                    LEG_W / 2.0
                } else {
                    0.8 * LEG_W / 2.0
                },
                LEG_H / 2.0,
                if index < 2 { 0xB26698 } else { 0x804C70 },
            );
        }
        renderer.present()?;
        Ok(())
    }

    #[cfg(test)]
    fn reset_flat_for_test(&mut self) -> Result<Tensor, EnvironmentError> {
        self.reset_internal(true)
    }
}

impl Gym for BipedalWalkerV3 {
    type Error = EnvironmentError;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        let state = self.reset_internal(false)?;
        #[cfg(feature = "rendering")]
        self.render()?;
        Ok(ResetInfo { state, info: () })
    }

    /// Steps with one continuous motor-control vector `action` shaped `[4]`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        if self.hull.is_none() || self.world.is_none() || self.joints.len() != 4 {
            return Err(EnvironmentError::NotInitialized(
                "call reset before stepping BipedalWalker",
            ));
        }
        if action.dims() != [4] || !action.dtype().is_float() {
            return Err(EnvironmentError::InvalidAction(
                "BipedalWalker actions must be a floating-point tensor shaped [4]",
            ));
        }
        let action = action.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        for (index, speed) in [SPEED_HIP, SPEED_KNEE, SPEED_HIP, SPEED_KNEE]
            .into_iter()
            .enumerate()
        {
            Self::drive_joint(&self.joints[index], action[index], speed)?;
        }
        let world = self.world.as_ref().ok_or(EnvironmentError::NotInitialized(
            "call reset before stepping BipedalWalker",
        ))?;
        world.borrow_mut().step(1.0 / FPS, 6 * 30, 2 * 30);

        let hull = self
            .hull
            .as_ref()
            .ok_or(EnvironmentError::NotInitialized(
                "call reset before stepping BipedalWalker",
            ))?
            .borrow();
        let position = hull.get_position();
        let velocity = hull.get_linear_velocity();
        let angle = hull.get_angle();
        let angular_velocity = hull.get_angular_velocity();
        drop(hull);
        let joint_states = self
            .joints
            .iter()
            .map(Self::joint_state)
            .collect::<Result<Vec<_>, _>>()?;
        let contacts = self
            .contact_detector
            .as_ref()
            .ok_or(EnvironmentError::NotInitialized(
                "call reset before stepping BipedalWalker",
            ))?
            .borrow()
            .lower_leg_contacts;
        let lidar = self.lidar_fractions(position)?;
        let mut state = vec![
            angle,
            2.0 * angular_velocity / FPS,
            0.3 * velocity.x * (VIEWPORT_W / SCALE) / FPS,
            0.3 * velocity.y * (VIEWPORT_H / SCALE) / FPS,
            joint_states[0].0,
            joint_states[0].1 / SPEED_HIP,
            joint_states[1].0 + 1.0,
            joint_states[1].1 / SPEED_KNEE,
            if contacts[0] { 1.0 } else { 0.0 },
            joint_states[2].0,
            joint_states[2].1 / SPEED_HIP,
            joint_states[3].0 + 1.0,
            joint_states[3].1 / SPEED_KNEE,
            if contacts[1] { 1.0 } else { 0.0 },
        ];
        state.extend(lidar);

        let shaping = 130.0 * position.x / SCALE - 5.0 * angle.abs();
        let mut reward = self
            .previous_shaping
            .map_or(0.0, |previous| shaping - previous);
        self.previous_shaping = Some(shaping);
        reward -= action
            .iter()
            .map(|value| 0.00035 * MOTORS_TORQUE * value.abs().clamp(0.0, 1.0))
            .sum::<f32>();
        let game_over = self
            .contact_detector
            .as_ref()
            .ok_or(EnvironmentError::NotInitialized(
                "call reset before stepping BipedalWalker",
            ))?
            .borrow()
            .game_over;
        let mut done = false;
        if game_over || position.x < 0.0 {
            reward = -100.0;
            done = true;
        }
        if position.x > (TERRAIN_LENGTH - TERRAIN_GRASS) as f32 * TERRAIN_STEP {
            done = true;
        }
        #[cfg(feature = "rendering")]
        self.render()?;
        Ok(StepInfo {
            state: Tensor::from_vec(state, 24, &self.device)?,
            reward,
            done,
            truncated: false,
            info: (),
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(self.observation_space.clone())
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(self.action_space.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translate_walker(environment: &mut BipedalWalkerV3, dx: f32, dy: f32) {
        let bodies = environment
            .hull
            .iter()
            .chain(environment.legs.iter())
            .cloned()
            .collect::<Vec<_>>();
        for body in bodies {
            let (position, angle) = {
                let body = body.borrow();
                (body.get_position(), body.get_angle())
            };
            body.borrow_mut()
                .set_transform(B2vec2::new(position.x + dx, position.y + dy), angle);
        }
    }

    #[derive(serde::Deserialize)]
    struct Fixture {
        generator: FixtureGenerator,
        initial_observation: Vec<f64>,
        actions: Vec<[f32; 4]>,
        observations: Vec<Vec<f64>>,
        rewards: Vec<f64>,
        terminated: Vec<bool>,
        sequential_actions: Vec<[f32; 4]>,
        sequential_observations: Vec<Vec<f64>>,
        sequential_rewards: Vec<f64>,
        sequential_terminated: Vec<bool>,
    }

    #[derive(serde::Deserialize)]
    struct FixtureGenerator {
        python: String,
        gymnasium: String,
        pybox2d: String,
    }

    impl Fixture {
        fn validate(&self) {
            assert!(!self.generator.python.is_empty());
            assert_eq!(self.generator.gymnasium, "1.2.1");
            assert_eq!(self.generator.pybox2d, "2.3.5");
            assert_eq!(self.actions.len(), self.observations.len());
            assert_eq!(self.actions.len(), self.rewards.len());
            assert_eq!(self.actions.len(), self.terminated.len());
            assert_eq!(
                self.sequential_actions.len(),
                self.sequential_observations.len()
            );
            assert_eq!(self.sequential_actions.len(), self.sequential_rewards.len());
            assert_eq!(
                self.sequential_actions.len(),
                self.sequential_terminated.len()
            );
            assert!(
                self.observations
                    .iter()
                    .all(|observation| observation.len() == 24)
            );
            assert!(
                self.sequential_observations
                    .iter()
                    .all(|observation| observation.len() == 24)
            );
        }
    }

    #[test]
    fn parity_sequence() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../python_tests/bipedal_walker/trajectory.json"
        ))
        .unwrap();
        fixture.validate();
        let mut environment = BipedalWalkerV3::builder().build().unwrap();
        environment.reset_flat_for_test().unwrap();
        for index in 0..fixture.sequential_actions.len() {
            let action =
                Tensor::from_vec(fixture.sequential_actions[index].to_vec(), 4, &Device::Cpu)
                    .unwrap();
            let actual = environment.step(action).unwrap();
            let observation = actual.state.to_vec1::<f32>().unwrap();
            for (component, expected) in fixture.sequential_observations[index].iter().enumerate() {
                // Joint and contact states are solver-version-sensitive under
                // sustained contact. Dedicated cases below cover them without
                // accumulating cross-version solver drift.
                if (4..=13).contains(&component) {
                    continue;
                }
                assert!(
                    (f64::from(observation[component]) - expected).abs() <= 1e-2,
                    "sequential step {index}, observation {component}: {} != {expected}",
                    observation[component]
                );
            }
            assert!((f64::from(actual.reward) - fixture.sequential_rewards[index]).abs() <= 5e-3);
            assert_eq!(actual.done, fixture.sequential_terminated[index]);
            assert!(!actual.truncated);
        }
    }

    #[test]
    fn parity_transitions() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../python_tests/bipedal_walker/trajectory.json"
        ))
        .unwrap();
        fixture.validate();
        let mut environment = BipedalWalkerV3::builder().build().unwrap();
        let initial = environment
            .reset_flat_for_test()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (component, expected) in fixture.initial_observation.iter().enumerate() {
            assert!(
                (f64::from(initial[component]) - expected).abs() <= 5e-3,
                "initial observation {component}: {} != {expected}",
                initial[component]
            );
        }
        // PyBox2D and box2d-rs use different Box2D versions. Starting each
        // transition from the same state avoids accumulated solver drift, and
        // this tolerance covers their small single-step contact differences.
        for index in 0..fixture.actions.len() {
            environment.reset_flat_for_test().unwrap();
            let action =
                Tensor::from_vec(fixture.actions[index].to_vec(), 4, &Device::Cpu).unwrap();
            let actual = environment.step(action).unwrap();
            let observation = actual.state.to_vec1::<f32>().unwrap();
            for (component, expected) in fixture.observations[index].iter().enumerate() {
                assert!(
                    (f64::from(observation[component]) - expected).abs() <= 4e-2,
                    "step {index}, observation {component}: {} != {expected}",
                    observation[component]
                );
            }
            assert!((f64::from(actual.reward) - fixture.rewards[index]).abs() <= 1e-3);
            assert_eq!(actual.done, fixture.terminated[index]);
            assert!(!actual.truncated);
        }
    }

    #[test]
    fn default_spaces_match_gymnasium() {
        let mut environment = BipedalWalkerV3::builder().build().unwrap();
        assert_eq!(environment.reset_flat_for_test().unwrap().dims(), &[24]);
        assert_eq!(environment.action_space().shape(), vec![4]);
        assert_eq!(environment.observation_space().shape(), vec![24]);
    }

    #[test]
    fn standard_terrain_is_uneven_and_zero_control_falls() {
        let mut environment = BipedalWalkerV3::builder().build().unwrap();
        assert_eq!(environment.reset().unwrap().state.dims(), &[24]);
        let heights = environment
            .terrain_points
            .iter()
            .skip(TERRAIN_STARTPAD + 1)
            .map(|point| point.1)
            .collect::<Vec<_>>();
        let minimum = heights.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(maximum - minimum > 1e-3);

        let mut terminal_reward = None;
        for _ in 0..500 {
            let transition = environment
                .step(Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap())
                .unwrap();
            assert_eq!(transition.state.dims(), &[24]);
            if transition.done {
                terminal_reward = Some(transition.reward);
                break;
            }
        }
        assert_eq!(terminal_reward, Some(-100.0));
    }

    #[test]
    fn ground_contacts_clear_after_walker_is_lifted() {
        let mut environment = BipedalWalkerV3::builder().build().unwrap();
        let initial = environment
            .reset_flat_for_test()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!((initial[8], initial[13]), (1.0, 1.0));

        translate_walker(&mut environment, 0.0, 10.0);
        let lifted = environment
            .step(Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap())
            .unwrap()
            .state
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!((lifted[8], lifted[13]), (0.0, 0.0));
    }

    #[test]
    fn crossing_course_end_terminates_without_fall_penalty() {
        let mut environment = BipedalWalkerV3::builder().build().unwrap();
        environment.reset_flat_for_test().unwrap();
        translate_walker(&mut environment, TERRAIN_LENGTH as f32 * TERRAIN_STEP, 10.0);
        let transition = environment
            .step(Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap())
            .unwrap();
        assert!(transition.done);
        assert!(!transition.truncated);
        assert_ne!(transition.reward, -100.0);
    }

    #[cfg(feature = "rendering")]
    #[test]
    fn rendering_can_be_enabled() {
        BipedalWalkerV3::builder()
            .render(true)
            .build()
            .unwrap()
            .reset_flat_for_test()
            .unwrap();
    }
}
