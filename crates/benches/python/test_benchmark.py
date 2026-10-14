import unittest
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
import benchmark


class ByteGym(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
    action_space = gym.spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.full((4, 84, 84), 10, dtype=np.uint8), {}

    def step(self, action):
        self.steps += 1
        return np.full((4, 84, 84), 20, dtype=np.uint8), 1.0, False, self.steps == 2, {}


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_replay_preserves_truncated_terminal_observations_after_reset(self):
        replay = benchmark.AlignedReplayBuffer(8, ByteGym.observation_space, ByteGym.action_space, device="cpu")
        shape = (1, 4, 84, 84)
        replay.add(np.full(shape, 10, np.uint8), np.full(shape, 200, np.uint8),
                   np.array([1]), np.array([1.0]), np.array([True]), [{"TimeLimit.truncated": True}])
        replay.add(np.full(shape, 20, np.uint8), np.full(shape, 30, np.uint8),
                   np.array([2]), np.array([1.0]), np.array([False]), [{}])
        batch = replay._get_samples(np.array([0, 1]))
        self.assertEqual(batch.observations.dtype, torch.uint8)
        self.assertEqual(batch.next_observations.dtype, torch.uint8)
        self.assertEqual(batch.observations.device.type, "cpu")
        self.assertTrue(torch.all(batch.next_observations[0] == 200))
        self.assertTrue(torch.all(batch.dones == 0))
        self.assertFalse(hasattr(replay, "next_observations"))

    def test_full_replay_samples_unique_rows_and_excludes_frontier(self):
        replay = benchmark.AlignedReplayBuffer(8, ByteGym.observation_space, ByteGym.action_space, device="cpu")
        shape = (1, 4, 84, 84)
        for index in range(10):
            replay.add(np.full(shape, index, np.uint8), np.full(shape, index + 1, np.uint8),
                       np.array([index]), np.array([0.0]), np.array([False]), [{}])
        batch = replay.sample(7)
        self.assertEqual(len(torch.unique(batch.actions)), 7)
        self.assertNotIn(2, batch.actions.flatten().tolist())

    def test_sb3_ppo_retains_bytes_through_collection_and_training(self):
        env = DummyVecEnv([ByteGym, ByteGym])
        model = benchmark.MatchedPPO(benchmark.ByteActorCriticPolicy, env, n_steps=2, batch_size=4,
                                     n_epochs=1, learning_rate=2.5e-4, clip_range=0.1,
                                     vf_coef=0.25, ent_coef=0.01, device="cpu",
                                     policy_kwargs=dict(net_arch=[], features_extractor_class=benchmark.NatureCNN,
                                                        normalize_images=False))
        received = []
        handle = model.policy.features_extractor.register_forward_pre_hook(
            lambda module, inputs: received.append(inputs[0].dtype))
        before = model.policy.action_net.weight.detach().clone()
        try:
            model.learn(4)
            self.assertEqual(model.rollout_buffer.observations.dtype, np.uint8)
            self.assertTrue(received)
            self.assertTrue(all(dtype == torch.uint8 for dtype in received))
            self.assertFalse(torch.equal(before, model.policy.action_net.weight))
            benchmark.assert_parameter_count(model.policy, benchmark.WORKLOADS["ppo"])
        finally:
            handle.remove()
            env.close()

    def test_sb3_dqn_retains_cpu_byte_replay_during_training(self):
        env = DummyVecEnv([ByteGym])
        model = benchmark.MatchedDQN(benchmark.ByteDQNPolicy, env, buffer_size=16, learning_starts=4,
                                     train_freq=4, gradient_steps=1, batch_size=4, target_update_interval=4,
                                     replay_buffer_class=benchmark.AlignedReplayBuffer, device="cpu",
                                     policy_kwargs=dict(net_arch=[], features_extractor_class=benchmark.NatureCNN,
                                                        normalize_images=False))
        received = []
        handle = model.q_net.features_extractor.register_forward_pre_hook(
            lambda module, inputs: received.append(inputs[0].dtype))
        try:
            model.learn(8)
            self.assertEqual(model._n_updates, 1)
            self.assertEqual(model.replay_buffer.observations.dtype, np.uint8)
            self.assertTrue(received)
            self.assertTrue(all(dtype == torch.uint8 for dtype in received))
            benchmark.assert_parameter_count(model.q_net, benchmark.WORKLOADS["dqn"])
        finally:
            handle.remove()
            env.close()

    def test_cleanrl_models_accept_only_native_bytes(self):
        for model, config in [(benchmark.CleanRLPPOModel(ByteGym.observation_space), benchmark.WORKLOADS["ppo"]),
                              (benchmark.CleanRLDQNModel(ByteGym.observation_space), benchmark.WORKLOADS["dqn"])]:
            benchmark.assert_parameter_count(model, config)
            observation = torch.full((1, 4, 84, 84), 127, dtype=torch.uint8)
            output = model.value(observation) if isinstance(model, benchmark.CleanRLPPOModel) else model(observation)
            output.sum().backward()
            self.assertEqual(observation.dtype, torch.uint8)
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(all(parameter.grad is not None for name, parameter in model.named_parameters()
                                if not name.startswith("actor")))
            with self.assertRaises(ValueError):
                model.encoder(observation.float())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_replay_transfers_bytes_to_cuda_models(self):
        replay = benchmark.AlignedReplayBuffer(8, ByteGym.observation_space, ByteGym.action_space, device="cuda")
        shape = (1, 4, 84, 84)
        for index in range(4):
            replay.add(np.full(shape, index, np.uint8), np.full(shape, index + 1, np.uint8),
                       np.array([index]), np.array([1.0]), np.array([False]), [{}])
        self.assertIsInstance(replay.observations, np.ndarray)
        self.assertEqual(replay.observations.dtype, np.uint8)
        batch = replay.sample(4)
        self.assertEqual(batch.observations.device.type, "cuda")
        self.assertEqual(batch.observations.dtype, torch.uint8)
        self.assertEqual(batch.next_observations.dtype, torch.uint8)
        model = benchmark.CleanRLDQNModel(ByteGym.observation_space).cuda()
        model(batch.observations).mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
