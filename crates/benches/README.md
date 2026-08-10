# ModuRL framework benchmarks

`modurl-benches` is an unpublished workspace crate for matched end-to-end reinforcement learning benchmarks.

The suite compares ModuRL with runners based on Stable-Baselines3, CleanRL reference implementations, and Tianshou across PPO, DQN, and SAC. It uses documented adaptations where framework defaults would perform different work. It saves every timing sample and reports each Python runner's elapsed time relative to ModuRL.

## Usage

After installing the Python packages described below, select an algorithm from the workspace root:

```console
python crates/benches/run.py --algorithm ppo --device cpu
python crates/benches/run.py --algorithm dqn --device cpu
python crates/benches/run.py --algorithm sac --device cpu
```

## CPU results

The checked-in results were measured on Windows with a 12th Gen Intel Core i7-1255U and one compute thread. Each value is the median of five fresh-process samples. A larger relative value means ModuRL completed the same algorithm workload sooner.

| Algorithm | Framework | Median time | Transitions/s | ModuRL speedup |
| --- | --- | ---: | ---: | ---: |
| PPO | ModuRL 0.1.0 | 2.102 s | 9,742 | 1.00x |
| PPO | Stable-Baselines3 2.9.0 | 9.551 s | 2,144 | 4.54x |
| PPO | Adapted CleanRL `ppo.py` | 9.390 s | 2,181 | 4.47x |
| PPO | Tianshou 2.0.1 | 13.007 s | 1,575 | 6.19x |
| DQN | ModuRL 0.1.0 | 0.476 s | 8,607 | 1.00x |
| DQN | Adapted Stable-Baselines3 2.9.0 | 2.341 s | 1,750 | 4.92x |
| DQN | Adapted CleanRL `dqn.py` | 1.503 s | 2,725 | 3.16x |
| DQN | Adapted Tianshou 2.0.1 | 6.248 s | 656 | 13.13x |
| SAC | ModuRL 0.1.0 | 2.198 s | 466 | 1.00x |
| SAC | Stable-Baselines3 2.9.0 | 6.079 s | 168 | 2.77x |
| SAC | Adapted CleanRL `sac_continuous_action.py` | 5.269 s | 194 | 2.40x |
| SAC | Tianshou 2.0.1 | 10.186 s | 101 | 4.63x |

The raw reports contain versions, every elapsed time, minimum and maximum times, and system metadata:

- [PPO CPU samples](results/ppo-cpu-windows-20260810.json)
- [DQN CPU samples](results/dqn-cpu-windows-20260810.json)
- [SAC CPU samples](results/sac-cpu-windows-20260810.json)

Do not compare transitions per second between algorithm rows. The algorithms perform different amounts of optimizer work per transition. The fair comparison is across frameworks within one algorithm.

## Fairness controls

Every runner for an algorithm uses the same environment, network topology, trainable parameter count, `float32` tensors, optimizer settings, collection count, update count, and target or return calculation. The runners assert their parameter count and emit their complete workload configuration. The orchestrator stops if those configuration objects differ.

The timed region includes policy inference, environment stepping and resets, rollout or replay storage, return or target calculation, loss computation, backpropagation, optimizer steps, and target-network updates. Device synchronization occurs immediately before and after timing.

The timed region excludes imports, process startup, Cargo compilation, environment construction, network construction, and warmup collection or training. Each CPU runner uses one compute thread. Framework order is shuffled for every repeat.

ModuRL uses its Rust Gymnasium-parity CartPole and Pendulum environments. Python runners use Gymnasium 1.2.2. Termination and time-limit truncation are handled separately for replay targets. Candle 0.11.0 cannot seed its CPU random number generator, so CPU trajectories are not seed-identical. Tensor shapes and the amount of training work are fixed, and trajectory differences do not change the loop structure being timed.

The suite measures implementation throughput. It does not measure sample efficiency, final reward, environment fidelity outside these tasks, or performance with large networks.

## PPO workload

| Setting | Value |
| --- | --- |
| Environment | Eight synchronous `CartPole-v1` environments |
| Collection | 256 steps per environment; 2,048 transitions per rollout |
| Networks | Separate 2x64 Tanh actor and critic; 9,155 trainable parameters |
| Optimizer | Adam; learning rate `3e-4`; epsilon `1e-5`; no weight decay |
| Update | Minibatch 64; 10 epochs; clip 0.2; no value-loss clipping |
| Returns | Gamma 0.99; generalized advantage estimation lambda 0.95 |
| Loss | Per-minibatch advantage normalization; value coefficient 0.5; entropy coefficient 0 |
| Gradients | Global norm clipped to 0.5 |
| Sample | One 2,048-transition warmup update; then 20,480 measured transitions |

Stable-Baselines3 and Tianshou use their public PPO and collection APIs. The CleanRL runner derives from [`cleanrl/ppo.py` revision `e421c2e5`](https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py). It removes logging and CleanRL's extra 0.5 value-loss factor so `vf_coef=0.5` has the same meaning in all four runners.

## DQN workload

| Setting | Value |
| --- | --- |
| Environment | One `CartPole-v1` environment |
| Networks | Online and target 2x64 Tanh Q-networks; 4,610 trainable online parameters |
| Replay | Capacity 10,000; batch 64; one-step targets |
| Optimizer | Adam; learning rate `2.5e-4`; epsilon `1e-5`; no weight decay |
| Update | Vanilla DQN; MSE loss; one update per four transitions; no gradient clipping |
| Target | Hard copy every 1,000 environment transitions |
| Exploration | Constant epsilon 0.1 |
| Discount | Gamma 0.99 |
| Sample | 1,024 replay warmup transitions; then 4,096 measured transitions and 1,024 updates |

Tianshou uses its `DQN`, policy, collector, and replay implementations with Double DQN disabled and MSE selected. A small subclass drives target copies from absolute environment transitions instead of Tianshou's optimizer-update counter. The CleanRL runner derives from [`cleanrl/dqn.py` revision `e421c2e5`](https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/dqn.py), with logging and schedules removed.

Stable-Baselines3 normally hard-codes Huber loss and gradient clipping for DQN. The benchmark subclass changes only its training update to the shared MSE loss with no clip. Collection, policy, replay, scheduling, and target-network behavior still use Stable-Baselines3. The result is labeled adapted rather than presented as an unmodified default.

## SAC workload

| Setting | Value |
| --- | --- |
| Environment | One `Pendulum-v1` environment with a 200-step time limit |
| Actor | 2x64 Tanh trunk with state-dependent mean and log standard deviation |
| Critics | Two 2x64 Tanh scalar critics and two target copies |
| Parameters | 13,637 trainable actor, online-critic, and temperature parameters |
| Replay | Capacity 10,000; batch 64; one-step targets |
| Optimizers | Adam; learning rate `3e-4`; epsilon `1e-5`; no weight decay |
| Update | One actor, twin-critic, and temperature update per transition |
| Targets | Minimum critic value; gamma 0.99; soft-update coefficient 0.005 |
| Entropy | Automatic tuning; initial alpha 1.0; target entropy -1.0 |
| Sample | 256 random-action replay warmup transitions; then 1,024 measured transitions and updates |

Stable-Baselines3 and Tianshou use their public SAC, policy, collection, and replay APIs. The CleanRL runner derives from [`cleanrl/sac_continuous_action.py` revision `e421c2e5`](https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/sac_continuous_action.py). It removes logging and changes the delayed actor cadence to one actor update per transition, matching the other implementations.

## Set up Python

Create an environment inside this crate and install the pinned packages.

On Windows PowerShell:

```console
python -m venv crates/benches/.venv
crates/benches/.venv/Scripts/python -m pip install -r crates/benches/requirements-cpu-windows.txt
```

On Linux or macOS:

```console
python3 -m venv crates/benches/.venv
crates/benches/.venv/bin/python -m pip install -r crates/benches/requirements.txt
```

`requirements-cpu-windows.txt` freezes the complete environment used for the checked-in CPU results. `requirements.txt` pins the direct dependencies and remains portable across operating systems and accelerator-specific PyTorch wheels.

## Run the comparisons

With the virtual environment active, select an algorithm and device from the workspace root:

```console
python crates/benches/run.py --algorithm ppo --device cpu
python crates/benches/run.py --algorithm dqn --device cpu
python crates/benches/run.py --algorithm sac --device cpu
```

Use `--python <path>` if the dependencies are in another Python environment. Use `--framework modurl`, `--framework sb3`, `--framework cleanrl`, or `--framework tianshou` to select runners. Repeat `--framework` to select more than one.

The command prints a Markdown table and writes a dated, algorithm-specific JSON file under `crates/benches/results`. It fails if runners report different workload configurations.

## Run CUDA or Metal

CUDA requires an NVIDIA CUDA toolkit supported by Candle 0.11.0 and a CUDA-enabled PyTorch 2.10.0 installation. Install the appropriate PyTorch wheel, then substitute `--device cuda` in any command above.

On macOS, Metal requires a working Candle Metal toolchain and PyTorch Metal Performance Shaders support. Substitute `--device metal` in any command above.

The orchestrator enables the matching Candle Cargo feature and selects the corresponding PyTorch device. A missing backend produces an error instead of silently using the CPU.

CPU was the only available backend on the checked-in result host. CUDA and Metal results are intentionally not estimated from CPU measurements. Add separate raw result files from those devices before making accelerator claims.
