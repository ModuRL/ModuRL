# Atari benchmarks

This suite measures end-to-end Breakout PPO and DQN throughput in ModuRL, Stable-Baselines3 (SB3), and adapted CleanRL runners.
Each runner uses the settings in [workloads.json](workloads.json). The orchestrator checks reported settings and collection counts.

## Run

Supply a local Breakout ROM. The runners load that same file without copying or installing it.

```console
python crates/benches/run.py --algorithm ppo --device cuda --rom roms/breakout.bin
python crates/benches/run.py --algorithm dqn --device cuda --rom roms/breakout.bin
python crates/benches/run.py --algorithm all --device cuda --rom roms/breakout.bin
```

Install Python dependencies in a virtual environment:

```console
python -m venv crates/benches/.venv
crates/benches/.venv/Scripts/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
crates/benches/.venv/Scripts/python -m pip install -r crates/benches/requirements.txt
```

On Linux or macOS, use `crates/benches/.venv/bin/python`.
Use `--python <path>` to select another Python environment.
Native compilation requires CMake and a C/C++ toolchain.

Use `--framework modurl`, `--framework sb3`, or `--framework cleanrl` to select runners. Repeat the flag to select several.
The default runs all three frameworks, with five fresh-process samples each and shuffled framework order.
`--algorithm all` runs PPO, then DQN, sequentially and writes a separate report for each.
`--steps` sets measured transitions, `--repeats` sets samples, and `--threads` sets tensor compute threads.
PPO still uses 32 environment workers when `--threads 1` is selected.

CUDA requires an NVIDIA driver and a CUDA-enabled PyTorch installation.
CUDA is the default benchmark target for all three frameworks.
Metal requires macOS and PyTorch MPS support.
The orchestrator enables the matching Cargo feature. Missing accelerators produce errors rather than a CPU fallback.
Accelerator features enable model operation fusion in ModuRL.
Python runners compile their CNN encoders with PyTorch Inductor when a CUDA-compatible Triton installation is available.
They report fusion as unavailable when that compiler is missing, including standard Windows PyTorch installations.
Compiler setup occurs during warmup; reports record each runner's fusion status.

## Shared workload

| Setting | PPO | DQN |
| --- | --- | --- |
| Environments | 32 parallel workers | One synchronous environment |
| Model | Shared Nature CNN, actor and critic heads | Nature CNN, four-action Q head |
| Trainable parameters | 1,686,693 | 1,686,180 online parameters |
| Observations | U8 `[batch_size, 4, 84, 84]` | U8 `[batch_size, 4, 84, 84]` |
| Collection | 128 steps per worker; 4,096-transition rollout | One transition per step |
| Updates | Four epochs; four 1,024-transition minibatches per epoch | One 32-transition update per four steps |
| Replay | Not applicable | 1M transitions on CPU; sampled batches move to the training device |
| Adam | Rate `2.5e-4`, epsilon `1e-5` | Rate `1e-4`, epsilon `1e-8` |
| Warmup | One complete rollout and update | 80,000 collection steps, then four steps and one update |
| Measured sample | 40,960 transitions; 160 optimizer steps | 4,096 transitions; 1,024 optimizer steps |

Environment output, rollout storage, and replay samples remain U8.
Models convert observations to F32 and divide by 255 inside their forward methods.
The DQN replay shares each next observation stack with the next transition's current stack.
Its stack storage needs about 26.3 GiB at capacity, plus metadata and saved truncation observations.
It does not deduplicate individual overlapping frames.

All environments use minimal actions, no sticky actions, up to 30 initial no-ops, action repeat four, and two-frame max pooling.
They also use episodic lives, FIRE reset, clipped rewards, 84-by-84 grayscale frames, and four-frame stacks.
PPO limits games to 400,000 emulator frames; DQN uses 108,000 frames.
These limits count emulator frames, rather than repeated agent actions.

PPO uses gamma 0.99, GAE lambda 0.95, clip range 0.1, entropy coefficient 0.01, and gradient norm limit 0.5.
Advantages use sample standard deviation within each minibatch.
The value objective is the maximum of clipped and unclipped squared errors, with total-loss coefficient 0.25.
DQN uses vanilla one-step targets, MSE without gradient clipping, gamma 0.99, and a hard target copy every 1,000 transitions.
Its epsilon decreases from 1.0 to 0.01 over 1M absolute transitions, including warmup.
Learning rates remain constant during both benchmarks.

## Timing and framework adaptations

Timing includes inference, stepping and resets, worker communication, storage, targets, losses, gradients, optimizer steps, and target updates.
It excludes process startup, compilation, construction, and warmup. Devices synchronize at the timing boundaries.
DQN warmup includes one update to initialize training kernels before timing.
Measured DQN transitions continue from warmup without resetting the environments.

ModuRL uses worker threads. SB3 uses `SubprocVecEnv`; CleanRL uses Gymnasium `AsyncVectorEnv`.
ModuRL initializes native environments serially before starting parallel reset and step commands, because ALE constructors change shared process data.
The Python workers use the spawn start method. Reports record thread or process workers and emulator versions.
ModuRL links bundled ALE 0.6; Python uses ALE 0.11.2. These versions and their random-number generators can produce different trajectories.
The suite matches preprocessing settings and training work, but does not claim identical emulator frames or rewards.

SB3 retains its policy, collection, rollout, replay, and scheduling machinery.
Its adapted PPO training method uses the shared value objective; its adapted DQN method uses MSE without gradient clipping.
Policy adapters pass byte observations directly into the feature model, including terminal-value prediction.
The replay adapter preserves truncated terminal observations and samples distinct transition rows.

CleanRL runners adapt [ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo_atari.py)
and [dqn_atari.py](https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/dqn_atari.py).
They use parallel PPO workers, U8 storage, constant learning rates, and terminal-value bootstrapping for truncations.
They retain the shared clipped value objective and use the same replay adapter as SB3.
Logging, evaluation, checkpointing, and learning-rate annealing are excluded.

The suite measures throughput, not final reward or sample efficiency.
Historical CartPole and Pendulum reports remain under `results`; their timings do not describe these Atari workloads.
`requirements-cpu-windows.txt` records the historical Python environment and is not the Atari installation list.

## Results and checks

The orchestrator writes every timing sample, versions, worker details, configuration, ROM SHA-256, and system metadata to JSON.
Complete runner output is saved under `target/benchmark-logs`; failed runs also print their output in the terminal.
Default filenames include `breakout`, algorithm, device, platform, and date, keeping historical reports separate.
Use `--output <path>` to select a report path.

```console
cargo test --locked -p modurl-benches
cargo clippy --locked -p modurl-benches --all-targets --no-deps -- -D warnings
crates/benches/.venv/Scripts/python -m unittest discover -s crates/benches/python
```

Unit tests use small synthetic byte observations and require no ROM.
