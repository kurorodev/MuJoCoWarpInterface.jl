# MuJoCoWarpInterface.jl

[![CI](https://github.com/kurorodev/MuJoCoWarpInterface.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/kurorodev/MuJoCoWarpInterface.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`MuJoCoWarpInterface.jl` is a Julia interface to [MuJoCo Warp (MJWarp)](https://github.com/google-deepmind/mujoco_warp) for batched GPU physics simulation.

The package is aimed at reinforcement learning, robotics, and control workloads where many copies of the same MuJoCo environment need to be simulated in parallel. It exposes MJWarp simulation state directly as CUDA.jl `CuArray`s using DLPack, allowing Julia code to read state and write controls without copying through CPU memory.

> **Status:** experimental `v0.1`. The public API is intentionally small and may evolve before `v1.0`.

## Features

- Batched MJWarp simulation from Julia with configurable `nworlds`
- Zero-copy interoperability between MJWarp/Warp arrays and CUDA.jl
- Julia access to `qpos`, `qvel`, and `ctrl` as `CuArray`s
- Batched `step!` and full `reset!`
- Automatic Python environment management through CondaPkg.jl
- Automatic installation of `mujoco-warp`
- Support for MuJoCo XML/MJCF model files
- GPU-oriented array layout convenient for Julia neural-network workloads

MJWarp stores batched state as `world × feature`. Through DLPack, Julia exposes the same memory as `feature × world`, which is convenient for batched policy evaluation.

## Requirements

For simulation in `v0.1`:

- Julia `1.10` or newer
- NVIDIA GPU
- Working NVIDIA driver compatible with the installed Warp/CUDA runtime
- CUDA.jl functional on the system

Python does not need to be installed or managed manually. PythonCall.jl and CondaPkg.jl create an isolated environment for the package and install `mujoco-warp` automatically.

`MuJoCoWarpInterface.jl` can be installed and precompiled on systems without an NVIDIA GPU, but the current `Simulation` implementation requires CUDA and will not run physics on AMD GPUs, Apple GPUs, or CPU-only systems.

## Installation

After registration in the Julia General registry:

```julia
using Pkg
Pkg.add("MuJoCoWarpInterface")
```

Then load the package:

```julia
using MuJoCoWarpInterface
```

For development before registration:

```julia
using Pkg
Pkg.add(url="https://github.com/kurorodev/MuJoCoWarpInterface.jl")
```

The first installation may take longer because CondaPkg creates the Python environment and installs MuJoCo Warp and its dependencies.

## Quick start

Assume `pendulum.xml` is a valid MuJoCo model with one actuator:

```julia
using MuJoCoWarpInterface
using CUDA

sim = Simulation(
    "pendulum.xml";
    nworlds = 1024,
)

@show sim
@show size(qpos(sim))
@show size(qvel(sim))
@show size(ctrl(sim))
```

For a one-DoF model with one actuator, the state/control layout is:

```text
qpos: (1, 1024)
qvel: (1, 1024)
ctrl: (1, 1024)
```

Apply the same action to all environments and advance the simulation:

```julia
ctrl(sim) .= 1.0f0

for _ in 1:100
    step!(sim)
end

positions = qpos(sim)
velocities = qvel(sim)
```

Alternatively, pass an action array directly:

```julia
actions = CUDA.fill(
    1.0f0,
    size(ctrl(sim)),
)

step!(sim, actions)
```

Reset all environments:

```julia
reset!(sim)
```

## Zero-copy CUDA interoperability

The main design goal of the package is to avoid a CPU round-trip between Julia and MJWarp.

```text
Julia / CUDA.jl
      │
      │  CuArray
      ▼
shared CUDA allocation
      ▲
      │  Warp array
      │
MuJoCo Warp
```

The arrays returned by:

```julia
qpos(sim)
qvel(sim)
ctrl(sim)
```

are CUDA.jl arrays backed by the same GPU allocations used by MJWarp.

This means control values can be produced by Julia GPU code and consumed by MJWarp without converting to NumPy or copying through host memory:

```julia
actions = CUDA.randn(Float32, size(ctrl(sim)))

copyto!(ctrl(sim), actions)
step!(sim)
```

Likewise, the new simulation state remains available directly to Julia GPU code after the physics step.

## Reinforcement learning use case

A typical batched RL loop can be structured as:

```julia
for t in 1:horizon
    obs = build_observations(qpos(sim), qvel(sim))

    actions = policy(obs)

    step!(sim, actions)

    rewards = compute_rewards(qpos(sim), qvel(sim))
end
```

The intended architecture is:

```text
MJWarp physics
     │
     │ zero-copy CUDA memory
     ▼
Julia observations / rewards
     │
     ▼
Julia policy / PPO
     │
     ▼
actions
     │
     └──────────────► MJWarp
```

`MuJoCoWarpInterface.jl` is a physics/interface layer. It does not implement PPO, policy networks, reward functions, or environment-specific observations.

## Public API

| Function                                     | Description                                                   |
| -------------------------------------------- | ------------------------------------------------------------- |
| `Simulation(path; nworlds=1, device=0, ...)` | Load a MuJoCo model and create batched MJWarp simulation data |
| `qpos(sim)`                                  | Generalized positions as a zero-copy Julia GPU array          |
| `qvel(sim)`                                  | Generalized velocities as a zero-copy Julia GPU array         |
| `ctrl(sim)`                                  | Actuator controls as a zero-copy Julia GPU array              |
| `step!(sim)`                                 | Advance all worlds by one MJWarp physics step                 |
| `step!(sim, actions)`                        | Copy actions into `ctrl` and advance all worlds               |
| `reset!(sim)`                                | Reset all worlds                                              |
| `nworlds(sim)`                               | Number of parallel worlds                                     |
| `nq(sim)`                                    | Number of generalized position coordinates                    |
| `nv(sim)`                                    | Number of generalized velocity coordinates                    |
| `nu(sim)`                                    | Number of actuator controls                                   |
| `backend_info()`                             | Report MuJoCo, MJWarp, and Warp versions                      |

## Backend information

You can inspect the Python-side backend versions with:

```julia
backend_info()
```

Example:

```text
(mujoco = "3.13.0", mujoco_warp = "3.13.0", warp = "1.17.0")
```

For low-level GPU diagnostics, NVIDIA Warp provides its own device diagnostics. Those backend objects are considered implementation details of this package and are not part of the stable public API.

## Testing

Run the test suite with:

```julia
using Pkg
Pkg.test()
```

GPU-specific tests run when `CUDA.functional()` is true.

On CI machines without an NVIDIA GPU, the package installation and backend-loading tests still run while GPU simulation tests are skipped.

The test suite includes checks for simulation construction, batched stepping, reset behavior, and pointer equality between MJWarp allocations and Julia CUDA arrays to guard the zero-copy contract.

## Current limitations

`v0.1` focuses on establishing a small, reliable GPU simulation interface.

The current implementation has the following limitations:

- MJWarp simulation is currently exposed only through the NVIDIA CUDA path.
- AMD ROCm and Apple Metal simulation are not supported.
- CPU physics execution is not currently exposed through `Simulation`.
- Only full-environment reset is part of the current public API.
- Rendering is not currently exposed.
- Contact, sensor, and other MJWarp data arrays are not yet part of the public API.
- `step!` currently uses explicit synchronization between CUDA.jl and Warp for correctness; finer CUDA stream/event synchronization is planned.
- The API should be considered experimental until `v1.0`.

## Roadmap

Planned work includes masked/per-world reset, additional MJWarp state and sensor access, improved CUDA stream synchronization, richer model/device configuration, documentation with Documenter.jl, benchmarking on robotic models such as humanoids, and evaluation of additional backends for non-NVIDIA hardware.

## Contributing

Issues, bug reports, documentation improvements, benchmarks, and pull requests are welcome.

For changes affecting GPU interoperability, please include tests where possible. In particular, changes to DLPack handling should preserve the zero-copy property between MJWarp and Julia arrays.

Before opening a pull request, run:

```julia
using Pkg
Pkg.test()
```

## License

`MuJoCoWarpInterface.jl` is distributed under the MIT License. See [`LICENSE`](LICENSE).

This package interfaces with third-party software that is distributed under its own licenses. In particular, MuJoCo Warp is an external project maintained by Google DeepMind and NVIDIA and is distributed under the Apache-2.0 license. Users should review the licenses of MuJoCo Warp, MuJoCo, NVIDIA Warp, and other transitive dependencies separately.

`MuJoCoWarpInterface.jl` is an independent Julia package and is not an official Google DeepMind, NVIDIA, MuJoCo, or JuliaLang project.
