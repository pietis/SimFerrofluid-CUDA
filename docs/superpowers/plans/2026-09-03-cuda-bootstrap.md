# CUDA Backend Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to execute this plan task by task,
> and `superpowers:verification-before-completion` before any success claim.

**Goal:** Deliver the first independently buildable CUDA backend slice: exact
plane/grid host contracts, checked device allocation and initialization, and a
standalone `demo_cuda` dry-run executable with compile/runtime CI separated.

**Architecture:** All implementation lives under `cuda/`; the top-level build
only includes the subproject.  CUDA targets never compile or link CPU physics
sources.  Host contracts run on macOS; CUDA 12.8 compile/link runs in CI;
runtime correctness requires an NVIDIA runner.

**Tech stack:** C++20, CUDA 12.8+, CUDA Runtime, Xmake, standalone assertion
tests, GitHub Actions.

**Spec:**
`docs/superpowers/specs/2026-09-03-separate-cuda-backend-design.md`

## Global constraints

- Do not edit CPU production sources in `core/`, `demo/`, or `fmmtl/`.
- No production CUDA target may compile or link those directories.
- Write and run a focused failing test before production behavior.
- Use binary64 and do not enable fast-math.
- Do not introduce numerical clamps, NaN replacement, fallbacks, or automatic
  resolution reduction.
- Unsupported scenes and solvers fail explicitly.
- Compile-only CI is labelled compile-only; runtime checks require NVIDIA.
- This bootstrap is not a fluid solver and must identify itself as such.

---

## Task 1: Add build boundary and exact host scene contract

**Files:**

- Modify: `xmake.lua`
- Create: `cuda/xmake.lua`
- Create: `cuda/core/grid/GridDesc.h`
- Create: `cuda/core/scene/PlaneScene.h`
- Create: `cuda/core/scene/PlaneScene.cpp`
- Create: `cuda/tests/host/PlaneSceneContractTest.cpp`

First create the test and run `xmake build -P . cuda_host_tests`; capture the
expected missing-header or missing-target failure.  Then implement independent,
trivially-copyable scalar types with no Eigen or CPU-core includes.

The test must cover scale 192 resolution `(192,144,192)`, boundary width 2,
spacing `0.12/188`, z-fast index/coordinate round trips including corners,
allocated and interior bounds, fill-plane height, exact zero initial time and
velocity, density, surface tension, gravity, damping, field, susceptibility,
lambda, epsilon factor, and ten IoB iterations.  Scale must be divisible by
four and greater than four; invalid values throw `std::invalid_argument`.

Verify:

```bash
xmake build -P . cuda_host_tests
xmake run -P . cuda_host_tests
rg -n '#include.*(core/|demo/|fmmtl/)' cuda || true
git diff --check
```

Commit as `feat(cuda): add independent plane scene contract`.

## Task 2: Add checked runtime ownership and device plane initialization

**Files:**

- Create: `cuda/core/runtime/CudaError.h`
- Create: `cuda/core/runtime/DeviceBuffer.h`
- Create: `cuda/core/runtime/CudaContext.h`
- Create: `cuda/core/runtime/CudaContext.cu`
- Create: `cuda/core/grid/FieldView.cuh`
- Create: `cuda/core/scene/PlaneScene.cu`
- Create: `cuda/tests/runtime/PlaneInitializationTest.cu`
- Modify: `cuda/xmake.lua`

Write the scale-16 runtime test first and capture its compile failure.  The test
initializes cell phi plus three MAC velocity arrays, copies test snapshots back,
and checks exact zero velocities/time, finite phi, correct plane signs and
values, z-fast storage, and repeat-run bitwise identity.

Create one nonblocking stream.  `DeviceBuffer<T>` is move-only, exposes
size/capacity, checks multiplication overflow, treats zero-sized buffers
without calling CUDA allocation, and never allocates from a kernel loop.  All
runtime calls and kernel launches return contextual errors.  No constructor or
destructor throws from cleanup.

Verify on CUDA 12.8+ NVIDIA:

```bash
xmake f -P . -m debug --cuda=y
xmake build -P . cuda_runtime_tests
xmake run -P . cuda_runtime_tests
compute-sanitizer --tool memcheck build/linux/x86_64/debug/cuda_runtime_tests
```

Commit as `feat(cuda): initialize plane state on device`.

## Task 3: Add standalone dry-run CLI and CI truth split

**Files:**

- Create: `cuda/core/sim/State.h`
- Create: `cuda/core/sim/Simulation.h`
- Create: `cuda/core/sim/Simulation.cu`
- Create: `cuda/demo/Main.cpp`
- Create: `.github/workflows/cuda.yml`
- Create: `cuda/tests/host/CudaSourceBoundaryTest.cpp`
- Modify: `cuda/xmake.lua`

Write the boundary/process assertions first.  The source-boundary test inspects
the resolved `sim_cuda`/`demo_cuda` source and link dependency lists and fails
if CPU physics paths or a CPU fallback option appear.  The CLI command

```bash
demo_cuda --test plane --scale 16 --end 1 --rate 500 --cfl .5 \
  --mag=false --dry-run
```

must initialize device state and print a machine-readable resolved manifest
containing `implementation: cuda`, `capability: bootstrap-only`, all scene
values, CUDA runtime/driver/device metadata, and exact time zero.  It rejects
non-plane scenes, non-IoB solver names, omitted required timing arguments, and
attempts to run frames without `--dry-run` because the fluid operators do not
exist yet.  There is no CPU fallback flag.

The CUDA 12.8 container CI job is named `CUDA compile-only (no GPU)` and only
compiles/links.  A distinct self-hosted NVIDIA job runs runtime tests and
Compute Sanitizer.  Neither job modifies or builds the CPU baseline.

Verify:

```bash
xmake build -P . cuda_host_tests
xmake run -P . cuda_host_tests
xmake build -P . demo_cuda cuda_runtime_tests
xmake run -P . demo_cuda -- --test plane --scale 16 --end 1 \
  --rate 500 --cfl .5 --mag=false --dry-run
git diff --check
```

Commit as `feat(cuda): add standalone plane bootstrap`.

## Final bootstrap verification

Run fresh host tests locally, compile/link in CUDA 12.8 CI, and run runtime
tests only if a named NVIDIA runner is available.  Report exact commands, exit
codes, pass/fail/skip counts, local CPU baseline failure, and the missing
runtime coverage.  Request a whole-bootstrap code review before pushing the
feature branch.

