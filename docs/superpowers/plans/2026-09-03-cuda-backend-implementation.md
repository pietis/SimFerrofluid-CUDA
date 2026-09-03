# Separate CUDA Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to execute this plan task by task,
> and `superpowers:verification-before-completion` before any success claim.

**Goal:** Build an independently executable, GPU-resident CUDA implementation
of SimFerrofluid's plane scene while preserving the CPU code as a comparison
baseline.

**Architecture:** New code lives only under `cuda/` and the top-level build
only includes that subproject.  Binary64 cell/MAC fields remain resident on one
CUDA stream across substeps.  Operators are ported and differentially tested in
the CPU phase order.  Small direct-sum magnetics establish an oracle before a
production deterministic GPU FMM is accepted.

**Tech stack:** C++20, CUDA 12.8+, CUDA Runtime, CCCL/CUB, cuSPARSE, cuBLAS,
Xmake, CTest-style standalone assertions, GitHub Actions compile CI, NVIDIA
runtime runner for numerical tests, Nsight Systems/Compute for profiles.

**Spec:**
`docs/superpowers/specs/2026-09-03-separate-cuda-backend-design.md`

## Global constraints

- Do not edit CPU production sources in `core/`, `demo/`, or `fmmtl/`.
- No production CUDA target may compile or link those directories.
- Write a failing focused test before each production behavior.
- Use binary64 and do not enable fast-math.
- Preserve existing numerical guards exactly; record any new guard before use.
- Every GPU test checks the last CUDA error and synchronizes before assertions.
- Compile-only CI is labelled compile-only.  Runtime checks require NVIDIA.
- Commit each task only after reviewing its diff and fresh command output.

---

## Project B0: Independent CUDA foundation

### Task 1: Add the build boundary and exact host scene contract

**Files:**

- Modify: `xmake.lua`
- Create: `cuda/xmake.lua`
- Create: `cuda/core/grid/GridDesc.h`
- Create: `cuda/core/scene/PlaneScene.h`
- Create: `cuda/core/scene/PlaneScene.cpp`
- Create: `cuda/tests/host/PlaneSceneContractTest.cpp`

**Step 1: Write the failing host contract test**

Assert scale 192 produces resolution `(192,144,192)`, boundary width 2,
spacing `0.12/188`, z-fast index round trips, the exact allocated/interior
bounds, plane height, initial time/velocity, material constants, field, chi,
lambda, and ten IoB iterations.  Also assert invalid scales (not divisible by
four or `<=4`) are rejected.

**Step 2: Run the test and observe RED**

```bash
xmake build -P . cuda_host_tests
```

Expected: compilation fails because the CUDA scene contract headers do not yet
exist.

**Step 3: Implement the minimum host-only types**

Use trivially-copyable scalar structs without Eigen or CPU-core includes.  Make
grid indexing and coordinate conversion checked on the host.  Derive values
from the declared scale and constants; do not paste paper-resolution origins as
the implementation.

**Step 4: Run GREEN and audit separation**

```bash
xmake build -P . cuda_host_tests
xmake run -P . cuda_host_tests
rg -n '#include.*(core/|demo/|fmmtl/)' cuda || true
```

Expected: one passing test executable and no CPU production include.

**Step 5: Commit**

```bash
git add xmake.lua cuda/xmake.lua cuda/core/grid/GridDesc.h \
  cuda/core/scene/PlaneScene.h cuda/core/scene/PlaneScene.cpp \
  cuda/tests/host/PlaneSceneContractTest.cpp
git commit -m "feat(cuda): add independent plane scene contract"
```

### Task 2: Add checked runtime ownership and device plane initialization

**Files:**

- Create: `cuda/core/runtime/CudaError.h`
- Create: `cuda/core/runtime/DeviceBuffer.h`
- Create: `cuda/core/runtime/CudaContext.h`
- Create: `cuda/core/runtime/CudaContext.cu`
- Create: `cuda/core/grid/FieldView.cuh`
- Create: `cuda/core/scene/PlaneScene.cu`
- Create: `cuda/tests/runtime/PlaneInitializationTest.cu`
- Modify: `cuda/xmake.lua`

**Step 1: Write the failing runtime test**

At scale 16, initialize cell phi and all three MAC fields, copy only test
snapshots back, and assert exact zero velocities/time, finite phi, z-fast
indices, the analytic plane sign at cells above/below the interface, and exact
field constants.  Repeat initialization and require bitwise-equal snapshots.

**Step 2: Run RED on an NVIDIA runner**

```bash
xmake f -P . -m debug --cuda=y
xmake build -P . cuda_runtime_tests
```

Expected: missing runtime/initialization symbols.

**Step 3: Implement minimal checked CUDA ownership**

Create one nonblocking stream.  `DeviceBuffer<T>` is move-only, records
capacity/size, never allocates in a kernel loop, and reports CUDA errors with
file/expression context.  Implement separate phi and MAC initialization
kernels using `GridDesc`; launch geometry is explicit and checked.

**Step 4: Run GREEN with Compute Sanitizer**

```bash
xmake run -P . cuda_runtime_tests
compute-sanitizer --tool memcheck build/linux/x86_64/debug/cuda_runtime_tests
```

Expected: tests pass and sanitizer reports zero errors.

**Step 5: Commit**

```bash
git add cuda/core/runtime cuda/core/grid/FieldView.cuh \
  cuda/core/scene/PlaneScene.cu cuda/tests/runtime cuda/xmake.lua
git commit -m "feat(cuda): initialize plane state on device"
```

### Task 3: Add standalone CLI and compile/runtime CI split

**Files:**

- Create: `cuda/core/sim/State.h`
- Create: `cuda/core/sim/Simulation.h`
- Create: `cuda/core/sim/Simulation.cu`
- Create: `cuda/demo/Main.cpp`
- Create: `cuda/demo/Driver.h`
- Create: `cuda/demo/Driver.cpp`
- Create: `.github/workflows/cuda.yml`
- Modify: `cuda/xmake.lua`

**Step 1: Write failing CLI/smoke assertions**

Add a process test for `demo_cuda --test plane --scale 16 --end 1
--rate 500 --cfl .5 --mag=false --dry-run`.  It must emit resolved values and
GPU/toolkit metadata, reject unsupported scenes/solvers, and contain no CPU
fallback option.

**Step 2: Run RED**

```bash
xmake build -P . demo_cuda
xmake run -P . demo_cuda -- --test plane --scale 16 --end 1 \
  --rate 500 --cfl .5 --mag=false --dry-run
```

Expected: target or CLI behavior is missing.

**Step 3: Implement the standalone state owner and dry-run driver**

`Simulation` owns `CudaContext`, config, device state, and time initialized to
zero.  The binary links only `sim_cuda` and its direct dependencies.  Dry-run
initializes state and prints a machine-readable resolved manifest without
claiming it advanced physics.

**Step 4: Add CI with explicit truth labels**

Use a CUDA 12.8 development image for `nvcc` compile/link.  The hosted job is
named `CUDA compile-only (no GPU)` and does not invoke runtime tests.  Add a
separate opt-in/self-hosted NVIDIA job for runtime tests and Compute Sanitizer.

**Step 5: Verify and commit**

```bash
xmake build -P . cuda_host_tests
xmake run -P . cuda_host_tests
xmake build -P . demo_cuda cuda_runtime_tests
git add cuda .github/workflows/cuda.yml
git commit -m "feat(cuda): add standalone plane executable"
```

---

## Project M0: Complete nonmagnetic CUDA plane

### Task 4: Port interpolation, RK2 backtrace, damping, and advection

**Files:**

- Create: `cuda/core/operators/Interpolation.cuh`
- Create: `cuda/core/operators/Advection.h`
- Create: `cuda/core/operators/Advection.cu`
- Create: `cuda/tests/runtime/AdvectionTest.cu`
- Create: `cuda/tests/differential/AdvectionDifferentialTest.cpp`

Test constant/affine fields, boundary-clamped tricubic phi interpolation,
trilinear MAC interpolation, RK2 traces, all three velocity components, and
buffer swap ordering.  Seed a reversed-trace-sign mutation and prove the test
fails.  Then integrate damping and both advections in the exact CPU order.

### Task 5: Port collider preprocessing and enforcement

**Files:**

- Create: `cuda/core/operators/Collider.h`
- Create: `cuda/core/operators/Collider.cu`
- Create: `cuda/tests/runtime/ColliderTest.cu`
- Create: `cuda/tests/differential/ColliderDifferentialTest.cpp`

Test tank signed distance, auxiliary cell phi, face fractions/normals, one-layer
phi extrapolation, difference, velocity enforcement, exact masks, containment,
and static zero velocity.  Preserve the released `faceFraction > .9` behavior;
do not introduce a new cutoff.

### Task 6: Port WENO/SSPRK reinitialization

**Files:**

- Create: `cuda/core/operators/Reinitialization.h`
- Create: `cuda/core/operators/Reinitialization.cu`
- Create: `cuda/tests/runtime/ReinitializationTest.cu`
- Create: `cuda/tests/differential/ReinitializationDifferentialTest.cpp`

Compare one WENO derivative, one Euler advance, one SSPRK iteration, and all ten
iterations.  Include constant, planar, sphere, and boundary stencils.  Preserve
the CPU WENO epsilon formula and order of operations; record ULP/L2/Linf errors.

### Task 7: Port deterministic marching cubes and volume

**Files:**

- Create: `cuda/core/surface/MarchingCubesTables.cuh`
- Create: `cuda/core/surface/SurfaceMesh.h`
- Create: `cuda/core/surface/MarchingCubes.h`
- Create: `cuda/core/surface/MarchingCubes.cu`
- Create: `cuda/core/surface/LevelSetGeometry.cu`
- Create: `cuda/core/surface/McVolume.cu`
- Create: `cuda/tests/runtime/MarchingCubesTest.cu`
- Create: `cuda/tests/differential/SurfaceDifferentialTest.cpp`

Use count/scan/emit passes with deterministic edge ownership.  Verify all 256
cell cases using canonical topology, then plane/sphere meshes, positions,
normals, areas, curvature, and volume.  Retain table provenance and source hash.

### Task 8: Assemble and solve pressure on GPU

**Files:**

- Create: `cuda/core/pressure/PressureAssembly.h`
- Create: `cuda/core/pressure/PressureAssembly.cu`
- Create: `cuda/core/pressure/Bicgstab.h`
- Create: `cuda/core/pressure/Bicgstab.cu`
- Create: `cuda/core/pressure/GeometricMg.h`
- Create: `cuda/core/pressure/GeometricMg.cu`
- Create: `cuda/tests/runtime/PressureTest.cu`
- Create: `cuda/tests/differential/PressureDifferentialTest.cpp`

Test exact unknown IDs, sorted CSR coefficients and RHS on hand-derived and
CPU-generated cut-cell grids before adding the solver.  Then test residual and
projected divergence on full-box and free-surface systems at several scales.
Seed an omitted-neighbor and pressure-jump-sign mutation.

### Task 9: Complete nonmagnetic scheduling, export, and restart

**Files:**

- Modify: `cuda/core/sim/Simulation.{h,cu}`
- Modify: `cuda/demo/Driver.{h,cpp}`
- Create: `cuda/core/io/Checkpoint.{h,cpp}`
- Create: `cuda/core/io/Exporter.{h,cpp}`
- Create: `cuda/core/io/Provenance.{h,cpp}`
- Create: `cuda/tests/smoke/PlaneNonmagneticTest.cpp`

Integrate the exact phase order, CFL/final-half-step scheduling, volume PI
state, gravity, six-layer extrapolation, frame export, and versioned
checkpoint.  At scales 16 and 32 require zero NaN/Inf, containment, declared
volume drift, phase traces, restart identity, and CPU lockstep snapshots.

---

## Project M1/M2: Magnetic oracle and production FMM

### Task 10: Add independent CPU and CUDA direct IoB oracles

**Files:**

- Create: `cuda/core/magnetic/DirectIob.h`
- Create: `cuda/core/magnetic/DirectIob.cu`
- Create: `cuda/tests/differential/DirectIobTest.cpp`

Implement a test-only independent CPU sum and a tiled CUDA binary64 sum.  Add a
hard maximum-N contract so production surfaces fail instead of using quadratic
work.  Compare every fixed-point iteration, charges, field components,
residual diagnostic, and final pressure for planes, spheres, random surfaces,
permutations, translations, and zero field.  Seed sign/scale mutations.

### Task 11: Flatten and validate the released CPU FMM plan

**Files:**

- Create: `cuda/core/magnetic/FmmPlan.h`
- Create: `cuda/core/magnetic/CpuPlanBridge.cpp`
- Create: `cuda/tests/differential/FmmPlanTest.cpp`

This test-only/transition component flattens tree boxes, body permutations,
parent/child ranges, and sorted near/far lists.  Compare deterministic hashes
and pair multisets with the released FMM.  The production CUDA library remains
free of CPU FMM linkage; bridge linkage is confined to a named transition/test
target.

### Task 12: Implement GPU evaluation of a fixed FMM plan

**Files:**

- Create: `cuda/core/magnetic/FmmOperators.cuh`
- Create: `cuda/core/magnetic/FmmEvaluate.h`
- Create: `cuda/core/magnetic/FmmEvaluate.cu`
- Create: `cuda/tests/runtime/FmmOperatorTest.cu`
- Create: `cuda/tests/differential/FmmEvaluateTest.cpp`

Implement and independently test S2M, M2M, P2P, M2L, L2L, and L2T for the
released P=6 Laplace basis.  Keep geometry, charges, coefficients, results,
iterations, and pressure resident.  Nsight must show zero H2D/D2H and zero
allocation inside the 11 evaluations.

### Task 13: Build the adaptive FMM plan on GPU

**Files:**

- Create: `cuda/core/magnetic/Morton.cuh`
- Create: `cuda/core/magnetic/FmmPlanBuilder.h`
- Create: `cuda/core/magnetic/FmmPlanBuilder.cu`
- Create: `cuda/tests/runtime/FmmPlanBuilderTest.cu`
- Create: `cuda/tests/differential/FmmPlanBuilderDifferentialTest.cpp`

Implement bbox reduction, stable Morton/original-ID radix sort, level scans,
adaptive box emission, deterministic dual traversal, and near/far CSR lists.
Validate topology/list hashes and pair multisets including duplicate and
degenerate inputs.  Record boxes, list entries, frontier peak, capacity growth,
and peak memory.

### Task 14: Couple IoB and run the small magnetic plane

**Files:**

- Create: `cuda/core/magnetic/IoBSolver.h`
- Create: `cuda/core/magnetic/IoBSolver.cu`
- Modify: `cuda/core/sim/Simulation.cu`
- Create: `cuda/tests/smoke/PlaneMagneticTest.cpp`

Execute exactly ten fixed-point updates plus the final evaluation, compute the
released magnetic pressure, couple it through the pressure jump, and compare
fixed-dt phase snapshots and surface metrics at scales 16 and 32.  Direct IoB
is available only under an explicit test target and size gate.

---

## Project M3: Paper scene validation and optimization

### Task 15: Add run manifests, strict audits, and performance instrumentation

**Files:**

- Create: `cuda/core/runtime/NvtxRanges.h`
- Create: `scripts/audit_cuda_separation.sh`
- Create: `scripts/profile_cuda_plane.sh`
- Create: `scripts/collect_run_manifest.py`
- Create: `docs/compliance/cuda-equation-map.md`
- Create: `NUMERICAL_DEVIATIONS.md`

Audit hidden CPU physics linkage, forbidden dense interaction tensors,
guards/fallbacks, per-step transfers, allocations, and nondeterministic
atomics.  Record phase latency, end-to-end latency, peak memory, boxes/lists,
solver iterations/residuals, hardware/toolkit, commit/diff, and artifact hashes.

### Task 16: Run differential, refinement, and scale-192 acceptance

**Files:**

- Create: `scripts/run_cuda_validation_campaign.py`
- Create: `scripts/analyze_cuda_validation.py`
- Create: `configs/plane/cuda_paper.yaml`
- Create: `configs/plane/cuda_refinement.yaml`

Run at least four spatial and four timestep levels plus a crossed fine run.
Fix CUDA tolerances from accepted CPU/reference uncertainty before inspecting
the CUDA paper output.  Then run 401 frames at scale 192.  Report volume drift,
NaN/Inf, substeps versus 2318, max mesh size versus 249.7k, onset, peak
height/count, dominant wavenumber, surface-spectrum error, phase timings, peak
memory, and speedup.  Only this evidence can promote the output to a CUDA
paper-scene numerical-reference candidate.

### Task 17: Final verification and integration decision

Run fresh:

```bash
xmake build -P . cuda_host_tests demo_cuda cuda_runtime_tests
xmake run -P . cuda_host_tests
xmake run -P . cuda_runtime_tests
compute-sanitizer --tool memcheck build/linux/x86_64/release/cuda_runtime_tests
python -m pytest -q
bash scripts/audit_cuda_separation.sh
git status --short
```

Review every diff, report pass/fail/skip counts and exit codes, list unverified
items and all numerical deviations, and request code review before merging.
