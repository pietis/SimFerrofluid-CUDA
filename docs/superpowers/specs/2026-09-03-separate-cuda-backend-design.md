# Separate CUDA Backend Design

## 1. Mode, goal, and non-goals

This work is `RESEARCH`.

The goal is a CUDA implementation of the paper's `plane` scene that is a
separate program and source tree from the released CPU implementation.  The
CPU `core/`, `demo/`, and `fmmtl/` sources remain an immutable comparison
baseline on `main`; CUDA development occurs on `codex/cuda-plane` under
`cuda/` and produces a distinct `demo_cuda` binary.

The end state is a GPU-resident coupled solver, not a launcher that copies
arrays to the GPU for one operation and silently returns to the CPU.  Host code
may schedule work, resolve the frame/CFL rule, write files, and build early FMM
plans while the GPU plan builder is being validated.  It may not execute a
hidden CPU physics fallback.

Initially supported:

- scene: `plane` only;
- magnetic solver: IoB only after its CUDA stages land;
- precision: IEEE binary64 for physical state and operators;
- CUDA toolkit: 12.8 or newer, to cover the intended Blackwell target;
- Linux x86-64 NVIDIA execution; macOS remains a host-contract test platform.

Unsupported scenes or solvers fail explicitly.  They are never redirected to
the CPU executable.

## 2. Source and build boundary

The only top-level build change is an additive `includes("cuda")`.  Production
CUDA targets do not compile or link files from `core/`, `demo/`, `fmmtl/`, the
vendored legacy `thrust/`, or the viewer.

```text
cuda/
  xmake.lua
  core/
    runtime/       CUDA context, errors, persistent buffers and arenas
    grid/          POD grid descriptions and device field views
    scene/         plane configuration and initialization
    sim/           device-owned state and exact phase scheduler
    operators/     interpolation, advection, reinitialization, collider
    surface/       marching cubes, level-set geometry and volume
    pressure/      CSR assembly, BiCGStab and GPU preconditioner
    magnetic/      direct oracle, FMM operators and plan builder
    io/            checkpoints, exports and provenance
  demo/            independent CLI, driver and scene builder
  tests/
    host/          no-CUDA contracts; runnable on macOS and Linux
    runtime/       small NVIDIA kernel and ownership tests
    differential/ CPU-reference comparisons, never linked into demo_cuda
    smoke/         fixed-dt and adaptive end-to-end runs
```

Targets:

| Target | Toolchain | Purpose |
|---|---|---|
| `cuda_host_tests` | ordinary C++20 | exact scene/grid/index contracts on any host |
| `sim_cuda` | CUDA C++20 | standalone CUDA solver library |
| `demo_cuda` | CUDA C++20 | standalone CUDA plane executable |
| `cuda_runtime_tests` | CUDA C++20 | device tests requiring an NVIDIA GPU |
| `cuda_compile_tests` | CUDA C++20 | compile/link checks that do not claim runtime correctness |

`sim_cuda` links CUDA Runtime, cuSPARSE, cuBLAS, and toolkit CCCL/CUB.  The
pressure and FMM modules must not depend on the CPU library.  A differential
test executable may link both implementations only in its test target.

## 3. State ownership and layouts

`Pivot::Cuda::Simulation` uniquely owns one `CudaContext` and every device
allocation.  One nonblocking physics stream is used initially to fix ordering.
cuSPARSE/cuBLAS handles and the stream-ordered memory pool belong to that
context.  CUB temporary-storage queries happen during capacity planning and
their workspaces persist; no allocation occurs inside an iteration.

Cross-step physical state, all in binary64:

- cell level set `phi[nx*ny*nz]`, z-fast as in CPU `Grid::IndexOf`;
- MAC velocities `u[(nx+1)*ny*nz]`, `v[nx*(ny+1)*nz]`, and
  `w[nx*ny*(nz+1)]`;
- `time`, initial/current volume, cumulative/instantaneous volume error, and
  maximum mesh size.

Static resident state:

- trivially-copyable cell/node/face grid descriptors;
- collider auxiliary level set, face fractions, normals, and zero wall
  velocity;
- immutable marching-cubes lookup tables.

Transient storage is persistent in capacity but reset or rebuilt in content:
advection/reinitialization ping-pong fields, surface mesh, pressure mapping and
CSR arrays, Krylov/MG vectors, and IoB/FMM tree/list/coefficient arrays.  A
capacity overflow either grows at a declared synchronized boundary and records
the event, or fails.  It never truncates geometry or reduces resolution.

Host storage is limited to configuration, scalar scheduling decisions,
provenance, and pinned frame/checkpoint staging.  Bulk fields are copied only
at explicit export/checkpoint boundaries.

## 4. Exact plane contract

The CUDA scene reproduces the released plane values without including its CPU
builder:

| Quantity | CUDA contract |
|---|---|
| scale | default `64`; paper run `192` |
| resolution | `(scale, 3*scale/4, scale)` |
| boundary width | `2` cells |
| spacing | `0.12 / (scale - 4)` m |
| domain center | `(0,0,0)` |
| liquid plane | interior minimum y plus `0.024` m |
| initial velocity/time | exactly zero |
| density / surface tension | `1000 kg/m^3` / `0.0728 N/m` |
| gravity / damping | `(0,-9.8,0) m/s^2` / `8 s^-1` |
| applied field | `(0,60000,0)` |
| susceptibility | `0.33`; `lambda=-chi/(2+chi)` |
| IoB regularization / iterations | `epsilon=dx`; exactly 10 fixed-point iterations |

The CPU program currently reaches its initial magnetic solve with an
uninitialized `Simulation::m_Time`.  The CUDA state initializes time to exact
zero.  Differential harnesses must call `SetTime(0)` before CPU `Initialize()`;
the unmodified CPU CLI's initial magnetic result is not a clean oracle.

## 5. Substep order

The CUDA scheduler preserves the released causal order:

1. reduce maximum absolute MAC velocity;
2. apply the existing frame/CFL and final-half-step rule on the host;
3. damp all velocity components by `exp(-8*dt)`;
4. RK2 backtrace and tricubic level-set advection;
5. RK2 backtrace and trilinear three-component MAC velocity advection, swapping
   buffers only after every component completes;
6. level-set collider extrapolation for one layer and collider difference;
7. ten SSPRK/WENO reinitialization iterations;
8. marching cubes, level-set normals/curvature, and MC volume reduction;
9. PI volume-error update;
10. gravity;
11. IoB magnetic pressure: 10 fixed-point evaluations plus the final field
    evaluation;
12. pressure unknown compaction, CSR/RHS assembly, solve, and projection;
13. six layers of velocity extrapolation;
14. collider enforcement.

No `--use-cpu`, implicit fallback, automatic resolution reduction, fast-math,
NaN replacement, or new numerical clamp is permitted.  The port preserves and
labels the released `.001` free-surface fraction floor, WENO epsilon, collider
fraction cutoff, and FMM regularization.  Any later change follows the
numerical-deviation contract and has an exact off/control path.

## 6. Pressure strategy

Pressure rows and their canonical z-fast IDs are generated on the GPU using an
integer mask and exclusive scan.  Assembly produces a sorted CSR row with the
same six-neighbor weights, interface term, pressure jump, and volume-error RHS
as CPU `Pressure.cpp`.

The first production solver is zero-initialized binary64 BiCGStab using
deterministic `CUSPARSE_SPMV_CSR_ALG2`, cuBLAS vector operations, and a
mask-aware geometric multigrid preconditioner.  Because CPU AMGCL uses smoothed
aggregation with SPAI0, this is an algorithmic approximation even when the
matrix is exact.  It must pass matrix/RHS, residual, divergence, and scene-level
differential gates and remains labelled `approximation` until then.  cuDSS is
excluded because its current API status is Preview.

## 7. IoB/FMM strategy

The magnetic path is introduced in three separately testable levels:

1. A binary64 CUDA direct-sum implementation for small meshes only.  It is the
   device oracle and must never run accidentally at production scale.
2. A CPU-plan/GPU-evaluate bridge: the released tree and interaction topology
   are flattened once per surface into POD/CSR; S2M, M2M, P2P, M2L, L2L, L2T,
   all 10 fixed-point updates, the final evaluation, and pressure calculation
   stay on device with no transfer between iterations.
3. A full GPU plan builder: bounding-box reduction, stable Morton/original-ID
   sort, adaptive octree construction, deterministic dual traversal, and
   near/far CSR compaction.

Direct sum is not a production fallback: 249,700 points imply about 62.35
billion pairs per matrix-vector evaluation and about 685.85 billion over the
11 evaluations.  The production backend therefore fails if GPU FMM is absent.

Science/reproduction mode uses stable `(Morton key, original index)` ordering,
integer scans for topology, fixed child/source accumulation order, and no
floating-point atomic accumulation.  Cross-CPU bitwise equality is not
required, while repeat runs on one fixed GPU/toolkit must be bitwise stable.

## 8. Milestones and truth labels

| Milestone | Capability | Allowed claim |
|---|---|---|
| B0 | independent build, host contracts, device allocation and plane initialization | CUDA backend foundation; not a solver |
| M0 | complete nonmagnetic scale-16/32 fixed/adaptive pipeline, export and restart | nonmagnetic CUDA solver candidate |
| M1 | CUDA direct IoB on small extracted surfaces | magnetic operator oracle; not production |
| M2 | production GPU FMM and coupled small plane | coupled CUDA solver candidate |
| M3 | scale-192, 401-frame run with convergence/performance evidence | CUDA paper-scene numerical reference candidate |

A compile-only CI job proves syntax and linkage, not GPU execution.  Until M3
passes the physical and differential gates, rendered spikes are diagnostic and
must not be called GT.

## 9. Verification gates

- exact grid sizes, z-fast indices, origins, spacing, plane height, zero time,
  zero velocity, field samples, and scalar constants;
- exact integer masks/mappings and canonicalized topology for all 256 marching
  cubes cases;
- phase-level CPU/CUDA differential snapshots at fixed `dt` for interpolation,
  advection, reinitialization, geometry, matrix/RHS, projection, and IoB;
- pressure matrix canonical equality plus final normalized residual and
  projected-divergence acceptance;
- CUDA direct versus an independent CPU direct magnetic oracle, then FMM versus
  direct at several small `N`;
- zero-field/static-state, volume drift, containment, translation consistency
  where supported, and permutation invariance;
- checkpoint restart identity and same-GPU repeat-run determinism;
- seeded sign, scale, omitted-gravity, and omitted-magnetic-force mutations
  must turn the relevant tests red;
- Nsight evidence that steady-state operator/IoB iterations contain no hidden
  H2D/D2H or allocation, plus five-point `N` and resolution scaling.

Numerical tolerances are fixed from CPU/reference refinement uncertainty before
the CUDA outputs are inspected.  A plausible render is never a pass condition.

## 10. Baseline evidence and current limitation

On 2026-09-03, the unmodified CPU checkout was configured with Xmake 3.1.1 on
Apple M5 Pro/macOS arm64.  `xmake build -P . -y demo` failed in the released
FMMTL because its forward declarations of `std::tuple_size` and
`std::tuple_element` conflict with current libc++.  This is retained as a
baseline compatibility failure rather than patched as part of the CUDA branch.

This machine has no `nvcc` or NVIDIA GPU.  Host-contract tests run locally;
CUDA compile/link is checked in a CUDA 12.8 Linux CI container, and numerical
runtime gates require a named NVIDIA runner.  No runtime claim may be inferred
from compile-only CI.

## 11. Compliance map

| Source requirement | Planned implementation | Status | Deviation | Validation |
|---|---|---|---|---|
| Paper Uniform/plane settings | `cuda/core/scene/PlaneScene.*` | exact target | none intended | host/device scene contracts |
| Released substep ordering | `cuda/core/sim/Simulation.*` | exact target | GPU reductions alter roundoff | phase trace and lockstep snapshots |
| Released hydro discretization | `cuda/core/operators/*`, `surface/*` | exact/equivalent target | parallel accumulation order | operator and physical oracles |
| Released pressure matrix | `cuda/core/pressure/PressureAssembly.cu` | exact target | solver/preconditioner differs | CSR/RHS equality, residual/divergence |
| Released IoB, P=6 FMM | `cuda/core/magnetic/*` | equivalent target | parallel FMM ordering | direct/operator/iteration parity |
| CPU implementation remains available | existing `demo` on `main`; distinct `demo_cuda` | exact | CPU baseline build incompatibility remains documented | source/link audit and CI matrices |

