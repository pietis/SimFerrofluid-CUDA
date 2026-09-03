# Uniform-Field Plane Hero Scene Reproduction Design

## 1. Work mode and scope

This work is `REPRODUCTION`.

The deliverable is a reproducible, measured copy of the paper's **Uniform
field** normal-field-instability scene, named `plane` in the official 3D code.
It must preserve the official CPU/IoB physics and numerics before any CUDA
port, parameter variation, or passive-marker export is introduced.

This design intentionally excludes:

- CUDA kernels or changes to arithmetic order;
- parameter sweeps or randomized training trajectories;
- passive markers and WorldParticle-format export;
- the `dipole`, `magsphere`, `lifting`, and `pattern*` scenes;
- claims of agreement with a laboratory ferrofluid experiment.

Until the gates in this document pass, outputs are `GT CANDIDATE`. After they
pass they may be called **paper-model numerical-reference GT**, not experimental
ground truth.

## 2. Authority and references

Authority order for this reproduction is:

1. Ni et al., *An Induce-on-Boundary Magnetostatic Solver for Grid-Based
   Ferrofluids*, ACM TOG 43(4), 2024, DOI
   [10.1145/3658124](https://doi.org/10.1145/3658124), especially Fig. 5,
   Section 6.2.1, and Table 3.
2. The authors' official 3D repository,
   [ferrofluid-simulation/SimFerrofluid](https://github.com/ferrofluid-simulation/SimFerrofluid),
   represented here by upstream commit
   `05a23a1229482c9d098cf032760cb1b895c8bca8`.
3. The authors' project page and supplementary video for qualitative appearance.

Paper values take precedence. Where the paper is silent, the official code at
the pinned commit supplies the value. Such values must be labelled
`code-specified`, rather than attributed to the paper.

## 3. Paper-to-code compliance table

| Item | Source | Implementation | Status | Deviation or unknown | Validation |
|---|---|---|---|---|---|
| Uniform field creates spikes parallel to the field | Paper Fig. 5 and Section 6.2.1 | `demo/SimBuilder.cpp`, `SimBuilder::BuildPlane` | equivalent | Paper calls the case `Uniform`; CLI calls it `plane` | final height-map and render checks |
| Resolution `192 x 144 x 192` | Paper Table 3 | `BuildPlane`, `scale=192`, resolution `(4,3,4)*scale/4` | exact | none | scene-inspection test |
| Frame interval `2 ms`, frame count `401` | Paper Table 3 | CLI `--rate=500 --end=401`; frames `0..400` | exact | final physical time is `0.8 s`; video playback is separately 30 FPS | output manifest test |
| CFL `0.5` | Paper Table 3 | CLI `--cfl=0.5` | exact | none | resolved-config and timestep log |
| Grid spacing `0.638 mm` | Paper Table 3 | `0.12/(192-4) = 0.638297872... mm` | exact up to table rounding | none | scene-inspection test |
| Susceptibility `chi=0.33` | Paper Table 3 | `Magnetic::m_Chi` default | exact | implicit default, not assigned in `BuildPlane` | resolved-config test |
| Damping `eta=8 s^-1` | Paper Table 3 and Section 5.2 | `BuildPlane`, `m_Damping=8`; applied as `exp(-eta*dt)` | exact | the paper presents this as an artificial-viscosity substitute, not a physical viscosity coefficient | resolved-config test |
| Gravity, surface tension, density | Paper Table 3 footnote | `Simulation` defaults and `ApplyBodyForces` | exact | code uses `-9.8 m/s^2` on `y`, `0.0728 N/m`, `1000 kg/m^3` | resolved-config test |
| Applied field is uniform and vertical | Paper Fig. 5 and Section 6.2.1 | `Hext=(0,60000,0)` | exact in form; code-specified in magnitude | Paper does not publish the magnitude or turn-on protocol in Fig. 5/Table 3 | field sampling test |
| Initially stable liquid in a tank | Paper Section 6.2.1 | zero velocity and planar level set in a closed box collider | equivalent; geometry code-specified | Paper does not publish tank dimensions or fill depth | geometry test and unforced short run |
| Initial simulation time | Physical initialization and the code's field callback imply `t=0` | `Simulation::m_Time` is uninitialized when `Initialize()` first calls the magnetic solver | missing / bug | reading the indeterminate value is undefined behavior even though `plane` currently ignores its `time` argument | sanitizer/contract test must fail first; initialize to exactly zero before golden run |
| IoB magnetostatic coupling | Paper Sections 4 and 5 | `--mag-solver=iob`; surface mesh to magnetic pressure jump | exact model path | none | magnetic-on stage test |
| IoB termination | Paper Algorithm 1 says iterate until its tolerance criterion | released code runs exactly 10 iterations; the declared `m_StopThres=1e-6` is unused | approximation | the Uniform row does not publish the run's tolerance or iteration policy, so exact published-run termination is unknown | preserve released behavior; record residual proxy if added without changing termination |
| Reported step count `2318` and max surface point count `249.7k` | Paper Table 3 | adaptive CFL substeps and marching-cubes surface points | to be measured | compiler and reduction ordering may change the adaptive trajectory | report relative discrepancy; not a silent pass/fail substitution |
| Explicit surface tension and level-set volume behavior | Paper limitation discussion | current `Simulation` path | exact to released code | Paper reports minor jitter and volume-loss limitations | retain and measure; do not add guards or implicit tension |

## 4. Scene definition

### 4.1 Purpose and outputs

- Intended phenomenon: normal-field instability of a ferrofluid layer under a
  uniform vertical magnetic field.
- Why needed: establish the spike-preserving CPU reference that every later
  CUDA stage must match.
- Final outputs: resolved configuration and provenance; initial-state views;
  magnetic-off short-run diagnostics; all 401 magnetic-on surface frames;
  checkpoints; spike metrics; representative stills and a 30 FPS review video.

### 4.2 World coordinates

- Units: metre, second, kilogram. The code does not annotate the magnetic-field
  unit; this reproduction interprets its `H` values as A/m under the paper's SI
  magnetostatic formulation and records that interpretation in provenance.
- Grid center/world origin: `(0,0,0)`.
- Up: `+y`.
- Gravity: `(0,-9.8,0) m/s^2`.
- Camera: not part of the physical scene and not specified by the paper or
  solver. Initial inspection uses fixed front (`-z`), side (`-x`), and top
  (`-y`) orthographic views with a visible axis triad. The final review camera
  is recorded separately and never changes solver state.

At the paper resolution:

- cell resolution: `(192,144,192)`;
- boundary width: 2 cells;
- spacing: `0.000638297872340425 m`;
- allocated-grid origin: `(-0.0612765957446808,
  -0.0459574468085106,-0.0612765957446808) m`;
- interior domain bounds: `(-0.06,-0.0446808510638298,-0.06) m` to
  `(0.06,0.0446808510638298,0.06) m`.

### 4.3 Geometry

| Name | Shape and dimensions | Center or bounds | Fixed or moving | Solid faces | Open faces |
|---|---|---|---|---|---|
| Computational tank | Axis-aligned box, interior `0.12 x 0.0893617021 x 0.12 m` | bounds above | fixed | all six collider faces | none |
| Initial ferrofluid | Region below a horizontal plane; analytic pre-discretization volume `0.0003456 m^3` | surface at `y=-0.0206808510638298 m`; fill depth `0.024 m` | evolving free surface | tank contact faces | liquid-air interface is free, not an opening in the collider |

There is one connected liquid region. Collider and liquid clipping use the same
`StaggeredGrid` domain box. No visual-only wall is authoritative.

### 4.4 State and external input

| State | Initial region | Resolution | Initial value | Evolution |
|---|---|---|---|---|
| Liquid level set | tank cell grid | `192 x 144 x 192` cells | signed distance to the horizontal plane, clipped to the tank | RK2 semi-Lagrangian advection plus the released reinitialization path |
| Velocity | staggered face grids | corresponding MAC faces | exactly zero | gravity, pressure projection, damping, magnetic and capillary pressure jumps |

- Magnetic-on input: constant `H_app(x,t)=(0,60000,0)` for every queried
  position and simulation time, as implemented by the released code.
- Unforced short run: disable only magnetics with `--mag=false`; retain gravity,
  surface tension, damping, collision, advection, reinitialization, volume
  control, and pressure projection.
- Random seed: none. The exact reproduction must not add a perturbation.

The lack of an explicit perturbation means spike selection is initiated by the
released discretization and floating-point execution. Controlled perturbations
belong to the later dataset-variation design, not this reproduction.

## 5. Execution stages

### Stage A: build and static inspection

Build the unmodified CPU release target and record its baseline result. A
sanitizer/contract test must first expose the uninitialized initial simulation
time. Then apply only the test-driven semantic correction
`m_Time = 0`; record it in the compliance table and provenance, and emit a
machine-readable resolved scene record. Verify all values in Section 4 from
live objects, not by grepping constants. Produce front, side, and top
initial-state views from the same contour consumed by the solver.

### Stage B: magnetic-off short run

Run a small deterministic smoke configuration first, then the paper resolution
for `0.02 s` with magnetics disabled. This checks the tank, initial equilibrium,
surface tension, gravity, pressure projection, and level-set path before the
requested magnetic effect is introduced.

### Stage C: magnetic-on paper run

Run exactly:

```bash
xmake r demo \
  --dirname runs/reproduction/plane_iob_cpu \
  --test plane \
  --end 401 \
  --rate 500 \
  --cfl 0.5 \
  --scale 192 \
  --mag=true \
  --mag-solver=iob \
  --stride 50
```

This follows the argument form in the official README. No parameter may be
changed to improve appearance.

### Stage D: render and measure

Render the initial frame, onset interval, representative developed-spike frame,
and final frame. The metrics script consumes the saved contour meshes, not a
separate visual approximation.

## 6. Predeclared acceptance checks

Raw values accompany every verdict. A failed earlier stage stops the run from
being promoted to the next stage.

| Check | Quantity | Pass condition | Evidence |
|---|---|---|---|
| Static geometry | resolution, spacing, domain bounds, plane height, analytic volume, initial velocity and time | exact integer equality; scalar absolute error `<=1e-12 m`; analytic volume exactly `0.0003456 m^3` within double arithmetic; max initial speed and initial time exactly zero | `resolved_scene.yaml`, geometry report |
| Applied field | samples at all 8 interior corners, center, and 16 deterministic points at `t={0,0.4,0.8}s` | every sample exactly `(0,60000,0)` | field report |
| Output cadence | output indices and physical times | exactly 401 frames, indices `0..400`, spacing `0.002 s`, final time `0.8 s` | manifest |
| Magnetic-off short run | NaN/Inf, volume, surface excursion, containment | zero NaN/Inf; no surface vertex farther than `1e-12 m` outside the tank; relative volume drift `<=0.5%`; max surface excursion from the initial plane `<=2*dx` at `0.02 s` | short-run metrics and three views |
| Magnetic effect | vertical surface growth relative to magnetic-off control | by `0.8 s`, magnetic-on maximum height rise is at least `4*dx` and exceeds magnetic-off by at least `3*dx` | height-map time series |
| Spike formation | distinct local maxima in a fixed horizontal height map | at least one peak with prominence `>=2*dx`; report count, positions, heights, and dominant horizontal wavenumber without tuning detector thresholds after viewing | spike metrics |
| Direction | spike displacement relative to field | developed peaks have positive `y` prominence; no claim of exact angular agreement beyond this axis-aligned setup | spike metrics |
| Paper statistics | total substep count and maximum surface point count | report discrepancy from `2318` and `249.7k`; a discrepancy over 5% blocks automatic equivalence and requires investigation | run summary |
| Final visual review | initial, onset, developed, and final surface | no clipping, inverted axes, tank penetration, missing surface, or render/solver mismatch in front, side, and top views | PNGs and review video |

These thresholds establish successful construction of a spike-forming copy of
the released paper scene. They do not by themselves establish grid/time
convergence. Before using the trajectory as GT, a separate refinement plan must
show that spike onset, peak height, count, and dominant wavenumber are stable at
the chosen production resolution and timestep.

## 7. Provenance contract

Every run records:

- run ID and UTC timestamp;
- paper DOI and upstream commit;
- current commit, dirty flag, and diff hash or patch artifact;
- recursive submodule commits;
- exact command and resolved scene values;
- compiler, xmake, Eigen, AMGCL, TBB, OpenMP, and OS versions;
- CPU model and thread counts; later CUDA runs also record GPU, driver, CUDA,
  architecture flags, and precision policy;
- frame/substep counts, exit code, NaN/Inf count, solver residual information,
  volume drift, and maximum mesh size;
- hashes and paths for raw checkpoints, contours, metrics, and renders.

Exploratory runs from a dirty tree are labelled `PRELIMINARY` and cannot become
the CPU golden reference.

## 8. CUDA boundary

The accepted CPU run becomes immutable comparison evidence. CUDA work proceeds
stage by stage and compares saved double-precision level set, staggered
velocity, contour topology/geometry, magnetic pressure, volume state, adaptive
substep count, and spike metrics. Bitwise equality is not required for
floating-point reductions or FMM accumulation order; tolerances must be fixed
from CPU refinement uncertainty before viewing CUDA results.

No CUDA result may replace the CPU golden reference until the complete coupled
pipeline passes these gates.

## 9. Assumptions requiring approval

- The initial reproduction uses the current official code's constant field
  magnitude `60000`, tank dimensions, fill depth, ten fixed IoB iterations,
  and instantaneous field activation. The paper does not publish those run
  values separately, and its Algorithm 1 describes tolerance-based rather than
  fixed-count termination; this reproduction preserves and reports that
  released-code approximation instead of silently changing it.
- The first implementation change initializes the currently indeterminate
  `Simulation::m_Time` to exactly zero. This is a narrowly scoped semantic bug
  fix required for reproducible initialization, not a physics or stability
  guard. The original sanitizer failure and corrected test remain in evidence.
- Quantitative agreement is initially against the paper-model implementation
  and Table 3, not a laboratory measurement.
- Dataset variation and passive-marker generation begin only after this fixed
  scene passes and are delivered in separate commits and run directories.
