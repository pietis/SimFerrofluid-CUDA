#pragma once

#include "../runtime/CudaContext.h"
#include "../scene/PlaneScene.h"
#include "State.h"

#include <array>
#include <cstddef>

namespace Pivot::Cuda {

struct SimulationStateMetadata {
    double time;
    std::size_t cell_phi_count;
    std::array<std::size_t, 3> velocity_counts;
};

class Simulation {
public:
    explicit Simulation(PlaneScene scene, int device_ordinal = 0);

    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    const PlaneScene& scene() const noexcept;
    const State& state() const noexcept;
    SimulationStateMetadata state_metadata() const noexcept;
    int device_ordinal() const noexcept;

private:
    PlaneScene scene_;
    CudaContext context_;
    State state_;
};

}  // namespace Pivot::Cuda
