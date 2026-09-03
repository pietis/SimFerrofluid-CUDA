#include "Simulation.h"

#include "../scene/PlaneScene.cuh"

namespace Pivot::Cuda {

Simulation::Simulation(const PlaneScene scene, const int device_ordinal)
    : scene_(scene), context_(device_ordinal) {
    state_.time = 0.0;
    state_.phi.Allocate(scene_.cell_grid.VertexCount(), context_.stream());
    for (int axis = 0; axis < 3; ++axis) {
        state_.velocity[axis].Allocate(scene_.face_grids[axis].VertexCount(), context_.stream());
    }
    InitializePlaneAsync(
        scene_, {state_.phi.data(), scene_.cell_grid},
        {{{state_.velocity[0].data(), scene_.face_grids[0]},
          {state_.velocity[1].data(), scene_.face_grids[1]},
          {state_.velocity[2].data(), scene_.face_grids[2]}}},
        context_.stream());
    context_.Synchronize();
}

const PlaneScene& Simulation::scene() const noexcept {
    return scene_;
}

const State& Simulation::state() const noexcept {
    return state_;
}

SimulationStateMetadata Simulation::state_metadata() const noexcept {
    return {state_.time, state_.phi.size(),
            {state_.velocity[0].size(), state_.velocity[1].size(), state_.velocity[2].size()}};
}

int Simulation::device_ordinal() const noexcept {
    return context_.device_ordinal();
}

}  // namespace Pivot::Cuda
