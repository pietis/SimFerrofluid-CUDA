#include "Cli.h"
#include "Manifest.h"

#include "core/runtime/CudaError.h"
#include "core/sim/Simulation.h"

#include <cstdio>
#include <iostream>
#include <string>

namespace {

Pivot::Cuda::CudaDeviceMetadata QueryDeviceMetadata(const Pivot::Cuda::Simulation& simulation) {
    int runtime_version = 0;
    int driver_version = 0;
    cudaDeviceProp properties{};
    PIVOT_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    PIVOT_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    PIVOT_CUDA_CHECK(cudaGetDeviceProperties(&properties, simulation.device_ordinal()));
    return {false, simulation.device_ordinal(), properties.name, runtime_version, driver_version,
            properties.major, properties.minor};
}

}  // namespace

int main(const int argc, const char* const* argv) {
    try {
        const Pivot::Cuda::BootstrapOptions options =
            Pivot::Cuda::ParseBootstrapCommandLine(argc, argv);
        const Pivot::Cuda::Simulation simulation(Pivot::Cuda::MakePlaneScene(options.scale));
        const Pivot::Cuda::SimulationStateMetadata state = simulation.state_metadata();
        const Pivot::Cuda::BootstrapStateMetadata manifest_state{
            state.time, state.cell_phi_count, state.velocity_counts};
        std::cout << Pivot::Cuda::BuildBootstrapManifestJson(
                         options, simulation.scene(), manifest_state, QueryDeviceMetadata(simulation))
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "demo_cuda: %s\n", error.what());
        return 2;
    }
}
