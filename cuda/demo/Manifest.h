#pragma once

#include "core/scene/PlaneScene.h"

#include <array>
#include <cstddef>
#include <string>

namespace Pivot::Cuda {

struct BootstrapOptions;

struct BootstrapStateMetadata {
    double time;
    std::size_t cell_phi_count;
    std::array<std::size_t, 3> velocity_counts;
};

struct CudaDeviceMetadata {
    bool synthetic;
    int ordinal;
    std::string name;
    int runtime_version;
    int driver_version;
    int compute_major;
    int compute_minor;
};

std::string BuildBootstrapManifestJson(const BootstrapOptions& options, const PlaneScene& scene,
                                       const BootstrapStateMetadata& state,
                                       const CudaDeviceMetadata& device);

}  // namespace Pivot::Cuda
