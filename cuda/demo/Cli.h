#pragma once

#include <cstdint>
#include <string>

namespace Pivot::Cuda {

struct BootstrapOptions {
    std::string scene;
    int scale = 64;
    std::uint64_t end_frame = 0U;
    double frame_rate = 0.0;
    double cfl = 0.0;
    bool magnetic_enabled = false;
    std::string magnetic_solver = "iob";
    bool dry_run = false;
};

BootstrapOptions ParseBootstrapCommandLine(int argc, const char* const* argv);

}  // namespace Pivot::Cuda
