#pragma once

#include "../runtime/DeviceBuffer.h"

#include <array>

namespace Pivot::Cuda {

struct State {
    double time = 0.0;
    DeviceBuffer<double> phi;
    std::array<DeviceBuffer<double>, 3> velocity;
};

}  // namespace Pivot::Cuda
