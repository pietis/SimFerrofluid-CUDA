#pragma once

#include <cstddef>

namespace Pivot::Cuda {

struct Int3 {
    int x;
    int y;
    int z;
};

struct Double3 {
    double x;
    double y;
    double z;
};

struct GridDesc {
    double spacing;
    double inverse_spacing;
    Int3 size;
    Double3 origin;

    std::size_t VertexCount() const;
    std::size_t IndexOf(Int3 coord) const;
    Int3 CoordOf(std::size_t index) const;
    Double3 PositionOf(Int3 coord) const;
};

}  // namespace Pivot::Cuda
