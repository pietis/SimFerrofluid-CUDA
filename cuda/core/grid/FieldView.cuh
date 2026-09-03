#pragma once

#include "GridDesc.h"

#include <cstddef>

namespace Pivot::Cuda {

template <class T>
struct FieldView {
    T* data;
    GridDesc grid;

    __host__ __device__ std::size_t VertexCount() const noexcept {
        return static_cast<std::size_t>(grid.size.x) * static_cast<std::size_t>(grid.size.y) *
               static_cast<std::size_t>(grid.size.z);
    }

    __host__ __device__ std::size_t IndexOf(const Int3 coord) const noexcept {
        return (static_cast<std::size_t>(coord.x) * static_cast<std::size_t>(grid.size.y) +
                static_cast<std::size_t>(coord.y)) *
                   static_cast<std::size_t>(grid.size.z) +
               static_cast<std::size_t>(coord.z);
    }

    __host__ __device__ Int3 CoordOf(const std::size_t index) const noexcept {
        const std::size_t z_size = static_cast<std::size_t>(grid.size.z);
        const std::size_t yz_size = static_cast<std::size_t>(grid.size.y) * z_size;
        return {
            static_cast<int>(index / yz_size),
            static_cast<int>((index / z_size) % static_cast<std::size_t>(grid.size.y)),
            static_cast<int>(index % z_size),
        };
    }

    __host__ __device__ Double3 PositionOf(const Int3 coord) const noexcept {
        return {
            grid.origin.x + grid.spacing * static_cast<double>(coord.x),
            grid.origin.y + grid.spacing * static_cast<double>(coord.y),
            grid.origin.z + grid.spacing * static_cast<double>(coord.z),
        };
    }
};

struct MacFieldView {
    FieldView<double> axis[3];
};

}  // namespace Pivot::Cuda
