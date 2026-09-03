#include "PlaneScene.h"

#include <limits>
#include <stdexcept>

namespace Pivot::Cuda {
namespace {

std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error("grid vertex count overflows std::size_t");
    }
    return left * right;
}

std::size_t CheckedDimensionProduct(const Int3 size) {
    return CheckedMultiply(
        CheckedMultiply(static_cast<std::size_t>(size.x),
                        static_cast<std::size_t>(size.y)),
        static_cast<std::size_t>(size.z));
}

GridDesc MakeGrid(const double spacing, const Int3 size, const Double3 origin) {
    return {spacing, 1.0 / spacing, size, origin};
}

}  // namespace

std::size_t GridDesc::VertexCount() const {
    return CheckedDimensionProduct(size);
}

std::size_t GridDesc::IndexOf(const Int3 coord) const {
    if (coord.x < 0 || coord.x >= size.x || coord.y < 0 || coord.y >= size.y ||
        coord.z < 0 || coord.z >= size.z) {
        throw std::out_of_range("grid coordinate is out of range");
    }

    const std::size_t vertex_count = VertexCount();
    (void)vertex_count;
    const std::size_t z_size = static_cast<std::size_t>(size.z);
    const std::size_t yz_size = CheckedMultiply(static_cast<std::size_t>(size.y), z_size);
    return static_cast<std::size_t>(coord.x) * yz_size +
           static_cast<std::size_t>(coord.y) * z_size +
           static_cast<std::size_t>(coord.z);
}

Int3 GridDesc::CoordOf(const std::size_t index) const {
    const std::size_t vertex_count = VertexCount();
    if (index >= vertex_count) {
        throw std::out_of_range("grid index is out of range");
    }

    const std::size_t z_size = static_cast<std::size_t>(size.z);
    const std::size_t yz_size = CheckedMultiply(static_cast<std::size_t>(size.y), z_size);
    return {
        static_cast<int>(index / yz_size),
        static_cast<int>((index / z_size) % static_cast<std::size_t>(size.y)),
        static_cast<int>(index % z_size),
    };
}

Double3 GridDesc::PositionOf(const Int3 coord) const {
    return {
        origin.x + spacing * static_cast<double>(coord.x),
        origin.y + spacing * static_cast<double>(coord.y),
        origin.z + spacing * static_cast<double>(coord.z),
    };
}

PlaneScene MakePlaneScene(const int scale) {
    if (scale <= 4 || scale % 4 != 0) {
        throw std::invalid_argument("scale must be greater than four and divisible by four");
    }

    constexpr int boundary_width = 2;
    const int cell_y_size = (scale / 4) * 3;
    const double spacing = 0.12 / static_cast<double>(scale - 4);
    const Double3 allocated_min{
        -0.5 * static_cast<double>(scale) * spacing,
        -0.5 * static_cast<double>(cell_y_size) * spacing,
        -0.5 * static_cast<double>(scale) * spacing,
    };
    const Double3 allocated_max{
        -allocated_min.x,
        -allocated_min.y,
        -allocated_min.z,
    };
    const Double3 interior_min{
        allocated_min.x + static_cast<double>(boundary_width) * spacing,
        allocated_min.y + static_cast<double>(boundary_width) * spacing,
        allocated_min.z + static_cast<double>(boundary_width) * spacing,
    };
    const Double3 interior_max{
        -interior_min.x,
        -interior_min.y,
        -interior_min.z,
    };
    const Int3 cell_size{scale, cell_y_size, scale};
    constexpr double susceptibility = 0.33;

    return {
        scale,
        boundary_width,
        MakeGrid(spacing, cell_size,
                 {allocated_min.x + spacing / 2.0, allocated_min.y + spacing / 2.0,
                  allocated_min.z + spacing / 2.0}),
        {
            MakeGrid(spacing, {scale + 1, cell_y_size, scale},
                     {allocated_min.x, allocated_min.y + spacing / 2.0,
                      allocated_min.z + spacing / 2.0}),
            MakeGrid(spacing, {scale, cell_y_size + 1, scale},
                     {allocated_min.x + spacing / 2.0, allocated_min.y,
                      allocated_min.z + spacing / 2.0}),
            MakeGrid(spacing, {scale, cell_y_size, scale + 1},
                     {allocated_min.x + spacing / 2.0, allocated_min.y + spacing / 2.0,
                      allocated_min.z}),
        },
        allocated_min,
        allocated_max,
        interior_min,
        interior_max,
        interior_min.y + 0.024,
        0.0,
        {0.0, 0.0, 0.0},
        1000.0,
        0.0728,
        {0.0, -9.8, 0.0},
        8.0,
        {0.0, 60000.0, 0.0},
        susceptibility,
        -susceptibility / (2.0 + susceptibility),
        1.0,
        10,
    };
}

}  // namespace Pivot::Cuda
