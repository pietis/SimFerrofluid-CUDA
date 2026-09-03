#include "PlaneScene.cuh"

#include "../runtime/CudaError.h"

#include <cstddef>

namespace Pivot::Cuda {
namespace {

constexpr unsigned int kThreadsPerBlock = 256U;

__global__ void InitializeLevelSetKernel(const PlaneScene scene, const FieldView<double> level_set) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= level_set.VertexCount()) {
        return;
    }

    const Int3 coord = level_set.CoordOf(index);
    const Double3 position = level_set.PositionOf(coord);
    level_set.data[index] = position.y - scene.liquid_surface_y;
}

}  // namespace

void InitializePlaneAsync(const PlaneScene& scene, const FieldView<double> level_set,
                          const MacFieldView velocity, const cudaStream_t stream) {
    const std::size_t cell_count = level_set.VertexCount();
    if (cell_count != 0U) {
        const unsigned int block_count = static_cast<unsigned int>(
            (cell_count + static_cast<std::size_t>(kThreadsPerBlock) - 1U) / kThreadsPerBlock);
        InitializeLevelSetKernel<<<block_count, kThreadsPerBlock, 0U, stream>>>(scene, level_set);
        PIVOT_CUDA_CHECK(cudaGetLastError());
    }

    for (int axis = 0; axis < 3; ++axis) {
        const std::size_t face_count = velocity.axis[axis].VertexCount();
        if (face_count != 0U) {
            PIVOT_CUDA_CHECK(cudaMemsetAsync(velocity.axis[axis].data, 0,
                                             face_count * sizeof(double), stream));
        }
    }
}

}  // namespace Pivot::Cuda
