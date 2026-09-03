#pragma once

#include "../grid/FieldView.cuh"
#include "PlaneScene.h"

#include <cuda_runtime_api.h>

namespace Pivot::Cuda {

void InitializePlaneAsync(const PlaneScene& scene, FieldView<double> level_set,
                          MacFieldView velocity, cudaStream_t stream);

}  // namespace Pivot::Cuda
