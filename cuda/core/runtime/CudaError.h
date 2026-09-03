#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>

namespace Pivot::Cuda {

class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const char* expression, const char* file, int line);

    cudaError_t code() const noexcept;

private:
    cudaError_t code_;
};

void CheckCuda(cudaError_t code, const char* expression, const char* file, int line);

}  // namespace Pivot::Cuda

#define PIVOT_CUDA_CHECK(expression) \
    ::Pivot::Cuda::CheckCuda((expression), #expression, __FILE__, __LINE__)
