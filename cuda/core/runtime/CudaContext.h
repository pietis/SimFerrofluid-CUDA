#pragma once

#include "CudaError.h"

namespace Pivot::Cuda {

class CudaContext {
public:
    explicit CudaContext(int device_ordinal = 0);
    ~CudaContext() noexcept;

    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;
    CudaContext(CudaContext&& other) noexcept;
    CudaContext& operator=(CudaContext&& other) noexcept;

    cudaStream_t stream() const noexcept;
    int device_ordinal() const noexcept;
    cudaMemPool_t memory_pool() const noexcept;
    void Synchronize() const;

private:
    int device_ordinal_;
    cudaStream_t stream_ = nullptr;
    cudaMemPool_t memory_pool_ = nullptr;

    void ReleaseNoThrow() noexcept;
};

}  // namespace Pivot::Cuda
