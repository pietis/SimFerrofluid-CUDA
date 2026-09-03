#include "CudaContext.h"

#include <cstdio>
#include <string>
#include <utility>

namespace Pivot::Cuda {
namespace {

std::string MakeCudaErrorMessage(const cudaError_t code, const char* expression, const char* file,
                                 const int line) {
    return std::string(file) + ":" + std::to_string(line) + ": " + expression + " failed: " +
           cudaGetErrorName(code) + " (" + cudaGetErrorString(code) + ")";
}

void ReportCleanupError(const char* operation, const cudaError_t code) noexcept {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CudaContext cleanup %s failed: %s (%s)\n", operation,
                     cudaGetErrorName(code), cudaGetErrorString(code));
    }
}

void DestroyStreamNoThrow(cudaStream_t& stream) noexcept {
    if (stream == nullptr) {
        return;
    }
    ReportCleanupError("cudaStreamSynchronize", cudaStreamSynchronize(stream));
    ReportCleanupError("cudaStreamDestroy", cudaStreamDestroy(stream));
    stream = nullptr;
}

}  // namespace

CudaError::CudaError(const cudaError_t code, const char* expression, const char* file, const int line)
    : std::runtime_error(MakeCudaErrorMessage(code, expression, file, line)), code_(code) {}

cudaError_t CudaError::code() const noexcept {
    return code_;
}

void CheckCuda(const cudaError_t code, const char* expression, const char* file, const int line) {
    if (code != cudaSuccess) {
        throw CudaError(code, expression, file, line);
    }
}

CudaContext::CudaContext(const int device_ordinal) : device_ordinal_(device_ordinal) {
    try {
        PIVOT_CUDA_CHECK(cudaSetDevice(device_ordinal_));
        PIVOT_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        PIVOT_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memory_pool_, device_ordinal_));
    } catch (...) {
        DestroyStreamNoThrow(stream_);
        memory_pool_ = nullptr;
        throw;
    }
}

CudaContext::CudaContext(CudaContext&& other) noexcept
    : device_ordinal_(std::exchange(other.device_ordinal_, -1)),
      stream_(std::exchange(other.stream_, nullptr)),
      memory_pool_(std::exchange(other.memory_pool_, nullptr)) {}

CudaContext& CudaContext::operator=(CudaContext&& other) noexcept {
    if (this != &other) {
        ReleaseNoThrow();
        device_ordinal_ = std::exchange(other.device_ordinal_, -1);
        stream_ = std::exchange(other.stream_, nullptr);
        memory_pool_ = std::exchange(other.memory_pool_, nullptr);
    }
    return *this;
}

CudaContext::~CudaContext() noexcept {
    ReleaseNoThrow();
}

void CudaContext::ReleaseNoThrow() noexcept {
    if (stream_ != nullptr) {
        ReportCleanupError("cudaSetDevice", cudaSetDevice(device_ordinal_));
        DestroyStreamNoThrow(stream_);
    }
    memory_pool_ = nullptr;
    device_ordinal_ = -1;
}

cudaStream_t CudaContext::stream() const noexcept {
    return stream_;
}

int CudaContext::device_ordinal() const noexcept {
    return device_ordinal_;
}

cudaMemPool_t CudaContext::memory_pool() const noexcept {
    return memory_pool_;
}

void CudaContext::Synchronize() const {
    PIVOT_CUDA_CHECK(cudaSetDevice(device_ordinal_));
    PIVOT_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace Pivot::Cuda
