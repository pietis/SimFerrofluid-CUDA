#pragma once

#include "CudaError.h"

#include <cstddef>
#include <cstdio>
#include <limits>
#include <utility>

namespace Pivot::Cuda {

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;

    DeviceBuffer(const std::size_t count, const cudaStream_t stream) {
        Allocate(count, stream);
    }

    ~DeviceBuffer() noexcept {
        ReleaseNoThrow();
    }

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0U)),
          capacity_(std::exchange(other.capacity_, 0U)),
          stream_(std::exchange(other.stream_, nullptr)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            ReleaseNoThrow();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0U);
            capacity_ = std::exchange(other.capacity_, 0U);
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void Allocate(const std::size_t count, const cudaStream_t stream) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("device buffer byte count overflows std::size_t");
        }

        Release();
        if (count == 0U) {
            return;
        }

        T* allocation = nullptr;
        PIVOT_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&allocation), count * sizeof(T),
                                         stream));
        data_ = allocation;
        size_ = count;
        capacity_ = count;
        stream_ = stream;
    }

    void Release() {
        if (data_ == nullptr) {
            size_ = 0U;
            capacity_ = 0U;
            stream_ = nullptr;
            return;
        }

        PIVOT_CUDA_CHECK(cudaFreeAsync(data_, stream_));
        data_ = nullptr;
        size_ = 0U;
        capacity_ = 0U;
        stream_ = nullptr;
    }

    T* data() noexcept {
        return data_;
    }

    const T* data() const noexcept {
        return data_;
    }

    std::size_t size() const noexcept {
        return size_;
    }

    std::size_t capacity() const noexcept {
        return capacity_;
    }

private:
    void ReleaseNoThrow() noexcept {
        if (data_ == nullptr) {
            return;
        }

        const cudaError_t error = cudaFreeAsync(data_, stream_);
        if (error != cudaSuccess) {
            std::fprintf(stderr, "DeviceBuffer cleanup failed: %s\n", cudaGetErrorString(error));
            return;
        }
        data_ = nullptr;
        size_ = 0U;
        capacity_ = 0U;
        stream_ = nullptr;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0U;
    std::size_t capacity_ = 0U;
    cudaStream_t stream_ = nullptr;
};

}  // namespace Pivot::Cuda
