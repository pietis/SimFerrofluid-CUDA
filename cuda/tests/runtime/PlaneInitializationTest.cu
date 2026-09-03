#include "core/grid/FieldView.cuh"
#include "core/runtime/CudaContext.h"
#include "core/runtime/DeviceBuffer.h"
#include "core/scene/PlaneScene.cuh"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void Fail(const char* expression, const int line) {
    std::fprintf(stderr, "PlaneInitializationTest:%d check failed: %s\n", line, expression);
    std::abort();
}

void Check(const bool condition, const char* expression, const int line) {
    if (!condition) {
        Fail(expression, line);
    }
}

#define CHECK(expression) Check((expression), #expression, __LINE__)

template <typename Exception, typename Callable>
void ExpectThrows(Callable&& callable) {
    try {
        callable();
        CHECK(false);
    } catch (const Exception&) {
    }
}

std::uint64_t Bits(const double value) {
    return std::bit_cast<std::uint64_t>(value);
}

void CheckPositiveZero(const double value) {
    CHECK(Bits(value) == 0U);
}

void CopyToHost(std::vector<double>& destination, const Pivot::Cuda::DeviceBuffer<double>& source,
                const Pivot::Cuda::CudaContext& context) {
    PIVOT_CUDA_CHECK(cudaMemcpyAsync(destination.data(), source.data(),
                                     destination.size() * sizeof(double),
                                     cudaMemcpyDeviceToHost, context.stream()));
}

void FillWithSentinel(Pivot::Cuda::DeviceBuffer<double>& buffer,
                      const Pivot::Cuda::CudaContext& context) {
    PIVOT_CUDA_CHECK(cudaMemsetAsync(buffer.data(), 0xff, buffer.size() * sizeof(double),
                                     context.stream()));
}

void TestPlaneInitializationAtScale16() {
    static_assert(!std::is_copy_constructible_v<Pivot::Cuda::CudaContext>);
    static_assert(!std::is_copy_assignable_v<Pivot::Cuda::CudaContext>);
    static_assert(std::is_nothrow_move_constructible_v<Pivot::Cuda::CudaContext>);
    static_assert(std::is_nothrow_move_assignable_v<Pivot::Cuda::CudaContext>);
    static_assert(!std::is_copy_constructible_v<Pivot::Cuda::DeviceBuffer<double>>);
    static_assert(!std::is_copy_assignable_v<Pivot::Cuda::DeviceBuffer<double>>);
    static_assert(std::is_nothrow_move_constructible_v<Pivot::Cuda::DeviceBuffer<double>>);
    static_assert(std::is_nothrow_move_assignable_v<Pivot::Cuda::DeviceBuffer<double>>);
    static_assert(std::is_trivially_copyable_v<Pivot::Cuda::FieldView<double>>);
    static_assert(std::is_trivially_copyable_v<Pivot::Cuda::MacFieldView>);

    const Pivot::Cuda::PlaneScene scene = Pivot::Cuda::MakePlaneScene(16);
    Pivot::Cuda::CudaContext original_context;
    Pivot::Cuda::CudaContext context(std::move(original_context));
    CHECK(original_context.stream() == nullptr);
    CHECK(original_context.memory_pool() == nullptr);
    CHECK(context.stream() != nullptr);
    CHECK(context.memory_pool() != nullptr);
    CHECK(context.device_ordinal() == 0);
    int active_device = -1;
    PIVOT_CUDA_CHECK(cudaGetDevice(&active_device));
    CHECK(active_device == context.device_ordinal());

    const std::size_t cell_count = scene.cell_grid.VertexCount();
    const std::size_t face_count_x = scene.face_grids[0].VertexCount();
    const std::size_t face_count_y = scene.face_grids[1].VertexCount();
    const std::size_t face_count_z = scene.face_grids[2].VertexCount();
    CHECK(cell_count == 3072U);
    CHECK(face_count_x == 3264U);
    CHECK(face_count_y == 3328U);
    CHECK(face_count_z == 3264U);

    Pivot::Cuda::DeviceBuffer<double> level_set(cell_count, context.stream());
    Pivot::Cuda::DeviceBuffer<double> velocity_x(face_count_x, context.stream());
    Pivot::Cuda::DeviceBuffer<double> velocity_y(face_count_y, context.stream());
    Pivot::Cuda::DeviceBuffer<double> velocity_z(face_count_z, context.stream());
    CHECK(level_set.size() == cell_count);
    CHECK(level_set.capacity() == cell_count);

    auto initialize = [&] {
        FillWithSentinel(level_set, context);
        FillWithSentinel(velocity_x, context);
        FillWithSentinel(velocity_y, context);
        FillWithSentinel(velocity_z, context);
        Pivot::Cuda::InitializePlaneAsync(
            scene, {level_set.data(), scene.cell_grid},
            {{{velocity_x.data(), scene.face_grids[0]},
              {velocity_y.data(), scene.face_grids[1]},
              {velocity_z.data(), scene.face_grids[2]}}},
            context.stream());
        context.Synchronize();
    };

    std::vector<double> level_set_snapshot(cell_count);
    std::vector<double> velocity_x_snapshot(face_count_x);
    std::vector<double> velocity_y_snapshot(face_count_y);
    std::vector<double> velocity_z_snapshot(face_count_z);
    initialize();
    CopyToHost(level_set_snapshot, level_set, context);
    CopyToHost(velocity_x_snapshot, velocity_x, context);
    CopyToHost(velocity_y_snapshot, velocity_y, context);
    CopyToHost(velocity_z_snapshot, velocity_z, context);
    context.Synchronize();

    CHECK(Bits(scene.initial_time) == 0U);
    CheckPositiveZero(scene.initial_velocity.x);
    CheckPositiveZero(scene.initial_velocity.y);
    CheckPositiveZero(scene.initial_velocity.z);
    for (const double value : velocity_x_snapshot) {
        CheckPositiveZero(value);
    }
    for (const double value : velocity_y_snapshot) {
        CheckPositiveZero(value);
    }
    for (const double value : velocity_z_snapshot) {
        CheckPositiveZero(value);
    }

    bool found_below_plane = false;
    bool found_above_plane = false;
    for (std::size_t index = 0; index < cell_count; ++index) {
        const Pivot::Cuda::Int3 coord = scene.cell_grid.CoordOf(index);
        const double expected = scene.cell_grid.PositionOf(coord).y - scene.liquid_surface_y;
        CHECK(std::isfinite(level_set_snapshot[index]));
        CHECK(Bits(level_set_snapshot[index]) == Bits(expected));
        if (expected < 0.0) {
            found_below_plane = true;
            CHECK(level_set_snapshot[index] < 0.0);
        }
        if (expected > 0.0) {
            found_above_plane = true;
            CHECK(level_set_snapshot[index] > 0.0);
        }
    }
    CHECK(found_below_plane);
    CHECK(found_above_plane);

    const std::vector<double> first_level_set = level_set_snapshot;
    const std::vector<double> first_velocity_x = velocity_x_snapshot;
    const std::vector<double> first_velocity_y = velocity_y_snapshot;
    const std::vector<double> first_velocity_z = velocity_z_snapshot;
    initialize();
    CopyToHost(level_set_snapshot, level_set, context);
    CopyToHost(velocity_x_snapshot, velocity_x, context);
    CopyToHost(velocity_y_snapshot, velocity_y, context);
    CopyToHost(velocity_z_snapshot, velocity_z, context);
    context.Synchronize();
    CHECK(std::memcmp(first_level_set.data(), level_set_snapshot.data(), cell_count * sizeof(double)) ==
          0);
    CHECK(std::memcmp(first_velocity_x.data(), velocity_x_snapshot.data(),
                      face_count_x * sizeof(double)) == 0);
    CHECK(std::memcmp(first_velocity_y.data(), velocity_y_snapshot.data(),
                      face_count_y * sizeof(double)) == 0);
    CHECK(std::memcmp(first_velocity_z.data(), velocity_z_snapshot.data(),
                      face_count_z * sizeof(double)) == 0);
}

void TestDeviceBufferMoveZeroAndOverflow() {
    Pivot::Cuda::CudaContext context;
    Pivot::Cuda::DeviceBuffer<double> empty;
    empty.Allocate(0U, context.stream());
    CHECK(empty.data() == nullptr);
    CHECK(empty.size() == 0U);
    CHECK(empty.capacity() == 0U);

    Pivot::Cuda::DeviceBuffer<double> source(1U, context.stream());
    Pivot::Cuda::DeviceBuffer<double> moved(std::move(source));
    CHECK(source.data() == nullptr);
    CHECK(source.size() == 0U);
    CHECK(source.capacity() == 0U);
    CHECK(moved.data() != nullptr);
    CHECK(moved.size() == 1U);
    CHECK(moved.capacity() == 1U);

    Pivot::Cuda::DeviceBuffer<double> overflow;
    ExpectThrows<std::overflow_error>([&] {
        overflow.Allocate(std::numeric_limits<std::size_t>::max() / sizeof(double) + 1U,
                          context.stream());
    });
}

}  // namespace

int main() {
    TestPlaneInitializationAtScale16();
    TestDeviceBufferMoveZeroAndOverflow();
    return 0;
}
