#include "PlaneScene.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#ifndef PIVOT_CUDA_BOOTSTRAP_NOT_A_FLUID_SOLVER
#error "cuda_host_tests must identify this target as not a fluid solver"
#endif

namespace {

constexpr double kTolerance = 1e-15;

[[noreturn]] void Fail(const char* expression, const int line) {
    std::fprintf(stderr, "PlaneSceneContractTest:%d check failed: %s\n", line, expression);
    std::abort();
}

void Check(const bool condition, const char* expression, const int line) {
    if (!condition) {
        Fail(expression, line);
    }
}

#define CHECK(expression) Check((expression), #expression, __LINE__)

void ExpectNear(double actual, double expected) {
    CHECK(std::abs(actual - expected) <= kTolerance);
}

void ExpectInt3(Pivot::Cuda::Int3 actual, Pivot::Cuda::Int3 expected) {
    CHECK(actual.x == expected.x);
    CHECK(actual.y == expected.y);
    CHECK(actual.z == expected.z);
}

void ExpectDouble3(Pivot::Cuda::Double3 actual, Pivot::Cuda::Double3 expected) {
    ExpectNear(actual.x, expected.x);
    ExpectNear(actual.y, expected.y);
    ExpectNear(actual.z, expected.z);
}

template <typename Exception, typename Callable>
void ExpectThrows(Callable&& callable) {
    try {
        callable();
        CHECK(false);
    } catch (const Exception&) {
    }
}

void TestSceneContractAtScale192() {
    const Pivot::Cuda::PlaneScene scene = Pivot::Cuda::MakePlaneScene(192);
    const double dx = 0.12 / 188.0;
    const Pivot::Cuda::Double3 allocated_min{
        -0.06 - 2.0 * dx,
        -(3.0 * 192.0 / 8.0) * dx,
        -0.06 - 2.0 * dx,
    };
    const Pivot::Cuda::Double3 allocated_max{
        -allocated_min.x,
        -allocated_min.y,
        -allocated_min.z,
    };
    const Pivot::Cuda::Double3 interior_min{
        allocated_min.x + 2.0 * dx,
        allocated_min.y + 2.0 * dx,
        allocated_min.z + 2.0 * dx,
    };
    const Pivot::Cuda::Double3 interior_max{
        -interior_min.x,
        -interior_min.y,
        -interior_min.z,
    };

    CHECK(scene.scale == 192);
    CHECK(scene.boundary_width == 2);
    ExpectNear(scene.cell_grid.spacing, dx);
    ExpectNear(scene.cell_grid.inverse_spacing, 1.0 / dx);
    ExpectInt3(scene.cell_grid.size, {192, 144, 192});
    ExpectDouble3(scene.allocated_min, allocated_min);
    ExpectDouble3(scene.allocated_max, allocated_max);
    ExpectDouble3(scene.interior_min, interior_min);
    ExpectDouble3(scene.interior_max, interior_max);
    ExpectDouble3(scene.cell_grid.origin,
                  {allocated_min.x + dx / 2.0,
                   allocated_min.y + dx / 2.0,
                   allocated_min.z + dx / 2.0});
    ExpectNear(scene.liquid_surface_y, interior_min.y + 0.024);

    ExpectInt3(scene.face_grids[0].size, {193, 144, 192});
    ExpectInt3(scene.face_grids[1].size, {192, 145, 192});
    ExpectInt3(scene.face_grids[2].size, {192, 144, 193});
    for (const Pivot::Cuda::GridDesc& face_grid : scene.face_grids) {
        ExpectNear(face_grid.spacing, dx);
        ExpectNear(face_grid.inverse_spacing, 1.0 / dx);
    }
    ExpectDouble3(scene.face_grids[0].origin,
                  {allocated_min.x, allocated_min.y + dx / 2.0,
                   allocated_min.z + dx / 2.0});
    ExpectDouble3(scene.face_grids[1].origin,
                  {allocated_min.x + dx / 2.0, allocated_min.y,
                   allocated_min.z + dx / 2.0});
    ExpectDouble3(scene.face_grids[2].origin,
                  {allocated_min.x + dx / 2.0, allocated_min.y + dx / 2.0,
                   allocated_min.z});

    ExpectNear(scene.initial_time, 0.0);
    ExpectDouble3(scene.initial_velocity, {0.0, 0.0, 0.0});
    ExpectNear(scene.density, 1000.0);
    ExpectNear(scene.surface_tension, 0.0728);
    ExpectDouble3(scene.gravity, {0.0, -9.8, 0.0});
    ExpectNear(scene.damping, 8.0);
    ExpectDouble3(scene.applied_field, {0.0, 60000.0, 0.0});
    ExpectNear(scene.susceptibility, 0.33);
    ExpectNear(scene.lambda, -scene.susceptibility / (2.0 + scene.susceptibility));
    ExpectNear(scene.iob_epsilon_factor, 1.0);
    CHECK(scene.iob_iterations == 10);
}

void TestZFastGridRoundTripsAndBounds() {
    const Pivot::Cuda::GridDesc grid{
        0.5,
        2.0,
        {2, 3, 4},
        {1.0, 2.0, 3.0},
    };

    CHECK(grid.VertexCount() == 24U);
    CHECK(grid.IndexOf({0, 0, 0}) == 0U);
    CHECK(grid.IndexOf({0, 0, 3}) == 3U);
    CHECK(grid.IndexOf({0, 1, 0}) == 4U);
    CHECK(grid.IndexOf({1, 0, 0}) == 12U);
    CHECK(grid.IndexOf({1, 2, 3}) == 23U);
    ExpectInt3(grid.CoordOf(0U), {0, 0, 0});
    ExpectInt3(grid.CoordOf(3U), {0, 0, 3});
    ExpectInt3(grid.CoordOf(4U), {0, 1, 0});
    ExpectInt3(grid.CoordOf(12U), {1, 0, 0});
    ExpectInt3(grid.CoordOf(23U), {1, 2, 3});
    ExpectInt3(grid.CoordOf(grid.IndexOf({0, 0, 0})), {0, 0, 0});
    ExpectInt3(grid.CoordOf(grid.IndexOf({0, 0, 3})), {0, 0, 3});
    ExpectInt3(grid.CoordOf(grid.IndexOf({0, 1, 0})), {0, 1, 0});
    ExpectInt3(grid.CoordOf(grid.IndexOf({1, 0, 0})), {1, 0, 0});
    ExpectInt3(grid.CoordOf(grid.IndexOf({1, 2, 3})), {1, 2, 3});
    ExpectDouble3(grid.PositionOf({1, 2, 3}), {1.5, 3.0, 4.5});

    ExpectThrows<std::out_of_range>([&] { grid.IndexOf({-1, 0, 0}); });
    ExpectThrows<std::out_of_range>([&] { grid.IndexOf({2, 0, 0}); });
    ExpectThrows<std::out_of_range>([&] { grid.CoordOf(24U); });

    const Pivot::Cuda::GridDesc overflowing{
        1.0,
        1.0,
        {std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
         std::numeric_limits<int>::max()},
        {0.0, 0.0, 0.0},
    };
    ExpectThrows<std::overflow_error>([&] { (void)overflowing.VertexCount(); });
}

void TestInvalidScalesAndTrivialCopies() {
    static_assert(std::is_trivially_copyable_v<Pivot::Cuda::Int3>);
    static_assert(std::is_trivially_copyable_v<Pivot::Cuda::Double3>);
    static_assert(std::is_trivially_copyable_v<Pivot::Cuda::GridDesc>);
    static_assert(std::is_trivially_copyable_v<Pivot::Cuda::PlaneScene>);

    ExpectThrows<std::invalid_argument>([] { Pivot::Cuda::MakePlaneScene(4); });
    ExpectThrows<std::invalid_argument>([] { Pivot::Cuda::MakePlaneScene(193); });
}

void TestBootstrapMarker() {
    CHECK(std::string_view(Pivot::Cuda::kCudaBootstrapStatus) ==
          "CUDA bootstrap only - NOT A FLUID SOLVER");
}

}  // namespace

int main() {
    std::puts(Pivot::Cuda::kCudaBootstrapStatus);
    TestSceneContractAtScale192();
    TestZFastGridRoundTripsAndBounds();
    TestInvalidScalesAndTrivialCopies();
    TestBootstrapMarker();
    return 0;
}
