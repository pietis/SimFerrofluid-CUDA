#include "demo/Cli.h"
#include "demo/Manifest.h"

#include "core/scene/PlaneScene.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void Fail(const char* expression, const int line) {
    std::fprintf(stderr, "CudaCliContractTest:%d check failed: %s\n", line, expression);
    std::abort();
}

void Check(const bool condition, const char* expression, const int line) {
    if (!condition) {
        Fail(expression, line);
    }
}

#define CHECK(expression) Check((expression), #expression, __LINE__)

Pivot::Cuda::BootstrapOptions Parse(std::initializer_list<const char*> arguments) {
    std::vector<const char*> argv(arguments);
    return Pivot::Cuda::ParseBootstrapCommandLine(static_cast<int>(argv.size()), argv.data());
}

void ExpectRejected(std::initializer_list<const char*> arguments) {
    try {
        (void)Parse(arguments);
        CHECK(false);
    } catch (const std::invalid_argument&) {
    }
}

void TestExactDryRunCommand() {
    const Pivot::Cuda::BootstrapOptions options =
        Parse({"demo_cuda", "--test", "plane", "--scale", "16", "--end", "1",
               "--rate", "500", "--cfl", ".5", "--mag=false", "--dry-run"});

    CHECK(options.scene == "plane");
    CHECK(options.scale == 16);
    CHECK(options.end_frame == 1U);
    CHECK(options.frame_rate == 500.0);
    CHECK(options.cfl == 0.5);
    CHECK(!options.magnetic_enabled);
    CHECK(options.magnetic_solver == "iob");
    CHECK(options.dry_run);
}

void TestEqualsFormAndDefaultScale() {
    const Pivot::Cuda::BootstrapOptions options =
        Parse({"demo_cuda", "--test=plane", "--end=2", "--rate=25", "--cfl=1",
               "--mag-solver=iob", "--mag=false", "--dry-run"});
    CHECK(options.scale == 64);
    CHECK(options.end_frame == 2U);
    CHECK(options.frame_rate == 25.0);
    CHECK(options.cfl == 1.0);
}

void TestUnsupportedAndMalformedRequestsAreRejected() {
    ExpectRejected({"demo_cuda", "--test", "dipole", "--end", "1", "--rate", "500",
                    "--cfl", ".5", "--mag=false", "--dry-run"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--end", "1", "--rate", "500",
                    "--cfl", ".5", "--mag=false", "--mag-solver", "fdm", "--dry-run"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--end", "1", "--rate", "500",
                    "--cfl", ".5", "--mag=true", "--dry-run"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--end", "1", "--rate", "500",
                    "--cfl", ".5", "--mag=false"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--rate", "500", "--cfl", ".5",
                    "--mag=false", "--dry-run"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--end", "1", "--cfl", ".5",
                    "--mag=false", "--dry-run"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--end", "1", "--rate", "500",
                    "--mag=false", "--dry-run"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--scale", "5", "--end", "1",
                    "--rate", "500", "--cfl", ".5", "--mag=false", "--dry-run"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--end", "one", "--rate", "500",
                    "--cfl", ".5", "--mag=false", "--dry-run"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--end", "1", "--rate", "0",
                    "--cfl", ".5", "--mag=false", "--dry-run"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--end", "1", "--rate", "500",
                    "--cfl", "nan", "--mag=false", "--dry-run"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--end", "1", "--rate", "500",
                    "--cfl", ".5", "--mag=maybe", "--dry-run"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--end", "1", "--rate", "500",
                    "--cfl", ".5", "--mag=false", "--dry-run", "--use-cpu"});
    ExpectRejected({"demo_cuda", "--test", "plane", "--end", "1", "--rate", "500",
                    "--cfl", ".5", "--mag=false", "--dry-run", "unexpected"});
}

void TestSyntheticManifestIsCompleteValidJson() {
    const Pivot::Cuda::BootstrapOptions options =
        Parse({"demo_cuda", "--test", "plane", "--scale", "16", "--end", "1",
               "--rate", "500", "--cfl", ".5", "--mag=false", "--dry-run"});
    const Pivot::Cuda::PlaneScene scene = Pivot::Cuda::MakePlaneScene(16);
    const Pivot::Cuda::BootstrapStateMetadata state{0.0, 3072U, {3264U, 3328U, 3264U}};
    const Pivot::Cuda::CudaDeviceMetadata device{
        true, 7, "Synthetic \"GPU\" \\ host", 12080, 12090, 12, 0};

    const std::string json = Pivot::Cuda::BuildBootstrapManifestJson(options, scene, state, device);
    const std::string expected =
        "{\"implementation\":\"cuda\",\"capability\":\"bootstrap-only\",\"advanced_frames\":0,"
        "\"scene\":{\"name\":\"plane\",\"scale\":16,\"boundary_width\":2,"
        "\"cell_grid\":{\"size\":[16,12,16],\"spacing\":0.01,\"inverse_spacing\":100,"
        "\"origin\":[-0.075,-0.055,-0.075]},\"face_grids\":["
        "{\"size\":[17,12,16],\"spacing\":0.01,\"inverse_spacing\":100,"
        "\"origin\":[-0.08,-0.055,-0.075]},"
        "{\"size\":[16,13,16],\"spacing\":0.01,\"inverse_spacing\":100,"
        "\"origin\":[-0.075,-0.06,-0.075]},"
        "{\"size\":[16,12,17],\"spacing\":0.01,\"inverse_spacing\":100,"
        "\"origin\":[-0.075,-0.055,-0.08]}],"
        "\"allocated_min\":[-0.08,-0.06,-0.08],\"allocated_max\":[0.08,0.06,0.08],"
        "\"interior_min\":[-0.06,-0.039999999999999994,-0.06],"
        "\"interior_max\":[0.06,0.039999999999999994,0.06],"
        "\"liquid_surface_y\":-0.015999999999999993,\"initial_time\":0,\"initial_velocity\":[0,0,0],"
        "\"density\":1000,\"surface_tension\":0.0728,\"gravity\":[0,-9.8,0],"
        "\"damping\":8,\"applied_field\":[0,60000,0],\"susceptibility\":0.33,"
        "\"lambda\":-0.14163090128755365,\"iob_epsilon_factor\":1,\"iob_iterations\":10},"
        "\"request\":{\"end_frame\":1,\"frame_rate\":500,\"cfl\":0.5,"
        "\"magnetic_enabled\":false,\"magnetic_solver\":\"iob\",\"dry_run\":true},"
        "\"state\":{\"time\":0,\"cell_phi_count\":3072,"
        "\"velocity_counts\":[3264,3328,3264]},"
        "\"device\":{\"synthetic\":true,\"ordinal\":7,"
        "\"name\":\"Synthetic \\\"GPU\\\" \\\\ host\",\"runtime_version\":12080,"
        "\"driver_version\":12090,\"compute_capability\":[12,0]}}";
    CHECK(json == expected);
}

}  // namespace

int main() {
    TestExactDryRunCommand();
    TestEqualsFormAndDefaultScale();
    TestUnsupportedAndMalformedRequestsAreRejected();
    TestSyntheticManifestIsCompleteValidJson();
    return 0;
}
