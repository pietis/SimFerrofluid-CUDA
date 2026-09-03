#include "Cli.h"

#include "core/scene/PlaneScene.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace Pivot::Cuda {
namespace {

[[noreturn]] void Reject(const char* message) {
    throw std::invalid_argument(message);
}

int ParseInt(const std::string_view value, const char* message) {
    int result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        Reject(message);
    }
    return result;
}

std::uint64_t ParseUint64(const std::string_view value, const char* message) {
    std::uint64_t result = 0U;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        Reject(message);
    }
    return result;
}

double ParsePositiveDouble(const std::string_view value, const char* message) {
    const std::string copy(value);
    char* end = nullptr;
    const double result = std::strtod(copy.c_str(), &end);
    if (end != copy.c_str() + copy.size() || !std::isfinite(result) || result <= 0.0) {
        Reject(message);
    }
    return result;
}

bool ParseBoolean(const std::string_view value) {
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    Reject("--mag must be true or false");
}

}  // namespace

BootstrapOptions ParseBootstrapCommandLine(const int argc, const char* const* argv) {
    BootstrapOptions options;
    bool seen_scene = false;
    bool seen_scale = false;
    bool seen_end = false;
    bool seen_rate = false;
    bool seen_cfl = false;
    bool seen_mag = false;
    bool seen_solver = false;
    bool seen_dry_run = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--dry-run") {
            if (seen_dry_run) {
                Reject("duplicate --dry-run");
            }
            seen_dry_run = true;
            options.dry_run = true;
            continue;
        }
        if (!argument.starts_with("--")) {
            Reject("unexpected positional argument");
        }

        const std::size_t equals = argument.find('=');
        const std::string_view key = argument.substr(0U, equals);
        std::string_view value;
        if (equals != std::string_view::npos) {
            value = argument.substr(equals + 1U);
        } else {
            if (index + 1 >= argc || std::string_view(argv[index + 1]).starts_with("--")) {
                Reject("option value is missing");
            }
            value = argv[++index];
        }
        if (value.empty()) {
            Reject("option value is empty");
        }

        if (key == "--test") {
            if (seen_scene) Reject("duplicate --test");
            seen_scene = true;
            options.scene = value;
        } else if (key == "--scale") {
            if (seen_scale) Reject("duplicate --scale");
            seen_scale = true;
            options.scale = ParseInt(value, "--scale must be an integer");
        } else if (key == "--end") {
            if (seen_end) Reject("duplicate --end");
            seen_end = true;
            options.end_frame = ParseUint64(value, "--end must be an unsigned integer");
        } else if (key == "--rate") {
            if (seen_rate) Reject("duplicate --rate");
            seen_rate = true;
            options.frame_rate = ParsePositiveDouble(value, "--rate must be finite and positive");
        } else if (key == "--cfl") {
            if (seen_cfl) Reject("duplicate --cfl");
            seen_cfl = true;
            options.cfl = ParsePositiveDouble(value, "--cfl must be finite and positive");
        } else if (key == "--mag") {
            if (seen_mag) Reject("duplicate --mag");
            seen_mag = true;
            options.magnetic_enabled = ParseBoolean(value);
        } else if (key == "--mag-solver") {
            if (seen_solver) Reject("duplicate --mag-solver");
            seen_solver = true;
            options.magnetic_solver = value;
        } else {
            Reject("unknown option");
        }
    }

    if (!seen_scene || !seen_end || !seen_rate || !seen_cfl || !seen_dry_run) {
        Reject("--test, --end, --rate, --cfl, and --dry-run are required");
    }
    if (options.scene != "plane") Reject("only --test plane is supported");
    if (options.magnetic_solver != "iob") Reject("only --mag-solver iob is supported");
    if (options.magnetic_enabled) Reject("magnetic execution is unavailable during bootstrap");
    (void)MakePlaneScene(options.scale);
    return options;
}

}  // namespace Pivot::Cuda
