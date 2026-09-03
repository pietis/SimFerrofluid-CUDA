#include "Manifest.h"

#include "Cli.h"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string_view>

namespace Pivot::Cuda {
namespace {

std::string JsonString(const std::string_view value) {
    std::string result;
    result.reserve(value.size() + 2U);
    result.push_back('"');
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char character : value) {
        switch (character) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (character < 0x20U) {
                    result += "\\u00";
                    result.push_back(hex[character >> 4U]);
                    result.push_back(hex[character & 0x0fU]);
                } else {
                    result.push_back(static_cast<char>(character));
                }
        }
    }
    result.push_back('"');
    return result;
}

std::string JsonNumber(const double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("manifest cannot serialize a non-finite number");
    }
    char buffer[64];
    const auto [end, error] = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                            std::chars_format::general);
    if (error != std::errc{}) {
        throw std::runtime_error("manifest number conversion failed");
    }
    return {buffer, end};
}

void AppendDouble3(std::string& json, const Double3 value) {
    json += '[' + JsonNumber(value.x) + ',' + JsonNumber(value.y) + ',' + JsonNumber(value.z) + ']';
}

void AppendGrid(std::string& json, const GridDesc& grid) {
    json += "{\"size\":[" + std::to_string(grid.size.x) + ',' + std::to_string(grid.size.y) +
            ',' + std::to_string(grid.size.z) + "],\"spacing\":" + JsonNumber(grid.spacing) +
            ",\"inverse_spacing\":" + JsonNumber(grid.inverse_spacing) + ",\"origin\":";
    AppendDouble3(json, grid.origin);
    json += '}';
}

}  // namespace

std::string BuildBootstrapManifestJson(const BootstrapOptions& options, const PlaneScene& scene,
                                       const BootstrapStateMetadata& state,
                                       const CudaDeviceMetadata& device) {
    std::string json = "{\"implementation\":\"cuda\",\"capability\":\"bootstrap-only\","
                       "\"advanced_frames\":0,\"scene\":{\"name\":\"plane\",\"scale\":" +
                       std::to_string(scene.scale) + ",\"boundary_width\":" +
                       std::to_string(scene.boundary_width) + ",\"cell_grid\":";
    AppendGrid(json, scene.cell_grid);
    json += ",\"face_grids\":[";
    AppendGrid(json, scene.face_grids[0]);
    json += ',';
    AppendGrid(json, scene.face_grids[1]);
    json += ',';
    AppendGrid(json, scene.face_grids[2]);
    json += "],\"allocated_min\":";
    AppendDouble3(json, scene.allocated_min);
    json += ",\"allocated_max\":";
    AppendDouble3(json, scene.allocated_max);
    json += ",\"interior_min\":";
    AppendDouble3(json, scene.interior_min);
    json += ",\"interior_max\":";
    AppendDouble3(json, scene.interior_max);
    json += ",\"liquid_surface_y\":" + JsonNumber(scene.liquid_surface_y) +
            ",\"initial_time\":" + JsonNumber(scene.initial_time) + ",\"initial_velocity\":";
    AppendDouble3(json, scene.initial_velocity);
    json += ",\"density\":" + JsonNumber(scene.density) +
            ",\"surface_tension\":" + JsonNumber(scene.surface_tension) + ",\"gravity\":";
    AppendDouble3(json, scene.gravity);
    json += ",\"damping\":" + JsonNumber(scene.damping) + ",\"applied_field\":";
    AppendDouble3(json, scene.applied_field);
    json += ",\"susceptibility\":" + JsonNumber(scene.susceptibility) +
            ",\"lambda\":" + JsonNumber(scene.lambda) +
            ",\"iob_epsilon_factor\":" + JsonNumber(scene.iob_epsilon_factor) +
            ",\"iob_iterations\":" + std::to_string(scene.iob_iterations) +
            "},\"request\":{\"end_frame\":" + std::to_string(options.end_frame) +
            ",\"frame_rate\":" + JsonNumber(options.frame_rate) +
            ",\"cfl\":" + JsonNumber(options.cfl) +
            ",\"magnetic_enabled\":" + (options.magnetic_enabled ? "true" : "false") +
            ",\"magnetic_solver\":" + JsonString(options.magnetic_solver) +
            ",\"dry_run\":" + (options.dry_run ? "true" : "false") +
            "},\"state\":{\"time\":" + JsonNumber(state.time) +
            ",\"cell_phi_count\":" + std::to_string(state.cell_phi_count) +
            ",\"velocity_counts\":[" + std::to_string(state.velocity_counts[0]) + ',' +
            std::to_string(state.velocity_counts[1]) + ',' +
            std::to_string(state.velocity_counts[2]) + "]},\"device\":{\"synthetic\":" +
            (device.synthetic ? "true" : "false") + ",\"ordinal\":" +
            std::to_string(device.ordinal) + ",\"name\":" + JsonString(device.name) +
            ",\"runtime_version\":" + std::to_string(device.runtime_version) +
            ",\"driver_version\":" + std::to_string(device.driver_version) +
            ",\"compute_capability\":[" + std::to_string(device.compute_major) + ',' +
            std::to_string(device.compute_minor) + "]}}";
    return json;
}

}  // namespace Pivot::Cuda
