#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void Fail(const char* expression, const int line) {
    std::fprintf(stderr, "CudaSourceBoundaryTest:%d check failed: %s\n", line, expression);
    std::abort();
}

void Check(const bool condition, const char* expression, const int line) {
    if (!condition) {
        Fail(expression, line);
    }
}

#define CHECK(expression) Check((expression), #expression, __LINE__)

std::vector<std::string> Split(const std::string_view value) {
    std::vector<std::string> parts;
    std::size_t begin = 0U;
    while (begin < value.size()) {
        const std::size_t end = value.find(';', begin);
        parts.emplace_back(value.substr(begin, end == std::string_view::npos ? end : end - begin));
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1U;
    }
    return parts;
}

bool Contains(const std::vector<std::string>& values, const std::string_view expected) {
    return std::find(values.begin(), values.end(), expected) != values.end();
}

bool IsInside(const std::filesystem::path& root, const std::filesystem::path& candidate) {
    const std::filesystem::path relative = candidate.lexically_relative(root);
    return !relative.empty() && *relative.begin() != "..";
}

void AuditIncludes(const std::filesystem::path& root, const std::filesystem::path& source,
                   std::set<std::filesystem::path>& visited) {
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(source);
    CHECK(IsInside(root, canonical));
    if (!visited.insert(canonical).second) {
        return;
    }

    std::ifstream input(canonical);
    CHECK(input.good());
    std::string line;
    while (std::getline(input, line)) {
        CHECK(line.find("--use-cpu") == std::string::npos);
        CHECK(line.find("--cpu-fallback") == std::string::npos);
        CHECK(line.find("--fallback") == std::string::npos);

        const std::size_t include = line.find("#include \"");
        if (include == std::string::npos) {
            continue;
        }
        const std::size_t begin = include + 10U;
        const std::size_t end = line.find('"', begin);
        CHECK(end != std::string::npos);
        const std::filesystem::path operand = line.substr(begin, end - begin);
        std::filesystem::path resolved = canonical.parent_path() / operand;
        if (!std::filesystem::exists(resolved)) {
            resolved = root / operand;
        }
        CHECK(std::filesystem::exists(resolved));
        AuditIncludes(root, resolved, visited);
    }
}

void AuditSources(const std::filesystem::path& root, const std::vector<std::string>& sources) {
    std::set<std::filesystem::path> visited;
    for (const std::string& declared : sources) {
        CHECK(declared.find("fmmtl") == std::string::npos);
        CHECK(declared.find("thrust") == std::string::npos);
        CHECK(declared.find("SimViewer") == std::string::npos);
        const std::filesystem::path source = root / declared;
        CHECK(std::filesystem::exists(source));
        AuditIncludes(root, source, visited);
    }
}

}  // namespace

int main(const int argc, char** argv) {
    CHECK(argc == 6);
    const std::filesystem::path root = std::filesystem::weakly_canonical(argv[1]);
    const std::vector<std::string> sim_sources = Split(argv[2]);
    const std::vector<std::string> sim_dependencies = Split(argv[3]);
    const std::vector<std::string> demo_sources = Split(argv[4]);
    const std::vector<std::string> demo_dependencies = Split(argv[5]);

    CHECK(sim_dependencies.empty());
    CHECK(demo_dependencies == std::vector<std::string>{"sim_cuda"});
    CHECK(Contains(sim_sources, "core/sim/Simulation.cu"));
    CHECK(Contains(demo_sources, "demo/Main.cpp"));
    CHECK(Contains(demo_sources, "demo/Cli.cpp"));
    CHECK(Contains(demo_sources, "demo/Manifest.cpp"));
    AuditSources(root, sim_sources);
    AuditSources(root, demo_sources);
    return 0;
}
