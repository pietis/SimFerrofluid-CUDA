option("cuda_backend")
    set_default(false)
    set_showmenu(true)
    set_description("Build the CUDA runtime bootstrap targets (CUDA 12.8+ required).")
option_end()

local cuda_root = path.absolute(os.scriptdir())

local function audit_cuda_target_boundary(target)
    local function is_within(root, candidate)
        local relative = path.relative(candidate, root)
        return relative == "." or
               (relative ~= ".." and not relative:startswith("../") and not path.is_absolute(relative))
    end

    local project_root = path.absolute(os.projectdir())
    local repository_root = path.absolute(path.directory(cuda_root))
    local function audit_local_includes(sourcefile, visited)
        local source = path.absolute(sourcefile, project_root)
        assert(is_within(cuda_root, source),
               string.format("CUDA boundary audit: quoted include source escapes CUDA root: %s", source))
        if visited[source] then
            return
        end
        visited[source] = true

        local contents = assert(io.readfile(source),
                                string.format("CUDA boundary audit: cannot read source: %s", source))
        for operand in contents:gmatch("#%s*include%s*\"([^\"]+)\"") do
            local resolved = path.absolute(operand, path.directory(source))
            if not os.isfile(resolved) then
                resolved = path.absolute(operand, cuda_root)
            end
            if os.isfile(resolved) then
                assert(is_within(cuda_root, resolved),
                       string.format("CUDA boundary audit: quoted include escapes CUDA root: %s -> %s",
                                     source, resolved))
                audit_local_includes(resolved, visited)
            end
        end
    end

    local allowed_source_prefix = {
        sim_cuda = "core/",
        demo_cuda = "demo/",
    }
    local allowed_dependencies = {
        sim_cuda = {},
        demo_cuda = {sim_cuda = true},
    }
    local source_prefix = assert(allowed_source_prefix[target:name()],
                                 "CUDA boundary audit attached to an unknown target")

    local visited_includes = {}
    for _, sourcebatch in pairs(target:sourcebatches()) do
        for _, sourcefile in ipairs(sourcebatch.sourcefiles or {}) do
            local source = path.relative(path.absolute(sourcefile, project_root), cuda_root)
            assert(source:startswith(source_prefix),
                   string.format("CUDA boundary audit: %s source escapes %s: %s",
                                 target:name(), source_prefix, source))
            audit_local_includes(sourcefile, visited_includes)
        end
    end

    local expected_dependencies = allowed_dependencies[target:name()]
    for dependency_name, _ in pairs(target:deps()) do
        assert(expected_dependencies[dependency_name],
               string.format("CUDA boundary audit: %s has forbidden dependency: %s",
                             target:name(), dependency_name))
    end
    for dependency_name, _ in pairs(expected_dependencies) do
        assert(target:deps()[dependency_name],
               string.format("CUDA boundary audit: %s is missing dependency: %s",
                             target:name(), dependency_name))
    end

    local cuda_sdk_root = get_config("cuda")
    local cuda_include_dir = cuda_sdk_root and path.absolute(path.join(cuda_sdk_root, "include"))
    if cuda_sdk_root then
        assert(not is_within(repository_root, cuda_include_dir),
               "CUDA boundary audit: configured CUDA SDK unexpectedly resides inside this repository")
    end
    for _, includedir in ipairs(table.wrap(target:get("includedirs", {public = true}) or {})) do
        local resolved = path.absolute(includedir, project_root)
        if is_within(repository_root, resolved) then
            assert(is_within(cuda_root, resolved),
                   string.format("CUDA boundary audit: %s include dir escapes CUDA root: %s",
                                 target:name(), resolved))
        else
            assert(resolved == cuda_include_dir,
                   string.format("CUDA boundary audit: %s has forbidden external include dir: %s",
                                 target:name(), resolved))
        end
    end
end

rule("cuda.bootstrap.boundary_audit")
    add_deps("cuda")
    after_load(audit_cuda_target_boundary)

local sim_cuda_sources = {
    "core/scene/PlaneScene.cpp",
    "core/runtime/CudaContext.cu",
    "core/scene/PlaneScene.cu",
    "core/sim/Simulation.cu",
}

local demo_cuda_sources = {
    "demo/Main.cpp",
    "demo/Cli.cpp",
    "demo/Manifest.cpp",
    "demo/DeviceLink.cu",
}

local demo_cuda_dependencies = {"sim_cuda"}

target("cuda_host_tests")
    set_default(false)
    set_kind("binary")
    set_languages("cxx20")
    add_defines("PIVOT_CUDA_BOOTSTRAP_NOT_A_FLUID_SOLVER=1", { public = true })
    add_includedirs(".", { public = true })
    add_includedirs("core/grid", "core/scene")
    add_files("core/scene/PlaneScene.cpp")
    add_files("tests/host/PlaneSceneContractTest.cpp")

target("cuda_cli_host_tests")
    set_default(false)
    set_kind("binary")
    set_languages("cxx20")
    add_includedirs(".")
    add_files("core/scene/PlaneScene.cpp")
    add_files("demo/Cli.cpp")
    add_files("demo/Manifest.cpp")
    add_files("tests/host/CudaCliContractTest.cpp")

target("cuda_source_boundary_tests")
    set_default(false)
    set_kind("binary")
    set_languages("cxx20")
    add_files("tests/host/CudaSourceBoundaryTest.cpp")
    set_runargs(os.scriptdir(), table.concat(sim_cuda_sources, ";"), "",
                table.concat(demo_cuda_sources, ";"),
                table.concat(demo_cuda_dependencies, ";"))

if has_config("cuda_backend") then
    local cuda_sdk_root = get_config("cuda")

    target("sim_cuda")
        set_default(false)
        set_kind("static")
        set_languages("cxx20")
        add_rules("cuda.bootstrap.boundary_audit")
        add_includedirs(".", { public = true })
        add_linkdirs(path.join(cuda_sdk_root, "lib64"), { public = true })
        add_links("cudart", { public = true })
        add_cugencodes("sm_80", "sm_89", "sm_120", "compute_120")
        add_cuflags("--fmad=false")
        add_files(table.unpack(sim_cuda_sources))

    target("cuda_runtime_tests")
        set_default(false)
        set_kind("binary")
        set_languages("cxx20")
        add_deps("sim_cuda")
        add_includedirs(".")
        add_cugencodes("sm_80", "sm_89", "sm_120", "compute_120")
        add_files("tests/runtime/PlaneInitializationTest.cu")

    target("demo_cuda")
        set_default(false)
        set_kind("binary")
        set_languages("cxx20")
        add_rules("cuda.bootstrap.boundary_audit")
        add_deps("sim_cuda")
        add_includedirs(".")
        add_cugencodes("sm_80", "sm_89", "sm_120", "compute_120")
        add_cuflags("--fmad=false")
        add_files(table.unpack(demo_cuda_sources))
end
