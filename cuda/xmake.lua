option("cuda_backend")
    set_default(false)
    set_showmenu(true)
    set_description("Build the CUDA runtime bootstrap targets (CUDA 12.8+ required).")
option_end()

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
        add_rules("cuda")
        add_deps("sim_cuda")
        add_includedirs(".")
        add_cugencodes("sm_80", "sm_89", "sm_120", "compute_120")
        add_cuflags("--fmad=false")
        add_files(table.unpack(demo_cuda_sources))
end
