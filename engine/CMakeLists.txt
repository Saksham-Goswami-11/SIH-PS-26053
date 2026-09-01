cmake_minimum_required(VERSION 3.18)
project(kavach_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── Options ──────────────────────────────────────────────────────────
option(KAVACH_ENABLE_CUDA "Enable CUDA GPU support" ON)
option(KAVACH_BUILD_TESTS "Build C++ unit tests" ON)

# ─── Find pybind11 ───────────────────────────────────────────────────
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Fallback: try to find via pip-installed pybind11
    execute_process(
        COMMAND python3 -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_result
    )
    if(pybind11_result EQUAL 0)
        find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

# ─── CUDA Detection ──────────────────────────────────────────────────
if(KAVACH_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        add_compile_definitions(KAVACH_HAS_CUDA)
        message(STATUS "CUDA found — building with GPU support (Tier 1)")
    else()
        message(STATUS "CUDA not found — building CPU-only (Tier 2)")
        set(KAVACH_ENABLE_CUDA OFF)
    endif()
endif()

# ─── Core Engine Library ─────────────────────────────────────────────
set(ENGINE_SOURCES
    src/grid_polar_cpu.cpp
    src/hardware_detector.cpp
)

if(KAVACH_ENABLE_CUDA)
    list(APPEND ENGINE_SOURCES src/grid_polar_cuda.cu)
endif()

add_library(kavach_core STATIC ${ENGINE_SOURCES})
target_include_directories(kavach_core PUBLIC include)

if(KAVACH_ENABLE_CUDA)
    set_target_properties(kavach_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
    )
    target_link_libraries(kavach_core PUBLIC ${CUDA_LIBRARIES})
endif()

# Optimization flags
target_compile_options(kavach_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# ─── pybind11 Python Module ──────────────────────────────────────────
pybind11_add_module(kavach_engine bindings/pybind_module.cpp)
target_link_libraries(kavach_engine PRIVATE kavach_core)
target_include_directories(kavach_engine PRIVATE include)

# Install the Python module into a sensible location
install(TARGETS kavach_engine LIBRARY DESTINATION .)

# ─── C++ Tests ───────────────────────────────────────────────────────
if(KAVACH_BUILD_TESTS)
    add_executable(test_engine tests/test_engine.cpp)
    target_link_libraries(test_engine PRIVATE kavach_core)
    target_include_directories(test_engine PRIVATE include)
endif()

message(STATUS "")
message(STATUS "╔══════════════════════════════════════════╗")
message(STATUS "║  KAVACH-2.5D Engine Build Configuration  ║")
message(STATUS "╠══════════════════════════════════════════╣")
if(KAVACH_ENABLE_CUDA)
    message(STATUS "║  Compute:  CUDA + C++ (Tier 1 + Tier 2) ║")
else()
    message(STATUS "║  Compute:  C++ only (Tier 2)             ║")
endif()
message(STATUS "║  Python:   pybind11 module               ║")
message(STATUS "║  Tests:    ${KAVACH_BUILD_TESTS}                          ║")
message(STATUS "╚══════════════════════════════════════════╝")
message(STATUS "")
