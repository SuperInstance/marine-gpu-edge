cmake_minimum_required(VERSION 3.20)

# ============================================================
# CUDA TOOLCHAIN — must be set before project() so CMake's
# language detection picks up the right nvcc.
# ============================================================
# Prefer CUDA 12.6 if present; fall back to whatever is on PATH.
if(EXISTS "/usr/local/cuda-12.6/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc"
        CACHE PATH "CUDA compiler" FORCE)
    set(CUDAToolkit_ROOT "/usr/local/cuda-12.6"
        CACHE PATH "CUDA toolkit root" FORCE)
endif()

project(MarineGPUEdge
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

# ============================================================
# STANDARDS & TOOLCHAIN
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit 12.6 REQUIRED)

# ============================================================
# BUILD TYPE DEFAULTS
# ============================================================

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG    "-g -O0 -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG   "-g -G -O0")

# Useful for all builds
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# ============================================================
# ARCHITECTURE TARGETS
# ============================================================
# sm_89 = RTX 4050 Ada Lovelace (workstation / eileen)
# sm_87 = Jetson Orin Nano      (edge / jetsonclaw1)
#
# Build for both by default so the same binary runs on either;
# for a Jetson-only cross-compile pass -DTARGET_ARCH=jetson.

option(TARGET_ARCH "Target architecture: 'workstation', 'jetson', or 'both'" "both")

if(TARGET_ARCH STREQUAL "workstation")
    set(CUDA_ARCH_LIST "89")
    set(ARCH_DEFINES WORKSTATION)
elseif(TARGET_ARCH STREQUAL "jetson")
    set(CUDA_ARCH_LIST "87")
    set(ARCH_DEFINES JETSON_ORIN)
else()  # both
    set(CUDA_ARCH_LIST "89;87")
    set(ARCH_DEFINES "")
endif()

# ============================================================
# CROSS-COMPILATION (aarch64 for Jetson)
# ============================================================
# Enable with:  cmake .. -DCROSS_COMPILE_JETSON=ON
#
# Requires aarch64-linux-gnu-g++ installed.
# Run from a separate build directory, e.g. build-jetson/.

option(CROSS_COMPILE_JETSON "Cross-compile for Jetson Orin (aarch64)" OFF)

if(CROSS_COMPILE_JETSON)
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(CROSS_PREFIX aarch64-linux-gnu-)
    set(CMAKE_C_COMPILER   ${CROSS_PREFIX}gcc)
    set(CMAKE_CXX_COMPILER ${CROSS_PREFIX}g++)
    # Tell nvcc which host compiler to use
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/aarch64-linux-gnu-g++)
    set(CUDA_ARCH_LIST "87")
    set(ARCH_DEFINES JETSON_ORIN)
    message(STATUS "Cross-compiling for Jetson Orin (aarch64, sm_87)")
endif()

# ============================================================
# SHARED INCLUDE DIRECTORY
# ============================================================

include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================
# TARGET: marine_fusion  (CUDA kernels library)
# ============================================================

add_library(marine_fusion STATIC
    src/marine_sensor_fusion.cu
)

set_target_properties(marine_fusion PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(marine_fusion PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(marine_fusion PRIVATE ${ARCH_DEFINES})

target_link_libraries(marine_fusion PUBLIC
    CUDA::cudart
)

# ============================================================
# TARGET: mep_bridge_test  (standalone MEP protocol test)
# ============================================================

add_executable(mep_bridge_test
    src/mep_bridge.cpp
)

set_target_properties(mep_bridge_test PROPERTIES
    CXX_STANDARD 17
)

target_include_directories(mep_bridge_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(mep_bridge_test PRIVATE
    MEP_STANDALONE_TEST
    ${ARCH_DEFINES}
)

# Link pthreads for std::thread used in standalone test
find_package(Threads REQUIRED)
target_link_libraries(mep_bridge_test PRIVATE Threads::Threads)

# ============================================================
# TARGET: marine_adaptive  (precision controller + scheduler)
# ============================================================

add_library(marine_adaptive STATIC
    src/adaptive_precision.cu
)

set_target_properties(marine_adaptive PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(marine_adaptive PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(marine_adaptive PRIVATE ${ARCH_DEFINES})

target_link_libraries(marine_adaptive PUBLIC
    CUDA::cudart
)

# ============================================================
# TARGET: marine_pipeline  (fusion pipeline orchestrator)
# ============================================================

add_library(marine_pipeline STATIC
    src/fusion_pipeline.cu
)

set_target_properties(marine_pipeline PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
    # fusion_pipeline.cu only calls host wrappers — no cross-TU device symbols
    # to resolve, so separable compilation is not needed here.
    CUDA_SEPARABLE_COMPILATION OFF
)

target_include_directories(marine_pipeline PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(marine_pipeline PRIVATE ${ARCH_DEFINES})

target_link_libraries(marine_pipeline PUBLIC
    marine_fusion
    CUDA::cudart
)

# ============================================================
# TARGET: bench_nmea  (NMEA parse benchmark)
# ============================================================

add_executable(bench_nmea
    benchmarks/bench_nmea.cu
)

set_target_properties(bench_nmea PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
)

target_include_directories(bench_nmea PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(bench_nmea PRIVATE ${ARCH_DEFINES})

target_link_libraries(bench_nmea PRIVATE
    marine_fusion
    CUDA::cudart
)

# ============================================================
# TARGET: bench_fusion  (full pipeline integration benchmark)
# ============================================================

add_executable(bench_fusion
    benchmarks/bench_fusion.cu
    # Compile fusion_pipeline.cu directly into this executable so all CUDA
    # source files share a single nvlink scope.  The separate marine_pipeline
    # static library exists for standalone / server-side use; including the
    # source here avoids mixing relocatable and non-relocatable device code
    # across library boundaries, which confuses nvcc -dlink.
    src/fusion_pipeline.cu
)

set_target_properties(bench_fusion PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(bench_fusion PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(bench_fusion PRIVATE ${ARCH_DEFINES})

target_link_libraries(bench_fusion PRIVATE
    marine_fusion
    marine_adaptive
    CUDA::cudart
)

# ============================================================
# TARGET: bench_scheduler  (task scheduler benchmark)
# ============================================================

add_executable(bench_scheduler
    benchmarks/bench_scheduler.cu
)

set_target_properties(bench_scheduler PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
)

target_include_directories(bench_scheduler PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(bench_scheduler PRIVATE ${ARCH_DEFINES})

target_link_libraries(bench_scheduler PRIVATE
    marine_adaptive
    CUDA::cudart
)

# ============================================================
# INSTALL RULES (optional)
# ============================================================

install(TARGETS marine_fusion
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

# ============================================================
# STATUS SUMMARY
# ============================================================

message(STATUS "")
message(STATUS "marine-gpu-edge configuration:")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA arch(s)    : ${CUDA_ARCH_LIST}")
message(STATUS "  Cross-compile   : ${CROSS_COMPILE_JETSON}")
message(STATUS "  Defines         : ${ARCH_DEFINES}")
message(STATUS "  CUDA toolkit    : ${CUDAToolkit_VERSION} @ ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "")
