cmake_minimum_required(VERSION 3.24)

# This is deliberately a separate configure/build, not part of the Strata target.
# The deployed toolkit must advertise sm_75 (check nvcc --list-gpu-arch).
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "Expert worker CUDA architecture")
project(strata_network_experts LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

get_filename_component(STRATA_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_executable(strata-network-experts
    worker.cu
    "${STRATA_SOURCE_ROOT}/src/kernels/cuda/iq_kernels.cu")
set_target_properties(strata-network-experts PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES CUDA_EXTENSIONS NO)
target_include_directories(strata-network-experts PRIVATE
    "${STRATA_SOURCE_ROOT}/include"
    "${STRATA_SOURCE_ROOT}/third_party/ggml"
    "${STRATA_SOURCE_ROOT}/third_party/ggml-tq"
    "${STRATA_SOURCE_ROOT}/src/kernels/cuda")
target_compile_options(strata-network-experts PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
target_link_libraries(strata-network-experts PRIVATE CUDA::cudart)
