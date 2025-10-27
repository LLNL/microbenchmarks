cmake_minimum_required(VERSION 3.12)

project(pingpong LANGUAGES CXX)

option(USE_HIP "Use HIP" OFF)
option(USE_CUDA "Use CUDA" OFF)
option(USE_CALIPER "Use Caliper" OFF)

add_executable(pingpong pingpong.cpp)

find_package(MPI REQUIRED)
target_link_libraries(pingpong PRIVATE MPI::MPI_CXX)

if (USE_CALIPER)
    add_definitions(-DUSE_CALIPER)
    find_package(caliper)
    find_package(adiak REQUIRED)
    include_directories(${caliper_INCLUDE_DIR})
    include_directories(${adiak_INCLUDE_DIRS})
    message(STATUS "Caliper on")

    target_link_libraries(pingpong PRIVATE caliper)
    target_link_libraries(pingpong PRIVATE adiak::adiak)
endif()

if (USE_HIP)
    find_package(hip CONFIG REQUIRED)
    add_definitions(-D__HIP_PLATFORM_AMD__)
    target_compile_definitions(pingpong PRIVATE USE_HIP)
    target_link_libraries(pingpong PRIVATE hip::amdhip64)
    message(STATUS "Rocm on")
endif()

if (USE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(pingpong PRIVATE USE_CUDA)
    find_package(CUDAToolkit QUIET)
    target_link_libraries(pingpong PRIVATE CUDA::cudart)
    message(STATUS "CUDA on")
endif()
