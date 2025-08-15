cmake_minimum_required(VERSION 3.12)

option(USE_ROCM "Use Rocm" OFF)

project(pingpong)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(pingpong pingpong.cpp)

target_link_libraries(pingpong ${MPI_CXX_LIBRARIES})

if (USE_CALIPER)
    add_definitions(-DUSE_CALIPER)
    find_package(caliper)
    find_package(adiak REQUIRED)
    include_directories(${caliper_INCLUDE_DIR})
    include_directories(${adiak_INCLUDE_DIRS})
    message(STATUS "Caliper on")

    target_link_libraries(pingpong caliper)
    target_link_libraries(pingpong adiak::adiak)
endif ()

if (USE_ROCM)
    include_directories(${ROCM_PATH}/include)    
    target_compile_options(${pingpong} PRIVATE -x rocm -DUSE_ROCM --offload-arch=${ROCM_ARCH})
    message(STATUS "Rocm on")
    target_link_libraries(pingpong "-L${ROCM_PATH}/lib -lamdhip64")
endif ()
