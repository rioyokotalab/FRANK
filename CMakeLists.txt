cmake_minimum_required(VERSION 3.0)

project(HICMA
    LANGUAGES CXX
    VERSION 1.0
)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Import external dependency loading script
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(add_external)

add_library(util OBJECT
    src/util/print.cpp
    src/util/timer.cpp
    src/util/counter.cpp
)
target_include_directories(util
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/yomm11-src/include>
)

add_library(gpu_batch OBJECT
    src/gpu_batch/batch.cpp
)
target_include_directories(gpu_batch
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/yomm11-src/include>
)

add_external_dependency(yomm11)
add_library(hicma
    src/node.cpp
    src/node_proxy.cpp
    src/dense.cpp
    src/low_rank.cpp
    src/low_rank_shared.cpp
    src/hierarchical.cpp
    src/uniform_hierarchical.cpp
    src/functions.cpp
    src/operations/gemm.cpp
    src/operations/geqrt.cpp
    src/operations/geqp3.cpp
    src/operations/getrf.cpp
    src/operations/get_dim.cpp
    src/operations/id.cpp
    src/operations/larfb.cpp
    src/operations/latms.cpp
    src/operations/misc.cpp
    src/operations/qr.cpp
    src/operations/tpmqrt.cpp
    src/operations/tpqrt.cpp
    src/operations/trmm.cpp
    src/operations/trsm.cpp
    $<TARGET_OBJECTS:util>
    $<TARGET_OBJECTS:gpu_batch>
)

# Custom compilation options
target_compile_options(util PUBLIC -O3)
target_compile_options(gpu_batch PUBLIC -O3)
target_compile_options(hicma PUBLIC -O3)

target_include_directories(hicma
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/yomm11-src/include>
)
target_link_libraries(hicma yomm11)

# Check for OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
  target_compile_options(hicma PUBLIC ${OpenMP_CXX_FLAGS})
endif()
# Flag for MKL Usage
if(NOT USE_MKL)
    set(USE_MKL OFF)
endif()
if(USE_MKL)
    target_compile_definitions(util PUBLIC -DUSE_MKL)
    target_compile_definitions(gpu_batch PUBLIC -DUSE_MKL)
    target_compile_definitions(hicma PUBLIC -DUSE_MKL)
    target_link_libraries(hicma mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread)
else()
    target_link_libraries(hicma cblas lapacke)
endif(USE_MKL)
unset(USE_MKL CACHE)

# Enable test for googletest
enable_testing()
add_subdirectory(test)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "default install path" FORCE )
endif()
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/hicma
    DESTINATION include
)
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/yomm11-src/include/yorel
    DESTINATION include
)
install(
    FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libyomm11.a
    DESTINATION lib
)
install(
    TARGETS hicma
    EXPORT hicma
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(
    EXPORT hicma
    DESTINATION cmake
)