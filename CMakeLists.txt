cmake_minimum_required(VERSION 3.0)

project(HICMA CXX)
set(HICMA_VERSION_MAJOR 1)
set(HICMA_VERSION_MINOR 0)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(util
    src/util/print.cpp
    src/util/timer.cpp
)
add_library(gpu_batch
    src/gpu_batch/batch.cpp
)
add_library(hicma
    src/any.cpp
    src/node.cpp
    src/dense.cpp
    src/low_rank.cpp
    src/hierarchical.cpp
    src/functions.cpp
)
target_link_libraries(hicma util gpu_batch blas lapacke)

add_executable(block_lu test/block_lu.cpp)
target_link_libraries(block_lu hicma)

add_executable(blr_lu test/blr_lu.cpp)
target_link_libraries(blr_lu hicma)

add_executable(h_lu test/h_lu.cpp)
target_link_libraries(h_lu hicma)

add_executable(rsvd test/rsvd.cpp)
target_link_libraries(rsvd hicma)