cmake_minimum_required(VERSION 3.0)

project(HICMA
    LANGUAGES CXX
    VERSION 1.0
)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_library(util
    src/util/print.cpp
    src/util/timer.cpp
)
target_include_directories(util PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(gpu_batch
    src/gpu_batch/batch.cpp
)
target_include_directories(gpu_batch PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(hicma
    src/any.cpp
    src/node.cpp
    src/dense.cpp
    src/low_rank.cpp
    src/hierarchical.cpp
    src/functions.cpp
)
target_include_directories(hicma PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(hicma util gpu_batch)
# Flag for MKL Usage
set(USE_MKL OFF)
if(USE_MKL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MKL")
  target_link_libraries(hicma mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread)
else()
  target_link_libraries(hicma blas lapacke)
endif(USE_MKL)

# Enable test for googletest
enable_testing()
# Import external dependency loading script
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(add_external)
add_subdirectory(test)
