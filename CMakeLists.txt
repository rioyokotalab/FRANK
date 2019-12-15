cmake_minimum_required(VERSION 3.0)

project(HICMA
    LANGUAGES CXX
    VERSION 1.0
)
set(CMAKE_CXX_STANDARD 11)

# Check for OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    list(APPEND HICMA_OPTIONS ${OpenMP_CXX_FLAGS})
endif()
# Flag for MKL Usage
if(NOT USE_MKL)
    set(USE_MKL OFF)
endif()
if(USE_MKL)
    list(APPEND HICMA_DEFINITIONS -DUSE_MKL)
    list(APPEND HICMA_DEPENDENCIES
        mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread
    )
else()
    find_package(CBLAS REQUIRED)
    find_package(LAPACKE REQUIRED)
    list(APPEND HICMA_DEPENDENCIES ${CBLAS_LIBRARIES} ${LAPACKE_LIBRARIES})
endif(USE_MKL)
unset(USE_MKL CACHE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

# Import external dependency loading script
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(add_external)

add_external_dependency(yomm11)
list(APPEND HICMA_DEPENDENCIES yomm11)
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/yomm11-src/include/yorel
    DESTINATION include
)

# Add main source tree
add_subdirectory(src)
# Enable test for googletest
enable_testing()
add_subdirectory(test)
