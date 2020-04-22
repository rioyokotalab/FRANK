cmake_minimum_required(VERSION 3.5)

project(HICMA
    LANGUAGES CXX
    VERSION 1.0
)

# Use Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Require C++ 17
if(${CMAKE_VERSION} VERSION_LESS "3.8.2")
    # TODO This is not compiler agnostic (only clang, gcc)
    list(APPEND HICMA_OPTIONS -std=c++17)
else()
    list(APPEND HICMA_FEATURES cxx_std_17)
endif()

# Set appropriate compiler options
list(APPEND HICMA_OPTIONS -Wall -Wextra -pedantic)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND HICMA_OPTIONS -O3)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND HICMA_OPTIONS -g)
endif()

# Output folder settings
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (
        CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/install" CACHE PATH "default install path" FORCE
    )
endif()


## Dependencies
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
    list(APPEND HICMA_DEFINITIONS USE_MKL)
    list(APPEND HICMA_DEPENDENCIES
        mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread
    )
else()
    find_package(CBLAS REQUIRED)
    find_package(LAPACKE REQUIRED)
    list(APPEND HICMA_DEPENDENCIES ${CBLAS_LIBRARIES} ${LAPACKE_LIBRARIES})
endif(USE_MKL)
unset(USE_MKL CACHE)

# Open Multi-Method depdency
# Import external dependency loading script
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(add_external)

add_external_dependency(yomm2)
list(APPEND HICMA_DEPENDENCIES yomm2)
list(APPEND DEPDENCY_INCLUDES ${CMAKE_BINARY_DIR}/yomm2-src/include)
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/yomm2-src/include/yorel
    DESTINATION include
)

# Add main source tree
add_subdirectory(src)
# Enable test for googletest
enable_testing()
add_subdirectory(test)
