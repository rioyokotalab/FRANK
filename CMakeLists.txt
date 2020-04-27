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

# Set warnings and optimization level
list(APPEND HICMA_OPTIONS -Wall -Wextra -pedantic)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND HICMA_OPTIONS -O3)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND HICMA_OPTIONS -g)
endif()

# Decide where build results should be saved
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
# Set default install path
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Default install path" FORCE
    )
endif()


## Dependencies
# Check for OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    list(APPEND HICMA_OPTIONS ${OpenMP_CXX_FLAGS})
endif()

# Boost
find_package(Boost 1.53 REQUIRED)
list(APPEND HICMA_DEPENDENCIES Boost::headers)

# MKL or other BLAS/LAPACK libraries (default is other)
option(USE_MKL "Use Intel Math Kernel Libraries (MKL)" OFF)
if(${USE_MKL})
    list(APPEND HICMA_DEFINITIONS USE_MKL)
    set(BLA_VENDOR Intel10_64lp)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    list(APPEND HICMA_DEPENDENCIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
    find_package(CBLAS REQUIRED)
    find_package(LAPACKE REQUIRED)
    list(APPEND HICMA_DEPENDENCIES ${CBLAS_LIBRARIES} ${LAPACKE_LIBRARIES})
endif()

# Import external dependency loading script
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(find_or_download)
# Open Multi-Method depdency
find_or_download(PACKAGE YOMM2 INSTALL_WITH_HiCMA)
list(APPEND HICMA_DEPENDENCIES YOMM2::yomm2)


## Main source tree
add_subdirectory(src)


## Testing code
option(BUILD_TESTS "Build unit and other tests" ON)
if(${BUILD_TESTS})
    find_or_download(PACKAGE GTest)
    # Enable test for googletest
    enable_testing()
    add_subdirectory(test)
endif()

## Install instruction
# Create version file for cmake package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  HiCMAConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/HiCMAConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/HiCMA
)
# Create config file of yomm2
# No public dependencies so a Config file with the targets is enough
install(EXPORT HiCMATargets
  FILE HiCMAConfig.cmake
  NAMESPACE HiCMA::
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/HiCMA
)
# Copy headers to install directory
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/hicma
    DESTINATION include
)
