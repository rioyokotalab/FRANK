cmake_minimum_required(VERSION 3.6)

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

# Use the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)
# When building, don't use the install RPATH (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# the RPATH to be used when installing, but only if it's not a system directory
list(
  FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
  "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir
)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")


## Dependencies
# Standard library dependency
list(APPEND HICMA_DEPENDENCIES stdc++ m dl)
# Import external dependency loading script
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(find_or_download)

# Check for OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
  list(APPEND HICMA_OPTIONS ${OpenMP_CXX_FLAGS})
endif()

# StarPU
find_or_download(starpu PKG_CONFIG VERSION 1.3 INSTALL_WITH_HiCMA)
link_directories(${starpu_LIBRARY_DIRS})

# Boost
find_or_download(nlohmann_json)
list(APPEND HICMA_DEPENDENCIES nlohmann_json::nlohmann_json)

# MKL or other BLAS/LAPACK libraries (default is other)
option(USE_MKL "Use Intel Math Kernel Libraries (MKL)" OFF)
if(${USE_MKL})
  list(APPEND HICMA_DEFINITIONS USE_MKL)
  set(BLA_VENDOR Intel10_64lp)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  list(APPEND HICMA_DEPENDENCIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  find_or_download(CBLAS)
  find_or_download(LAPACKE)
  list(APPEND HICMA_DEPENDENCIES cblas lapacke)
endif()

# Open Multi-Method depdency
find_or_download(YOMM2)
list(APPEND HICMA_DEPENDENCIES YOMM2::yomm2)

# stars-H dependency
# find_or_download(starsh)
# list(APPEND HICMA_DEPENDENCIES starsh::starsh)

## Main source tree
add_subdirectory(src)


## Add documentation
option(BUILD_DOCS "Build documentation using doxygen" OFF)
if(${BUILD_DOCS})
  add_subdirectory(docs)
endif()


## Testing code
option(BUILD_TESTS "Build unit and other tests" ON)
if(${BUILD_TESTS})
  find_or_download(GTest)
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
# Create config file of HiCMA
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
