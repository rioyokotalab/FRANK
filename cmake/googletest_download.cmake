cmake_minimum_required(VERSION 2.8.2)
include(ExternalProject)
project(googletest)
ExternalProject_Add(googletest
    GIT_REPOSITORY      https://github.com/google/googletest.git
    GIT_TAG             master
    SOURCE_DIR          "${CMAKE_SOURCE_DIR}/dependencies/googletest"
    BINARY_DIR          "${CMAKE_CURRENT_BINARY_DIR}/googletest-build"
    INSTALL_DIR         ${CMAKE_SOURCE_DIR}/dependencies
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                        -DCMAKE_BUILD_TYPE=Release
)