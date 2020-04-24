cmake_minimum_required(VERSION 3.5)
include(ExternalProject)
project(yomm2)
ExternalProject_Add(yomm2
    GIT_REPOSITORY      https://github.com/jll63/yomm2.git
    GIT_TAG             master
    SOURCE_DIR          "${CMAKE_SOURCE_DIR}/dependencies/yomm2"
    BINARY_DIR          "${CMAKE_CURRENT_BINARY_DIR}/yomm2-build"
    INSTALL_DIR         ${CMAKE_SOURCE_DIR}/dependencies
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                        -DCMAKE_BUILD_TYPE=Release
)