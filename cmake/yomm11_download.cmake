cmake_minimum_required(VERSION 3.5)
include(ExternalProject)
ExternalProject_Add(
    yomm11
    GIT_REPOSITORY    https://github.com/jll63/yomm11.git
    GIT_TAG           master
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/yomm11-src"
    BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/yomm11-build"
    CMAKE_ARGS        -DCMAKE_BUILD_TYPE=Release
    BUILD_COMMAND     make yomm11
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)