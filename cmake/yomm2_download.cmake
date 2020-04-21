cmake_minimum_required(VERSION 3.5)
include(ExternalProject)
project(yomm2)
ExternalProject_Add(yomm2
    GIT_REPOSITORY      https://github.com/jll63/yomm2.git
    GIT_TAG             master
    SOURCE_DIR          "${CMAKE_CURRENT_BINARY_DIR}/yomm2-src"
    BINARY_DIR          "${CMAKE_CURRENT_BINARY_DIR}/yomm2-build"
    CONFIGURE_COMMAND   ""
    BUILD_COMMAND       ""
    INSTALL_COMMAND     ""
    TEST_COMMAND        ""
)