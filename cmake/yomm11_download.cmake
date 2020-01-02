cmake_minimum_required(VERSION 3.5)
include(ExternalProject)
project(yomm11)
ExternalProject_Add(yomm11
    GIT_REPOSITORY      https://github.com/jll63/yomm11.git
    GIT_TAG             master
    SOURCE_DIR          "${CMAKE_CURRENT_BINARY_DIR}/yomm11-src"
    BINARY_DIR          "${CMAKE_CURRENT_BINARY_DIR}/yomm11-build"
    CONFIGURE_COMMAND   ""
    BUILD_COMMAND       ""
    INSTALL_COMMAND     ""
    TEST_COMMAND        ""
)