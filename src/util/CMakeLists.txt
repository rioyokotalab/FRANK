add_library(util OBJECT
    counter.cpp
    experiment_setup.cpp
    get_memory_usage.cpp
    l2_error.cpp
    print.cpp
    timer.cpp
)
target_include_directories(util
PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/yomm11-src/include>
)
target_compile_options(util PUBLIC ${HICMA_OPTIONS})
target_compile_definitions(util PUBLIC ${HICMA_DEFINITIONS})