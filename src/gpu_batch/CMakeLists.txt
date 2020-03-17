add_library(gpu_batch OBJECT
    batch.cpp
)
target_include_directories(gpu_batch
PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_include_directories(gpu_batch
SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/yomm11-src/include>
)
target_compile_options(gpu_batch PUBLIC ${HICMA_OPTIONS})
target_compile_definitions(gpu_batch PUBLIC ${HICMA_DEFINITIONS})