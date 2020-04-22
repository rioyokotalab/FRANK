add_library(gpu_batch OBJECT
    batch.cpp
)
target_include_directories(gpu_batch
PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
foreach(DEPENDENCY_INCLUDE ${DEPDENCY_INCLUDES})
    target_include_directories(
        gpu_batch SYSTEM PUBLIC $<BUILD_INTERFACE:${DEPENDENCY_INCLUDE}>)
endforeach()
target_compile_definitions(gpu_batch PUBLIC ${HICMA_DEFINITIONS})
target_compile_features(gpu_batch PRIVATE ${HICMA_FEATURES})
target_compile_options(gpu_batch PRIVATE ${HICMA_OPTIONS})
