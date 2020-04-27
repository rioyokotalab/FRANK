add_library(hicma STATIC functions.cpp)
add_subdirectory(classes)
add_subdirectory(gpu_batch)
add_subdirectory(operations)
add_subdirectory(util)
target_include_directories(hicma
PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(hicma PRIVATE ${HICMA_DEFINITIONS})
target_compile_features(hicma PRIVATE ${HICMA_FEATURES})
target_compile_options(hicma PRIVATE ${HICMA_OPTIONS})
target_link_libraries(hicma PRIVATE ${HICMA_DEPENDENCIES})

# Install instructions
install(TARGETS hicma EXPORT HiCMATargets DESTINATION lib)
