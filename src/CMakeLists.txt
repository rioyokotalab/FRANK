add_library(hicma SHARED functions.cpp)
add_subdirectory(classes)
add_subdirectory(gpu_batch)
add_subdirectory(operations)
add_subdirectory(starpu)
add_subdirectory(util)
target_include_directories(hicma
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE ${PROJECT_SOURCE_DIR}/private_include ${starpu_INCLUDE_DIRS}
)
target_compile_definitions(hicma PRIVATE ${HICMA_DEFINITIONS})
target_compile_features(hicma PRIVATE ${HICMA_FEATURES})
target_compile_options(hicma PRIVATE ${HICMA_OPTIONS})
target_link_libraries(hicma PRIVATE ${HICMA_DEPENDENCIES})
target_link_libraries(hicma PRIVATE ${starpu_LIBRARIES})

# Install instructions
install(TARGETS hicma EXPORT HiCMATargets DESTINATION lib)
