add_library(hicma
    functions.cpp
)
add_subdirectory(classes)
add_subdirectory(gpu_batch)
add_subdirectory(operations)
add_subdirectory(util)
target_include_directories(hicma
PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_definitions(hicma PUBLIC ${HICMA_DEFINITIONS})
target_compile_features(hicma PRIVATE ${HICMA_FEATURES})
target_compile_options(hicma PRIVATE ${HICMA_OPTIONS})
target_link_libraries(hicma ${HICMA_DEPENDENCIES})

# Install instructions
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/hicma
    DESTINATION include
)
install(
    TARGETS hicma
    EXPORT hicma
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(
    EXPORT hicma
    DESTINATION cmake
)
