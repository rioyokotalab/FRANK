add_library(FRANK SHARED functions.cpp)
add_subdirectory(classes)
add_subdirectory(operations)
add_subdirectory(util)
target_include_directories(FRANK
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(FRANK PRIVATE ${FRANK_DEFINITIONS})
target_compile_features(FRANK PRIVATE ${FRANK_FEATURES})
target_compile_options(FRANK PRIVATE ${FRANK_OPTIONS})
target_link_libraries(FRANK PRIVATE ${FRANK_DEPENDENCIES})

# Install instructions
install(TARGETS FRANK EXPORT FRANKTargets DESTINATION lib)
