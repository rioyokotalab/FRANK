add_subdirectory(classes)
add_subdirectory(gpu_batch)
add_subdirectory(operations)
add_subdirectory(util)

add_library(hicma
    functions.cpp
    $<TARGET_OBJECTS:classes>
    $<TARGET_OBJECTS:operations>
    $<TARGET_OBJECTS:util>
    $<TARGET_OBJECTS:gpu_batch>
)
target_include_directories(hicma
PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_include_directories(hicma
SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/yomm11-src/include>
)
target_compile_options(hicma PUBLIC ${HICMA_OPTIONS})
target_compile_definitions(hicma PUBLIC ${HICMA_DEFINITIONS})
target_link_libraries(hicma ${HICMA_DEPENDENCIES})

# Install instructions
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/hicma
    DESTINATION include
)
if(${CMAKE_VERSION} VERSION_LESS "3.10.0")
    install(
        FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libyomm11.a
        DESTINATION lib
    )
    install(
        TARGETS hicma
        EXPORT hicma
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
else()
    install(
        TARGETS hicma yomm11
        EXPORT hicma
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()
install(
    EXPORT hicma
    DESTINATION cmake
)