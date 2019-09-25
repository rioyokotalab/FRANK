# Below code is directly adapted from the google/googletest repo
# with minor changes
macro(add_external_dependency DEPENDENCY)
    # Download and unpack dependency at configure time.
    # Assumes there is a file called <DEPENDENCY>_download.cmake from which
    # to read the download settings.
    # Googletest provides such a file as CMakeLists.txt.in.
    configure_file(
        ${DEPENDENCY}_download.cmake
        ${DEPENDENCY}-download/CMakeLists.txt
    )
    # Configure step
    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DEPENDENCY}-download
    )
    if(result)
        message(FATAL_ERROR "CMake step for ${DEPENDENCY} failed: ${result}")
    endif()
    # Build step
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DEPENDENCY}-download
    )
    if(result)
        message(FATAL_ERROR "Build step for ${DEPENDENCY} failed: ${result}")
    endif()

    # Add dependency directly to build. This defines the compilation targets.
    add_subdirectory(
        ${CMAKE_CURRENT_BINARY_DIR}/${DEPENDENCY}-src
        ${CMAKE_CURRENT_BINARY_DIR}/${DEPENDENCY}-build
        EXCLUDE_FROM_ALL
    )
endmacro()