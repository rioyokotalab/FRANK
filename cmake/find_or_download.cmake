function(find_or_download PACKAGE)
    set(options INSTALL_WITH_HiCMA)
    cmake_parse_arguments(find_or_download
        "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
    )
    find_package(${PACKAGE} QUIET)
    if(find_or_download_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "Unparsed arguments: ${find_or_download_UNPARSED_ARGUMENTS}.\n"
            "Ensure that correct arguments are passed to find_or_download!"
        )
    endif()
    if(${${PACKAGE}_FOUND})
        message("Found dependency \"${PACKAGE}\" installed in system.")
    else()
        message(STATUS "Package \"${PACKAGE}\" not found in system.")
        message(STATUS
            "Downloading dependency \"${PACKAGE}\" and building from source."
        )

        if(${find_or_download_INSTALL_WITH_HiCMA})
            set(DEPENDENCY_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
        else()
            set(DEPENDENCY_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/dependencies)
        endif()
        # Prepare download instructions for dependency
        configure_file(
            ${CMAKE_SOURCE_DIR}/cmake/${PACKAGE}_download.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE}-download/CMakeLists.txt
        )

        # Download dependency
        execute_process(
            COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE}-download
            OUTPUT_QUIET
        )
        if(result)
            message(FATAL_ERROR
                "Download of dependency ${PACKAGE} failed: ${result}."
            )
        endif()

        # Build dependency
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE}-download
        )
        if(result)
            message(FATAL_ERROR
                "Build of dependency ${PACKAGE} failed: ${result}."
            )
        endif()

        # Update search path and use regular find_package to add dependency
        # TODO Use same directory here as for configure_file up there and inside
        # download instructions!
        list(APPEND CMAKE_PREFIX_PATH "${DEPENDENCY_INSTALL_PREFIX}")
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
        find_package(${PACKAGE} NO_MODULE REQUIRED)
    endif()
endfunction()
