add_library(util OBJECT
    counter.cpp
    experiment_setup.cpp
    get_memory_usage.cpp
    l2_error.cpp
    omm_error_handler.cpp
    print.cpp
    timer.cpp
)
target_include_directories(util
PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
foreach(DEPENDENCY_INCLUDE ${DEPDENCY_INCLUDES})
    target_include_directories(
        util SYSTEM PUBLIC $<BUILD_INTERFACE:${DEPENDENCY_INCLUDE}>)
endforeach()
target_compile_definitions(util PUBLIC ${HICMA_DEFINITIONS})
target_compile_features(util PRIVATE ${HICMA_FEATURES})
target_compile_options(util PRIVATE ${HICMA_OPTIONS})
