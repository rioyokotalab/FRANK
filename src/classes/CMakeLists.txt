add_library(classes OBJECT
    node_proxy.cpp
    dense.cpp
    hierarchical.cpp
    low_rank.cpp
    low_rank_shared.cpp
    no_copy_split.cpp
    uniform_hierarchical.cpp
    initialization_helpers/cluster_tree.cpp
    initialization_helpers/matrix_initializer.cpp
)
target_include_directories(classes
PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
foreach(DEPENDENCY_INCLUDE ${DEPENDENCY_INCLUDES})
    target_include_directories(
        classes SYSTEM PUBLIC $<BUILD_INTERFACE:${DEPENDENCY_INCLUDE}>)
endforeach()
target_compile_definitions(classes PUBLIC ${HICMA_DEFINITIONS})
target_compile_features(classes PRIVATE ${HICMA_FEATURES})
target_compile_options(classes PRIVATE ${HICMA_OPTIONS})
