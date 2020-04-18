add_library(classes OBJECT
    node_proxy.cpp
    dense.cpp
    hierarchical.cpp
    low_rank.cpp
    low_rank_shared.cpp
    no_copy_split.cpp
    uniform_hierarchical.cpp
    initialization_helpers/cluster_tree.cpp
)
target_include_directories(classes
PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_include_directories(classes
SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/yomm2-src/include>
)
target_compile_options(classes PRIVATE ${HICMA_OPTIONS})
target_compile_definitions(classes PUBLIC ${HICMA_DEFINITIONS})
