add_library(classes OBJECT
    index_range.cpp
    node.cpp
    node_proxy.cpp
    dense.cpp
    low_rank.cpp
    low_rank_shared.cpp
    hierarchical.cpp
    no_copy_split.cpp
    uniform_hierarchical.cpp
)
target_include_directories(classes
PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/yomm11-src/include>
)
target_compile_options(classes PUBLIC ${HICMA_OPTIONS})
target_compile_definitions(classes PUBLIC ${HICMA_DEFINITIONS})