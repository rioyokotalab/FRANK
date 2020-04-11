add_library(classes OBJECT
    index_range.cpp
    node_proxy.cpp
    dense.cpp
    dense_view.cpp
    low_rank.cpp
    low_rank_shared.cpp
    low_rank_view.cpp
    hierarchical.cpp
    no_copy_split.cpp
    uniform_hierarchical.cpp
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
