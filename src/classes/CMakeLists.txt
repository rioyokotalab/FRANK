target_sources(hicma PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/dense.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hierarchical.cpp
    ${CMAKE_CURRENT_LIST_DIR}/low_rank.cpp
    ${CMAKE_CURRENT_LIST_DIR}/low_rank_shared.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matrix_proxy.cpp
    ${CMAKE_CURRENT_LIST_DIR}/no_copy_split.cpp
    ${CMAKE_CURRENT_LIST_DIR}/uniform_hierarchical.cpp
    ${CMAKE_CURRENT_LIST_DIR}/initialization_helpers/cluster_tree.cpp
    ${CMAKE_CURRENT_LIST_DIR}/initialization_helpers/matrix_initializer.cpp
)
