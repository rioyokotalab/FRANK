add_external_dependency(googletest)

# Class tests
add_executable(dense_test dense_test.cpp)
target_link_libraries(dense_test gtest_main hicma)
add_test(NAME dense_test COMMAND dense_test)

# add_executable(low_rank_test low_rank_test.cpp)
# target_link_libraries(low_rank_test gtest_main hicma)
# add_test(NAME low_rank_test COMMAND low_rank_test)

# add_executable(hierarchical_test hierarchical_test.cpp)
# target_link_libraries(hierarchical_test gtest_main hicma)
# add_test(NAME hierarchical_test COMMAND hierarchical_test)

# Legacy execution files
add_executable(lr_add lr_add.cpp)
target_link_libraries(lr_add hicma)

add_executable(block_lu block_lu.cpp)
target_link_libraries(block_lu hicma)

add_executable(blr_lu blr_lu.cpp)
target_link_libraries(blr_lu hicma)

add_executable(h_lu h_lu.cpp)
target_link_libraries(h_lu hicma)

add_executable(rsvd rsvd.cpp)
target_link_libraries(rsvd hicma)