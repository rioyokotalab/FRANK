list(APPEND GTEST_TESTS 
     "dense" "low_rank" "hierarchical"
     "kernel" #"initializer"
     "dense_arithmetic" "dense_gemm" "dense_getrf" "dense_trsm" "dense_svd" "dense_id" "dense_qr"
     "h_lu"
     "rsvd"
     #"template"
    )
foreach(TEST ${GTEST_TESTS})
  add_executable(${TEST}_test ${TEST}_test.cpp)
  target_link_libraries(${TEST}_test hicma GTest::gtest_main stdc++ m dl)
  set_target_properties (${TEST}_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_UNIT_TEST_OUTPUT_DIRECTORY})
  add_test(NAME ${TEST}_test COMMAND ${TEST}_test WORKING_DIRECTORY ${CMAKE_UNIT_TEST_OUTPUT_DIRECTORY})
  target_compile_definitions(${TEST}_test PRIVATE ${HICMA_DEFINITIONS})
  target_compile_features(${TEST}_test PRIVATE ${HICMA_FEATURES})
  target_compile_options(${TEST}_test PRIVATE ${HICMA_OPTIONS})
endforeach()

# TODO maybe change?
add_custom_command(TARGET dense_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/test_data ${CMAKE_UNIT_TEST_OUTPUT_DIRECTORY}/test_data)
