list(
  APPEND GTEST_TESTS
  "dense"
  "dense_arithmetic"
  "dense_gemm"
  "dense_getrf"
  "dense_trsm"
  "dense_svd"
  "dense_id"
  "dense_qr"
  "rsvd"
  "lowrank"
  "blr_fixed_rank"
  "blr_fixed_acc"
  "hierarchical_fixed_rank"
  "hierarchical_fixed_acc"
  "h_lu"
)
foreach(TEST ${GTEST_TESTS})
  add_executable(${TEST}_test ${TEST}_test.cpp)
  target_link_libraries(${TEST}_test hicma GTest::gtest_main stdc++ m dl)
  set_target_properties (${TEST}_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_UNIT_TEST_OUTPUT_DIRECTORY})
  add_test(NAME ${TEST}_test COMMAND ${TEST}_test)
  target_compile_definitions(${TEST}_test PRIVATE ${HICMA_DEFINITIONS})
  target_compile_features(${TEST}_test PRIVATE ${HICMA_FEATURES})
  target_compile_options(${TEST}_test PRIVATE ${HICMA_OPTIONS})
endforeach()
