list(APPEND GTEST_TESTS "dense" "id")
foreach(TEST ${GTEST_TESTS})
  add_executable(${TEST}_test ${TEST}_test.cpp)
  target_link_libraries(${TEST}_test hicma GTest::gtest_main stdc++ m dl)
  add_test(NAME ${TEST}_test COMMAND ${TEST}_test)
  target_compile_definitions(${TEST}_test PRIVATE ${HICMA_DEFINITIONS})
  target_compile_features(${TEST}_test PRIVATE ${HICMA_FEATURES})
  target_compile_options(${TEST}_test PRIVATE ${HICMA_OPTIONS})
endforeach()

# Legacy execution files
list(
  APPEND EXECUTABLES
  "lr_add"
  "block_qr"
  "tiled_qr"
  "tlr_qr"
  "blr_qr"
  "h_lu"
  "h_qr"
  "rsvd"
  # "blr-lu-exp3d"
  # "kernel-test"
)
if(OpenMP_FOUND)
  list(APPEND EXECUTABLES "forkjoin_tlr_qr")
  list(APPEND EXECUTABLES "forkjoin_blr_qr")
  list(APPEND EXECUTABLES "taskbased_tlr_qr")
endif()
foreach(EXECUTABLE ${EXECUTABLES})
  add_executable(${EXECUTABLE} ${EXECUTABLE}.cpp)
  target_link_libraries(${EXECUTABLE} hicma stdc++ m dl)
  target_link_options(${EXECUTABLE} PRIVATE ${HICMA_LFLAGS})
  target_compile_definitions(${EXECUTABLE} PRIVATE ${HICMA_DEFINITIONS})
  target_compile_features(${EXECUTABLE} PRIVATE ${HICMA_FEATURES})
  target_compile_options(${EXECUTABLE} PRIVATE ${HICMA_OPTIONS})
endforeach()
