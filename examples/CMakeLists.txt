# Legacy execution files
list(
  APPEND EXECUTABLES
  "lr_add"
  "tiled_householder_dense_qr"
  "blr_compress"
  "blocked_mgs_blr_qr"
  "blocked_householder_blr_qr"
  "tiled_householder_blr_qr"
  "blocked_mgs_h_qr"
  "h_lu"
  "kernel-test"
)
foreach(EXECUTABLE ${EXECUTABLES})
  add_executable(${EXECUTABLE} ${EXECUTABLE}.cpp)
  target_link_libraries(${EXECUTABLE} hicma stdc++ m dl)
  target_compile_definitions(${EXECUTABLE} PRIVATE ${HICMA_DEFINITIONS})
  target_compile_features(${EXECUTABLE} PRIVATE ${HICMA_FEATURES})
  target_compile_options(${EXECUTABLE} PRIVATE ${HICMA_OPTIONS})
endforeach()

# OpenMP multithreaded execution files
if(OpenMP_FOUND)
  list(
    APPEND MULTITHREADED_EXECUTABLES
    "forkjoin_blocked_mgs_blr_qr"
    "forkjoin_blocked_householder_blr_qr"
    "forkjoin_tiled_householder_blr_qr"
    "taskbased_tiled_householder_blr_qr"
  )
foreach(EXECUTABLE ${MULTITHREADED_EXECUTABLES})
    add_executable(${EXECUTABLE} ${EXECUTABLE}.cpp)
    target_link_libraries(${EXECUTABLE} hicma stdc++ m dl ${OpenMP_LIB_NAMES})
    target_compile_definitions(${EXECUTABLE} PRIVATE ${HICMA_DEFINITIONS})
    target_compile_features(${EXECUTABLE} PRIVATE ${HICMA_FEATURES})
    target_compile_options(${EXECUTABLE} PRIVATE ${HICMA_OPTIONS})
  endforeach()
endif()
