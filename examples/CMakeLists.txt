# Legacy execution files
list(
  APPEND EXECUTABLES
  "lr_add"
  "blocked_mgs_blr_qr"
  "blocked_householder_blr_qr"
  "tiled_householder_blr_qr"
  "blocked_mgs_h_qr"
  "h_lu"
  "Hmatrix_to_json"
)
foreach(EXECUTABLE ${EXECUTABLES})
  add_executable(${EXECUTABLE} ${EXECUTABLE}.cpp)
  target_link_libraries(${EXECUTABLE} FRANK stdc++ m dl)
  target_compile_definitions(${EXECUTABLE} PRIVATE ${FRANK_DEFINITIONS})
  target_compile_features(${EXECUTABLE} PRIVATE ${FRANK_FEATURES})
  target_compile_options(${EXECUTABLE} PRIVATE ${FRANK_OPTIONS})
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
    target_compile_definitions(${EXECUTABLE} PRIVATE ${FRANK_DEFINITIONS})
    target_compile_features(${EXECUTABLE} PRIVATE ${FRANK_FEATURES})
    target_compile_options(${EXECUTABLE} PRIVATE ${FRANK_OPTIONS} ${OpenMP_CXX_FLAGS})
    target_link_libraries(${EXECUTABLE} FRANK stdc++ m dl ${OpenMP_LIB_NAMES})
  endforeach()
endif()
