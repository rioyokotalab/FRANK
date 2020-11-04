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
foreach(EXECUTABLE ${EXECUTABLES})
    add_executable(${EXECUTABLE} ${EXECUTABLE}.cpp)
    target_link_libraries(${EXECUTABLE} hicma stdc++ m dl)
    target_compile_definitions(${EXECUTABLE} PRIVATE ${HICMA_DEFINITIONS})
    target_compile_features(${EXECUTABLE} PRIVATE ${HICMA_FEATURES})
    target_compile_options(${EXECUTABLE} PRIVATE ${HICMA_OPTIONS})
endforeach()
