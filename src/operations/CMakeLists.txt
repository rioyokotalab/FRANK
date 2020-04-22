add_library(operations OBJECT
    arithmetic/addition.cpp
    arithmetic/multiplication.cpp
    arithmetic/subtraction.cpp
    BLAS/gemm.cpp
    BLAS/trmm.cpp
    BLAS/trsm.cpp
    LAPACK/geqrt.cpp
    LAPACK/geqp3.cpp
    LAPACK/getrf.cpp
    LAPACK/id.cpp
    LAPACK/larfb.cpp
    LAPACK/latms.cpp
    LAPACK/qr.cpp
    LAPACK/svd.cpp
    LAPACK/tpmqrt.cpp
    LAPACK/tpqrt.cpp
    misc/get_dim.cpp
    misc/misc.cpp
    misc/norm.cpp
    misc/transpose.cpp
    randomized/rid.cpp
    randomized/rsvd.cpp
)
target_include_directories(operations
PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
foreach(DEPENDENCY_INCLUDE ${DEPDENCY_INCLUDES})
    target_include_directories(
        operations SYSTEM PUBLIC $<BUILD_INTERFACE:${DEPENDENCY_INCLUDE}>)
endforeach()
target_compile_definitions(operations PUBLIC ${HICMA_DEFINITIONS})
target_compile_features(operations PRIVATE ${HICMA_FEATURES})
target_compile_options(operations PRIVATE ${HICMA_OPTIONS})
