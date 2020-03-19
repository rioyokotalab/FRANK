add_library(operations OBJECT
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
    misc/addition.cpp
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
target_include_directories(operations
SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/yomm2-src/include>
)
target_compile_options(operations PRIVATE ${HICMA_OPTIONS})
target_compile_definitions(operations PUBLIC ${HICMA_DEFINITIONS})
