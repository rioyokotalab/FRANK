#ifndef hicma_operations_h
#define hicma_operations_h

#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/BLAS/trmm.h"
#include "hicma/operations/BLAS/trsm.h"

#include "hicma/operations/LAPACK/geqp3.h"
#include "hicma/operations/LAPACK/geqrt.h"
#include "hicma/operations/LAPACK/getrf.h"
#include "hicma/operations/LAPACK/id.h"
#include "hicma/operations/LAPACK/larfb.h"
#include "hicma/operations/LAPACK/latms.h"
#include "hicma/operations/LAPACK/qr.h"
#include "hicma/operations/LAPACK/svd.h"
#include "hicma/operations/LAPACK/tpmqrt.h"
#include "hicma/operations/LAPACK/tpqrt.h"

#include "hicma/operations/misc/get_dim.h"
#include "hicma/operations/misc/misc.h"
#include "hicma/operations/misc/norm.h"
#include "hicma/operations/misc/transpose.h"

#include "hicma/operations/randomized/rid.h"
#include "hicma/operations/randomized/rsvd.h"

#endif // hicma_operations_h
