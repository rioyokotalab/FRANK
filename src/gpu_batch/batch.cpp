#include "hicma/gpu_batch/batch.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/operations/BLAS.h"

#include <cstdint>
#include <vector>


namespace hicma
{

std::vector<Dense> vecA;
std::vector<Dense> vecB;
std::vector<Dense*> vecC;
std::vector<MatrixProxy*> vecLR;

void rsvd_push(MatrixProxy& A, Dense& Aij, int64_t rank) {
  A = LowRank(Aij, rank);
}

void gemm_push(const Dense& A, const Dense& B, Dense& C) {
  gemm(A, B, C, false, false, 1, 1);
}

void rsvd_batch() {}

void gemm_batch() {}

} // namespace hicma
