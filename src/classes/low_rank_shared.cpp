#include "hicma/classes/low_rank_shared.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/extension_headers/classes.h"
#include "hicma/operations/BLAS.h"

#include "yorel/yomm2/cute.hpp"

#include <memory>


namespace hicma
{

LowRankShared::LowRankShared(
  const Dense& S, std::shared_ptr<Dense> U, std::shared_ptr<Dense> V
) : U(U), V(V), S(S), dim{U->dim[0], V->dim[1]}, rank(S.dim[0]) {}

define_method(void, fill_dense_from, (const LowRankShared& A, Dense& B)) {
  // TODO exactly the same as the LowRank method. Consider inheritance!
  gemm(gemm(A.U, A.S), A.V, B, 1, 0);
}

} // namespace hicma
