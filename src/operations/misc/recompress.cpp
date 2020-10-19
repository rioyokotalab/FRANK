#include "hicma/operations/misc.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/pre_scheduler.h"
#include "hicma/operations/arithmetic.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>


namespace hicma
{

declare_method(
  void, recompress_col_omm,
  (virtual_<Matrix&>, virtual_<const Matrix&>, Dense&, const Dense&)
)

void recompress_col(Matrix& AU, const Matrix& BU, Dense& AS, const Dense& BS) {
  recompress_col_omm(AU, BU, AS, BS);
}

define_method(
  void, recompress_col_omm,
  (Dense& AU, const Dense& BU, Dense& AS, const Dense& BS)
) {
  Dense newU(AU.dim[0], AU.dim[1]);
  Dense newS(AS.dim[0], AS.dim[1]);
  add_recompress_col_task(newU, newS, AU, BU, AS, BS);
  AU = std::move(newU);
  AS = std::move(newS);
}

define_method(
  void, recompress_col_omm,
  (Matrix& AU, const Matrix& BU, Dense&, const Dense&)
) {
  omm_error_handler("recompress_col", {AU, BU}, __FILE__, __LINE__);
  std::abort();
}

declare_method(
  void, recompress_row_omm,
  (virtual_<Matrix&>, virtual_<const Matrix&>, Dense&, const Dense&)
)

void recompress_row(Matrix& AV, const Matrix& BV, Dense& AS, const Dense& BS) {
  recompress_row_omm(AV, BV, AS, BS);
}

define_method(
  void, recompress_row_omm,
  (Dense& AV, const Dense& BV, Dense& AS, const Dense& BS)
) {
  Dense newV(AV.dim[0], AV.dim[1]);
  Dense newS(AS.dim[0], AS.dim[1]);
  add_recompress_row_task(newV, newS, AV, BV, AS, BS);
  AV = std::move(newV);
  AS = std::move(newS);
}

define_method(
  void, recompress_row_omm,
  (Matrix& AV, const Matrix& BV, Dense&, const Dense&)
) {
  omm_error_handler("recompress_row", {AV, BV}, __FILE__, __LINE__);
  std::abort();
}

LowRank recombine_col(Hierarchical& A, MatrixProxy& V) {
  // TODO The following needs to be done like the all-dense recompress, as in we
  // need to combine this for all blocks in the block row
  Hierarchical S_colH(A.dim[0], 1);
  for (int64_t i=0; i<A.dim[0]; ++i) {
    LowRank Ai(std::move(A[i]));
    S_colH[i] = std::move(Ai.S);
    A[i] = std::move(Ai);
  }
  Dense S_col(S_colH);
  Dense new_trans_mats(S_col.dim[0], S_col.dim[1]);
  Dense new_S(S_col.dim[1], S_col.dim[1]);
  qr(S_col, new_trans_mats, new_S);
  std::vector<Dense> new_trans_matsH = new_trans_mats.split(A.dim[0], 1, true);
  Hierarchical new_U(A.dim[0], 1);
  for (int64_t i=0; i<A.dim[0]; ++i) {
    LowRank Ai(std::move(A[i]));
    new_U[i] = NestedBasis(Ai.U, new_trans_matsH[i], true);
  }
  return LowRank(std::move(new_U), std::move(new_S), std::move(V));
}

LowRank recombine_row(Hierarchical& A, MatrixProxy& U) {
  // TODO The following needs to be done like the all-dense recompress, as in we
  // need to combine this for all blocks in the block col
  Hierarchical S_rowH(1, A.dim[1]);
  for (int64_t j=0; j<A.dim[1]; ++j) {
    LowRank Aj(std::move(A[j]));
    S_rowH[j] = std::move(Aj.S);
    A[j] = std::move(Aj);
  }
  Dense S_row(S_rowH);
  Dense new_trans_mats(S_row.dim[0], S_row.dim[1]);
  Dense new_S(S_row.dim[0], S_row.dim[0]);
  rq(S_row, new_S, new_trans_mats);
  std::vector<Dense> new_trans_matsH = new_trans_mats.split(1, A.dim[1], true);
  Hierarchical new_V(1, A.dim[1]);
  for (int64_t j=0; j<A.dim[1]; ++j) {
    LowRank Aj(std::move(A[j]));
    new_V[j] = NestedBasis(Aj.V, new_trans_matsH[j], false);
  }
  return LowRank(std::move(U), std::move(new_S), std::move(new_V));
}

} // namespace hicma
