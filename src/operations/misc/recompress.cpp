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

declare_method(void, recompress_omm,
  (
    virtual_<const Matrix&>, virtual_<const Matrix&>,
    virtual_<const Matrix&>, virtual_<const Matrix&>, LowRank&,
    double, double, bool, bool
  )
)

void recompress(
  const Matrix& A, const Matrix& B, LowRank& C,
  double alpha, double beta, bool TransA, bool TransB
) {
  recompress_omm(C.U, C.V, A, B, C, alpha, beta, TransA, TransB);
}

define_method(
  void, recompress_omm,
  (
    const Hierarchical&, const Hierarchical&,
    const Hierarchical& A, const LowRank& B, LowRank& C,
    double alpha, double beta, bool TransA, bool TransB
  )
) {
  assert(is_shared(B.V, C.V));
  Hierarchical BH = split(B, A.dim[1], 1);
  Hierarchical CH = split(C, A.dim[0], 1);
  gemm(A, BH, CH, alpha, beta, TransA, TransB);
  // TODO The following needs to be done like the all-dense recompress, as in we
  // need to combine this for all blocks in the block row
  Hierarchical S_colH(A.dim[0], 1);
  for (int64_t i=0; i<A.dim[0]; ++i) {
    LowRank Ci(std::move(CH[i]));
    S_colH[i] = std::move(Ci.S);
    CH[i] = std::move(Ci);
  }
  Dense S_col(S_colH);
  Dense new_trans_mats(S_col.dim[0], S_col.dim[1]);
  Dense new_S(C.S.dim[0], C.S.dim[1]);
  qr(S_col, new_trans_mats, new_S);
  C.S = std::move(new_S);
  std::vector<Dense> new_trans_matsH = new_trans_mats.split(A.dim[0], 1, true);
  Hierarchical new_U(A.dim[0], 1);
  for (int64_t i=0; i<A.dim[0]; ++i) {
    LowRank Ci(std::move(CH[i]));
    new_U[i] = NestedBasis(Ci.U, new_trans_matsH[i], true);
  }
  C.U = std::move(new_U);
}

define_method(
  void, recompress_omm,
  (
    const Hierarchical&, const Hierarchical&,
    const LowRank& A, const Hierarchical& B, LowRank& C,
    double alpha, double beta, bool TransA, bool TransB
  )
) {
  assert(is_shared(A.U, C.U));
  Hierarchical AH = split(A, 1, B.dim[0]);
  Hierarchical CH = split(C, 1, B.dim[1]);
  gemm(AH, B, CH, alpha, beta, TransA, TransB);
  // TODO The following needs to be done like the all-dense recompress, as in we
  // need to combine this for all blocks in the block col
  Hierarchical S_rowH(1, B.dim[1]);
  for (int64_t j=0; j<B.dim[1]; ++j) {
    LowRank Cj(std::move(CH[j]));
    S_rowH[j] = std::move(Cj.S);
    CH[j] = std::move(Cj);
  }
  Dense S_row(S_rowH);
  Dense new_trans_mats(S_row.dim[0], S_row.dim[1]);
  Dense new_S(C.S.dim[0], C.S.dim[1]);
  rq(S_row, new_S, new_trans_mats);
  C.S = std::move(new_S);
  std::vector<Dense> new_trans_matsH = new_trans_mats.split(1, B.dim[1], true);
  Hierarchical new_V(1, B.dim[1]);
  for (int64_t j=0; j<B.dim[1]; ++j) {
    LowRank Cj(std::move(CH[j]));
    new_V[j] = NestedBasis(Cj.V, new_trans_matsH[j], false);
  }
  C.V = std::move(new_V);
}

define_method(
  void, recompress_omm,
  (
    const Dense&, const Dense&,
    const Hierarchical& A, const LowRank& B, LowRank& C,
    double alpha, double beta, bool TransA, bool
  )
) {
  MatrixProxy AxBU = gemm(A, B.U, alpha, TransA, false);
  LowRank AxB(AxBU, B.S, B.V);
  C.S *= beta;
  C += AxB;
}

define_method(
  void, recompress_omm,
  (
    const Dense&, const Dense&,
    const LowRank& A, const Hierarchical& B, LowRank& C,
    double alpha, double beta, bool, bool TransB
  )
) {
  MatrixProxy AVxB = gemm(A.V, B, alpha, false, TransB);
  LowRank AxB(A.U, A.S, AVxB);
  C.S *= beta;
  C += AxB;
}

define_method(
  void, recompress_omm,
  (
    const Matrix& CU, const Matrix& CV,
    const Matrix& A, const Matrix& B, LowRank&,
    double, double, bool, bool
  )
) {
  omm_error_handler("recompress", {CU, CV, A, B}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
