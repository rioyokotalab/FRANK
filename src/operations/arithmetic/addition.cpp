#include "hicma/operations/arithmetic.h"

#include "hicma/definitions.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/operations/arithmetic.h"
#include "hicma/operations/misc.h"
#include "hicma/util/global_key_value.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>


namespace hicma
{

declare_method(
  Matrix&, addition_omm,
  (virtual_<Matrix&>, virtual_<const Matrix&>)
)

Matrix& operator+=(Matrix& A, const Matrix& B) { return addition_omm(A, B); }

define_method(Matrix&, addition_omm, (Dense& A, const Dense& B)) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) += B(i, j);
    }
  }
  return A;
}

define_method(Matrix&, addition_omm, (Dense& A, const LowRank& B)) {
  gemm(gemm(B.U, B.S), B.V, A, 1, 1);
  return A;
}

define_method(Matrix&, addition_omm, (Hierarchical& A, const Hierarchical& B)) {
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) += B(i, j);
    }
  }
  return A;
}

define_method(Matrix&, addition_omm, (Hierarchical& A, const LowRank& B)) {
  const Hierarchical BH = split(B, A.dim[0], A.dim[1]);
  A += BH;
  return A;
}

void naive_addition(LowRank& A, const LowRank& B) {
  //Truncate and Recompress if rank > min(nrow, ncol)
  if (A.rank+B.rank >= std::min(A.dim[0], A.dim[1])) {
    A = LowRank(Dense(A) + Dense(B), A.rank);
  } else {
    Hierarchical U_merge(1, 2);
    U_merge[0] = std::move(A.U);
    U_merge[1] = shallow_copy(B.U);
    Hierarchical V_merge(2, 1);
    V_merge[0] = std::move(A.V);
    V_merge[1] = shallow_copy(B.V);
    Hierarchical S_merge(2, 2);
    S_merge(0, 0) = std::move(A.S);
    S_merge(0, 1) = Dense(A.rank, B.rank);
    S_merge(1, 0) = Dense(B.rank, A.rank);
    S_merge(1, 1) = B.S.shallow_copy();
    A.rank += B.rank;
    A.U = Dense(U_merge);
    A.S = Dense(S_merge);
    A.V = Dense(V_merge);
  }
}

// Rounded Addition with SVD
// See Bebendorf HMatrix Book p16 for reference
void rounded_addition(LowRank& A, const LowRank& B) {
  //Concat U bases
  Dense Uc(get_n_rows(A.U), A.S.dim[1]+B.S.dim[1]);
  const IndexRange U_row_range(0, Uc.dim[0]);
  const IndexRange U_col_range(0, Uc.dim[1]);
  auto Uc_splits = Uc.split({ U_row_range }, U_col_range.split_at(A.S.dim[1]), false);
  gemm(A.U, A.S, Uc_splits[0], 1, 0);
  gemm(B.U, B.S, Uc_splits[1], 1, 0);
  Dense Qu(Uc.dim[0], std::min(Uc.dim[0], Uc.dim[1]));
  Dense Ru(std::min(Uc.dim[0], Uc.dim[1]), Uc.dim[1]);
  qr(Uc, Qu, Ru);

  //Concat V bases
  Dense VcT(A.V.dim[0]+B.V.dim[0], get_n_cols(A.V));
  const IndexRange V_row_range(0, VcT.dim[0]);
  const IndexRange V_col_range(0, VcT.dim[1]);
  auto VcT_splits = VcT.split(V_row_range.split_at(A.V.dim[0]), { V_col_range }, false);
  A.V.copy_to(VcT_splits[0]);
  B.V.copy_to(VcT_splits[1]);
  Dense RvT(VcT.dim[0], std::min(VcT.dim[0], VcT.dim[1]));
  Dense QvT(std::min(VcT.dim[0], VcT.dim[1]), VcT.dim[1]);
  rq(VcT, RvT, QvT);

  //SVD and truncate
  Dense RuRvT = gemm(Ru, RvT);
  Dense RRU, RRS, RRV;
  std::tie(RRU, RRS, RRV) = svd(RuRvT);
  // Find truncation rank if needed
  const bool use_eps = (A.eps != 0);
  if(use_eps) A.rank = find_svd_truncation_rank(RRS, A.eps);
  // Truncate
  A.S = resize(RRS, A.rank, A.rank);
  A.U = gemm(Qu, resize(RRU, RRU.dim[0], A.rank));
  A.V = gemm(resize(RRV, A.rank, RRV.dim[1]), QvT);
}

// Fast rounded addition that exploits existing orthogonality in U and V matrices
// See Bebendorf HMatrix Book p17 for reference
// Note that this method only works when both A.U and A.V have orthonormal columns
// Which is not always the case in general
void fast_rounded_addition(LowRank& A, const LowRank& B) {
  // Fallback to rounded addition if fixed accuracy compression is used
  // Since A.V does not have orthonormal columns
  if(A.eps != 0.) {
    // TODO consider orthogonalize A.V? Impact to overall cost?
    rounded_addition(A, B);
    return;
  }
  // Form U bases
  Dense Zu = gemm(A.U, B.U, 1, true, false);
  Dense Yu(B.U);
  gemm(A.U, Zu, Yu, -1, 1);
  Dense Qu(Yu.dim[0], std::min(Yu.dim[0], Yu.dim[1]));
  Dense Ru(std::min(Yu.dim[0], Yu.dim[1]), Yu.dim[1]);
  qr(Yu, Qu, Ru);
  // Uc = [A.U  Qu]
  Dense Uc(A.U.dim[0], A.U.dim[1]+Qu.dim[1]);
  const IndexRange U_row_range(0, Uc.dim[0]);
  const IndexRange U_col_range(0, Uc.dim[1]);
  auto Uc_splits = Uc.split({ U_row_range }, U_col_range.split_at(A.U.dim[1]), false);
  A.U.copy_to(Uc_splits[0]);
  Qu.copy_to(Uc_splits[1]);

  // Form V bases
  Dense ZvT = gemm(B.V, A.V, 1, false, true);
  Dense YvT(B.V);
  gemm(ZvT, A.V, YvT, -1, 1);
  Dense RvT(YvT.dim[0], std::min(YvT.dim[0], YvT.dim[1]));
  Dense QvT(std::min(YvT.dim[0], YvT.dim[1]), YvT.dim[1]);
  rq(YvT, RvT, QvT);
  // Vc^T = [A.V  Qv]^T
  Dense VcT(A.V.dim[0]+QvT.dim[0], A.V.dim[1]);
  const IndexRange V_row_range(0, VcT.dim[0]);
  const IndexRange V_col_range(0, VcT.dim[1]);
  auto VcT_splits = VcT.split(V_row_range.split_at(A.V.dim[0]), { V_col_range }, false);
  A.V.copy_to(VcT_splits[0]);
  QvT.copy_to(VcT_splits[1]);

  Dense M(A.S.dim[0]+B.S.dim[0], A.S.dim[1]+B.S.dim[1]);
  const IndexRange M_row_range(0, M.dim[0]);
  const IndexRange M_col_range(0, M.dim[1]);
  auto M_splits = M.split(M_row_range.split_at(A.S.dim[0]), M_col_range.split_at(A.S.dim[1]), false);
  Dense ZuSb = gemm(Zu, B.S);
  Dense RuSb = gemm(Ru, B.S);
  A.S.copy_to(M_splits[0]);
  gemm(ZuSb, ZvT, M_splits[0], 1, 1); // M(0, 0) = A.S + Zu*B.S*ZvT
  gemm(ZuSb, RvT, M_splits[1], 1, 0); // M(0, 1) = Zu*B.S*RvT
  gemm(RuSb, ZvT, M_splits[2], 1, 0); // M(1, 0) = Ru*B.S*ZvT
  gemm(RuSb, RvT, M_splits[3], 1, 0); // M(1, 1) = Ru*B.S*RvT

  // SVD and truncate
  Dense Um, Sm, VmT;
  std::tie(Um, Sm, VmT) = svd(M);
  A.S = resize(Sm, A.rank, A.rank);
  A.U = gemm(Uc, resize(Um, Um.dim[0], A.rank));
  A.V = gemm(resize(VmT, A.rank, VmT.dim[1]), VcT);  
}

define_method(Matrix&, addition_omm, (LowRank& A, const LowRank& B)) {
  assert(A.dim[0] == B.dim[0]);
  assert(A.dim[1] == B.dim[1]);
  if (getGlobalValue("HICMA_LRA") == "naive") {
    naive_addition(A, B);
  } else if (getGlobalValue("HICMA_LRA") == "rounded_addition") {
    rounded_addition(A, B);
  } else {
    // TODO consider changing default to rounded_addition?
    fast_rounded_addition(A, B);
  }
  return A;
}

define_method(Matrix&, addition_omm, (Matrix& A, const Matrix& B)) {
  omm_error_handler("operator+=", {A, B}, __FILE__, __LINE__);
  std::abort();
}

Dense operator+(const Dense& A, const Dense& B) {
  Dense out(A.dim[0], A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      out(i, j) = A(i, j) + B(i, j);
    }
  }
  return out;
}

} // namespace hicma
