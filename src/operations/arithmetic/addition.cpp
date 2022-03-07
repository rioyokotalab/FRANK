#include "hicma/operations/arithmetic.h"
#include "hicma/extension_headers/operations.h"
#include "hicma/extension_headers/tuple_types.h"

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
  Hierarchical BH = split(B, A.dim[0], A.dim[1]);
  A += BH;
  return A;
}

declare_method(
  DensePair, merge_col_basis,
  (virtual_<const Matrix&>, virtual_<const Matrix&>)
)

define_method(DensePair, merge_col_basis, (const Dense& Au, const Dense& Bu)) {
  assert(Au.dim[0] == Bu.dim[0]);
  int64_t Arank = Au.dim[1];
  int64_t Brank = Bu.dim[1];
  assert(Arank == Brank);

  Dense InnerU(Arank+Brank, Brank);
  Hierarchical InnerH = split(InnerU, 2, 1);
  gemm(Au, Bu, InnerH[0], 1, 0, true, false);

  Dense Bu_AuAutBu(Bu);
  gemm(Au, InnerH[0], Bu_AuAutBu, -1, 1);

  Dense Q(Au.dim[0], Brank);
  qr(Bu_AuAutBu, Q, InnerH[1]);

  return {std::move(Q), std::move(InnerU)};
}

define_method(
  DensePair, merge_col_basis, (const Matrix& Au, const Matrix& Bu)
) {
  omm_error_handler("merge_col_basis", {Au, Bu}, __FILE__, __LINE__);
  std::abort();
}

declare_method(
  DensePair, merge_row_basis,
  (virtual_<const Matrix&>, virtual_<const Matrix&>)
)

define_method(
  DensePair, merge_row_basis,
  (const Dense& Av, const Dense& Bv)
) {
  assert(Av.dim[1] == Bv.dim[1]);
  int64_t Arank = Av.dim[0];
  int64_t Brank = Bv.dim[0];
  assert(Arank == Brank);

  Dense InnerV(Brank, Arank+Brank);
  Hierarchical InnerH = split(InnerV, 1, 2);
  gemm(Bv, Av, InnerH[0], 1, 0, false, true);

  Dense Bv_BvAvtAv(Bv);
  gemm(InnerH[0], Av, Bv_BvAvtAv, -1, 1);

  Dense Q(Brank, Av.dim[1]);
  rq(Bv_BvAvtAv, InnerH[1], Q);

  return {std::move(Q), std::move(InnerV)};
}

define_method(
  DensePair, merge_row_basis, (const Matrix& Av, const Matrix& Bv)
) {
  omm_error_handler("merge_row_basis", {Av, Bv}, __FILE__, __LINE__);
  std::abort();
}

std::tuple<Dense, Dense, Dense> merge_S(
  const Dense& As, const Dense& Bs,
  const Dense& InnerU, const Dense& InnerV
) {
  assert(As.dim[0] == As.dim[1]);
  int64_t rank = As.dim[0];

  Dense InnerUBs = gemm(InnerU, Bs);

  Dense M = gemm(InnerUBs, InnerV);
  Hierarchical MH = split(M, 2, 2);
  MH(0, 0) += As;

  Dense Uhat, Shat, Vhat;
  std::tie(Uhat, Shat, Vhat) = svd(M);
  return {
    resize(Uhat, Uhat.dim[0], rank),
    resize(Shat, rank, rank),
    resize(Vhat, rank, Vhat.dim[1])
  };
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

void orthogonality_preserving_addition(LowRank& A, const LowRank& B) {
  //Bebendorf HMatrix Book p16
  //Rounded Addition
  //Concat U bases
  Dense Uc(get_n_rows(A.U), A.S.dim[0]+B.S.dim[0]);
  IndexRange U_row_range(0, Uc.dim[0]);
  IndexRange U_col_range(0, Uc.dim[1]);
  auto Uc_splits = Uc.split({ U_row_range }, U_col_range.split_at(A.S.dim[0]), false);
  gemm(A.U, A.S, Uc_splits[0], 1, 0);
  gemm(B.U, B.S, Uc_splits[1], 1, 0);
  Dense Qu(Uc.dim[0], std::min(Uc.dim[0], Uc.dim[1]));
  Dense Ru(std::min(Uc.dim[0], Uc.dim[1]), Uc.dim[1]);
  qr(Uc, Qu, Ru);

  //Concat V bases
  Dense Vc(A.S.dim[1]+B.S.dim[1], get_n_cols(A.V));
  IndexRange V_row_range(0, Vc.dim[0]);
  IndexRange V_col_range(0, Vc.dim[1]);
  auto Vc_splits = Vc.split(V_row_range.split_at(A.S.dim[1]), { V_col_range }, false);
  A.V.copy_to(Vc_splits[0]);
  B.V.copy_to(Vc_splits[1]);
  Dense Rv(Vc.dim[0], std::min(Vc.dim[0], Vc.dim[1]));
  Dense Qv(std::min(Vc.dim[0], Vc.dim[1]), Vc.dim[1]);
  rq(Vc, Rv, Qv);

  //SVD and truncate
  Dense RuRv = gemm(Ru, Rv);
  Dense RRU, RRS, RRV;
  std::tie(RRU, RRS, RRV) = svd(RuRv);
  // Find truncation rank if needed
  bool use_eps = (A.eps != 0);
  if(use_eps) A.rank = find_svd_truncation_rank(RRS, A.eps);
  // Truncate
  A.S = resize(RRS, A.rank, A.rank);
  A.U = gemm(Qu, resize(RRU, RRU.dim[0], A.rank));
  A.V = gemm(resize(RRV, A.rank, RRV.dim[1]), Qv);
}

void formatted_addition(LowRank& A, const LowRank& B) {
  //Bebendorf HMatrix Book p17
  //Rounded addition by exploiting orthogonality
  timing::start("LR += LR");

  timing::start("Merge col basis");
  Hierarchical OuterU(1, 2);
  Dense InnerU;
  std::tie(OuterU[1], InnerU) = merge_col_basis(A.U, B.U);
  OuterU[0] = std::move(A.U);
  timing::stop("Merge col basis");

  timing::start("Merge row basis");
  Hierarchical OuterV(2, 1);
  Dense InnerVt;
  std::tie(OuterV[1], InnerVt) = merge_row_basis(A.V, B.V);
  OuterV[0] = std::move(A.V);
  timing::stop("Merge row basis");

  timing::start("Merge S");
  Dense Uhat, Vhat;
  std::tie(Uhat, A.S, Vhat) = merge_S(A.S, B.S, InnerU, InnerVt);
  timing::stop("Merge S");

  // TODO Find a way to use more convenient D=gemm(D, D) here?
  // Restore moved-from U and V and finalize basis
  A.U = Dense(A.dim[0], A.rank);
  A.V = Dense(A.rank, A.dim[1]);
  gemm(OuterU, Uhat, A.U, 1, 0);
  gemm(Vhat, OuterV, A.V, 1, 0);

  timing::stop("LR += LR");
}

define_method(Matrix&, addition_omm, (LowRank& A, const LowRank& B)) {
  assert(A.dim[0] == B.dim[0]);
  assert(A.dim[1] == B.dim[1]);
  assert(A.rank == B.rank);
  if (getGlobalValue("HICMA_LRA") == "naive") {
    naive_addition(A, B);
  } else if (getGlobalValue("HICMA_LRA") == "rounded_orth") {
    orthogonality_preserving_addition(A, B);
  } else {
    formatted_addition(A, B);
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
