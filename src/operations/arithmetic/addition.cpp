#include "hicma/operations/arithmetic.h"
#include "hicma/extension_headers/operations.h"
#include "hicma/extension_headers/tuple_types.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/classes/initialization_helpers/basis_tracker.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/operations/arithmetic.h"
#include "hicma/operations/misc.h"
#include "hicma/util/counter.h"
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

define_method(Matrix&, addition_omm, (Hierarchical& A, const LowRank& B)) {
  Hierarchical BH(B, A.dim[0], A.dim[1], false);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) += BH(i, j);
    }
  }
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
  Hierarchical InnerH(InnerU, 2, 1, false);
  gemm(Au, Bu, InnerH[0], true, false, 1, 0);

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
  Hierarchical InnerH(InnerV, 1, 2, false);
  gemm(Bv, Av, InnerH[0], false, true, 1, 0);

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

  // TODO Consider using move for As if possible!
  Dense M(rank*2, rank*2);
  for (int64_t i=0; i<rank; i++) {
    for (int64_t j=0; j<rank; j++) {
      M(i, j) = As(i, j);
    }
  }
  gemm(InnerUBs, InnerV, M, 1, 1);

  Dense Uhat, Shat, Vhat;
  std::tie(Uhat, Shat, Vhat) = svd(M);
  return {
    resize(Uhat, Uhat.dim[0], rank),
    resize(Shat, rank, rank),
    resize(Vhat, rank, Vhat.dim[1])
  };
}

define_method(Matrix&, addition_omm, (LowRank& A, const LowRank& B)) {
  assert(A.dim[0] == B.dim[0]);
  assert(A.dim[1] == B.dim[1]);
  assert(A.rank == B.rank);
  if (is_shared(A.U, B.U) && is_shared(A.V, B.V)) {
    A.S += B.S;
    return A;
  } else if (!is_shared(A.U, B.U) && is_shared(A.V, B.V)) {
    // TODO Only implemented for single layer atm, as in strong-admis UBLR
    if (!concatenated_basis_done(A.U, B.U)) {
      // Concatenate U's
      // TODO Inefficient, double copy
      Hierarchical concatU(1, 2);
      concatU[0] = A.U;
      concatU[1] = get_part(B.U, get_n_rows(B.U), get_n_cols(B.U), 0, 0, false);
      register_concatenated_basis(A.U, B.U, Dense(concatU));
    }
    A.U = share_basis(get_concatenated_basis(A.U, B.U));
    // Concatenate S's
    Hierarchical concatS(2, 1);
    concatS[0] = std::move(A.S);
    concatS[1] = get_part(B.S, get_n_rows(B.S), get_n_cols(B.S), 0, 0, false);
    A.S = Dense(concatS);
    return A;
  } else if (is_shared(A.U, B.U) && !is_shared(A.V, B.V)) {
    // TODO Only implemented for single layer atm, as in strong-admis UBLR
    if (!concatenated_basis_done(A.V, B.V)) {
      // Concatenate U's
      // TODO Inefficient, double copy
      Hierarchical concatV(2, 1);
      concatV[0] = A.V;
      concatV[1] = get_part(B.V, get_n_rows(B.V), get_n_cols(B.V), 0, 0, false);
      register_concatenated_basis(A.V, B.V, Dense(concatV));
    }
    A.V = share_basis(get_concatenated_basis(A.V, B.V));
    // Concatenate S's
    Hierarchical concatS(1, 2);
    concatS[0] = std::move(A.S);
    concatS[1] = get_part(B.S, get_n_rows(B.S), get_n_cols(B.S), 0, 0, false);
    A.S = Dense(concatS);
    return A;
  }
  if (getCounter("LR_ADDITION_COUNTER") == 1) updateCounter("LR-addition", 1);
  if (getCounter("LRA") == 0) {
    //Truncate and Recompress if rank > min(nrow, ncol)
    if (A.rank+B.rank >= std::min(A.dim[0], A.dim[1])) {
      A = LowRank(Dense(A) + Dense(B), A.rank);
    } else {
      LowRank C(A.dim[0], A.dim[1], A.rank+B.rank);
      C.mergeU(A, B);
      C.mergeS(A, B);
      C.mergeV(A, B);
      A.rank += B.rank;
      A.U = std::move(C.U);
      A.S = std::move(C.S);
      A.V = std::move(C.V);
    }
  } else if (getCounter("LRA") == 1) {
    //Bebendorf HMatrix Book p16
    //Rounded Addition
    LowRank C(A.dim[0], A.dim[1], A.rank+B.rank);
    C.mergeU(A, B);
    C.mergeS(A, B);
    C.mergeV(A, B);

    // TODO No need to save C.U first, just do two opertions and save into a
    // Hierarchical
    C.U = gemm(C.U, C.S);

    Dense Qu(get_n_rows(C.U), get_n_cols(C.U));
    Dense Ru(get_n_cols(C.U), get_n_cols(C.U));
    qr(C.U, Qu, Ru);

    // TODO Probably better to do RQ decomposition (maybe make hierarchical
    // version and avoid copies?)
    C.V = transpose(C.V);
    Dense Qv(get_n_rows(C.V), get_n_cols(C.V));
    Dense Rv(get_n_cols(C.V), get_n_cols(C.V));
    qr(C.V, Qv, Rv);

    Dense RuRvT = gemm(Ru, Rv, 1, false, true);

    Dense RRU, RRS, RRV;
    std::tie(RRU, RRS, RRV) = svd(RuRvT);

    A.S = resize(RRS, A.rank, A.rank);
    gemm(Qu, resize(RRU, RRU.dim[0], A.rank), A.U, 1, 0);
    gemm(resize(RRV, A.rank, RRV.dim[1]), Qv, A.V, false, true, 1, 0);
  } else {
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

    // Restore moved-from U and V and finalize basis
    // TODO Find a way to use more convenient D=gemm(D, D) here?
    // Restore moved-from U and V and finalize basis
    A.U = Dense(A.dim[0], A.rank);
    A.V = Dense(A.rank, A.dim[1]);
    gemm(OuterU, Uhat, A.U, 1, 0);
    gemm(Vhat, OuterV, A.V, 1, 0);

    timing::stop("LR += LR");
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
