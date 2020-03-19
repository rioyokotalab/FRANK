#include "hicma/operations/misc/addition.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/no_copy_split.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/util/counter.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <iostream>
#include <tuple>
#include <utility>


namespace hicma
{

void operator+=(Node& A, const Node& B) {
  addition_omm(A, B);
}

define_method(void, addition_omm, (Dense& A, const LowRank& B)) {
  Dense UxS(B.dim[0], B.rank);
  gemm(B.U(), B.S(), UxS, 1, 0);
  gemm(UxS, B.V(), A, 1, 1);
}

define_method(void, addition_omm, (Hierarchical& A, const LowRank& B)) {
  NoCopySplit BH(B, A.dim[0], A.dim[1]);
  for (int i=0; i<A.dim[0]; i++) {
    for (int j=0; j<A.dim[1]; j++) {
      A(i, j) += BH(i, j);
    }
  }
}


std::tuple<Dense, Dense> merge_col_basis(
  const Dense& U, const Dense& Au
) {
  assert(U.dim[0] == Au.dim[0]);
  int Arank = U.dim[1];
  int Brank = Au.dim[1];
  assert(Arank == Brank);

  Dense InnerU(Arank+Brank, Brank);
  NoCopySplit InnerH(InnerU, 2, 1);
  gemm(U, Au, InnerH[0], true, false, 1, 0);

  // TODO This copy has significant cost. Avoidable?
  Dense U_UUtAu(Au);
  gemm(U, InnerH[0], U_UUtAu, -1, 1);

  Dense Q(U.dim[0], Brank);
  qr(U_UUtAu, Q, InnerH[1]);

  return {std::move(Q), std::move(InnerU)};
}

std::tuple<Dense, Dense> merge_row_basis(
  const Dense& V, const Dense& Av
) {
  assert(V.dim[1] == Av.dim[1]);
  int Arank = V.dim[0];
  int Brank = Av.dim[0];
  assert(Arank == Brank);

  Dense InnerV(Brank, Arank+Brank);
  NoCopySplit InnerH(InnerV, 1, 2);
  gemm(Av, V, InnerH[0], false, true, 1, 0);

  // TODO This copy has significant cost. Avoidable?
  Dense Av_AvVtV(Av);
  gemm(InnerH[0], V, Av_AvVtV, -1, 1);

  Dense Q(Brank, V.dim[1]);
  rq(Av_AvVtV, InnerH[1], Q);

  return {std::move(Q), std::move(InnerV)};
}


std::tuple<Dense, Dense, Dense> merge_S(
  const Dense& S, const Dense& AS,
  const Dense& InnerU, const Dense& InnerV
) {
  assert(S.dim[0] == S.dim[1]);
  int rank = S.dim[0];

  Dense InnerUAS(InnerU.dim[0], AS.dim[1]);
  gemm(InnerU, AS, InnerUAS, 1, 0);

  // TODO Consider using move for S if possible!
  Dense M(rank*2, rank*2);
  for (int i=0; i<rank; i++) {
    for (int j=0; j<rank; j++) {
      M(i, j) = S(i, j);
    }
  }
  gemm(InnerUAS, InnerV, M, 1, 1);

  Dense Uhat, Shat, Vhat;
  std::tie(Uhat, Shat, Vhat) = svd(M);

  Shat.resize(rank, rank);
  Uhat.resize(Uhat.dim[0], rank);
  Vhat.resize(rank, Vhat.dim[1]);

  return {std::move(Uhat), std::move(Shat), std::move(Vhat)};
}

define_method(void, addition_omm, (LowRank& A, const LowRank& B)) {
  assert(A.dim[0] == B.dim[0]);
  assert(A.dim[1] == B.dim[1]);
  assert(A.rank == B.rank);
  if(getCounter("LR_ADDITION_COUNTER") == 1) updateCounter("LR-addition", 1);
  if(getCounter("LRA") == 0) {
    //Truncate and Recompress if rank > min(nrow, ncol)
    if (A.rank+B.rank >= std::min(A.dim[0], A.dim[1])) {
      A = LowRank(Dense(A) + Dense(B), A.rank);
    }
    else {
      LowRank C(A.dim[0], A.dim[1], A.rank+B.rank, A.i_abs, A.j_abs, A.level);
      C.mergeU(A, B);
      C.mergeS(A, B);
      C.mergeV(A, B);
      A.rank += B.rank;
      A.U() = std::move(C.U());
      A.S() = std::move(C.S());
      A.V() = std::move(C.V());
    }
  } else if(getCounter("LRA") == 1) {
    //Bebendorf HMatrix Book p16
    //Rounded Addition
    LowRank C(A.dim[0], A.dim[1], A.rank+B.rank, A.i_abs, A.j_abs, A.level);
    C.mergeU(A, B);
    C.mergeS(A, B);
    C.mergeV(A, B);

    Dense CU_copy(C.U());
    gemm(CU_copy, C.S(), C.U(), 1, 0);

    Dense Qu(C.U().dim[0], C.U().dim[1]);
    Dense Ru(C.U().dim[1], C.U().dim[1]);
    qr(C.U(), Qu, Ru);

    C.V().transpose();
    Dense Qv(C.V().dim[0], C.V().dim[1]);
    Dense Rv(C.V().dim[1], C.V().dim[1]);
    qr(C.V(), Qv, Rv);

    Dense RuRvT(Ru.dim[0], Rv.dim[0]);
    gemm(Ru, Rv, RuRvT, false, true, 1, 0);

    Dense RRU, RRS, RRV;
    std::tie(RRU, RRS, RRV) = svd(RuRvT);

    RRS.resize(A.rank, A.rank);
    A.S() = std::move(RRS);
    RRU.resize(RRU.dim[0], A.rank);
    gemm(Qu, RRU, A.U(), 1, 0);
    RRV.resize(A.rank, RRV.dim[1]);
    gemm(RRV, Qv, A.V(), false, true, 1, 0);
  } else {
    //Bebendorf HMatrix Book p17
    //Rounded addition by exploiting orthogonality
    timing::start("LR += LR");

    timing::start("Merge col basis");
    Hierarchical OuterU(1, 2);
    Dense InnerU;
    std::tie(OuterU[1], InnerU) = merge_col_basis(A.U(), B.U());
    OuterU[0] = std::move(A.U());
    timing::stop("Merge col basis");

    timing::start("Merge row basis");
    Hierarchical OuterV(2, 1);
    Dense InnerVt;
    std::tie(OuterV[1], InnerVt) = merge_row_basis(A.V(), B.V());
    OuterV[0] = std::move(A.V());
    timing::stop("Merge row basis");

    timing::start("Merge S");
    Dense Uhat, Vhat;
    std::tie(Uhat, A.S(), Vhat) = merge_S(A.S(), B.S(), InnerU, InnerVt);
    timing::stop("Merge S");

    // Restore moved-from U and V and finalize basis
    A.U() = Dense(A.dim[0], A.rank);
    A.V() = Dense(A.rank, A.dim[1]);
    gemm(OuterU, Uhat, A.U(), 1, 0);
    gemm(Vhat, OuterV, A.V(), 1, 0);

    timing::stop("LR += LR");
  }
}

define_method(void, addition_omm, (Node& A, const Node& B)) {
  std::cerr << A.type() << " += " << B.type();
  std::cerr << " undefined." << std::endl;
  abort();
}

} // namespace hicma
