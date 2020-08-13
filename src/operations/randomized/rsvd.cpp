#include "hicma/operations/randomized_factorizations.h"

#include "hicma/classes/dense.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

std::tuple<Dense, Dense, Dense> rsvd(const Dense& A, int64_t sample_size) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense Y = gemm(A, RN);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense QtA = gemm(Q, A, 1, true, false);
  Dense Ub, S, V;
  std::tie(Ub, S, V) = svd(QtA);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  Dense U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, Dense, Dense> old_rsvd(const Dense& A, int64_t sample_size) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense Y = gemm(A, RN);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense Bt = gemm(A, Q, 1, true, false);
  Dense Qb(A.dim[1], sample_size);
  Dense Rb(sample_size, sample_size);
  qr(Bt, Qb, Rb);
  Dense Ur, S, Vr;
  std::tie(Ur, S, Vr) = svd(Rb);
  // TODO Resizing Ur (and thus U) before this operation might save some time!
  Dense U = gemm(Q, Ur, 1, false, true);
  // TODO Resizing Vr (and thus V) before this operation might save some time!
  Dense V = gemm(Vr, Qb, 1, true, true);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, Dense, Dense> rsvd_pow(const Dense& A, int64_t sample_size, int64_t q) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense Y = gemm(A, RN);
  for (int64_t i=0; i<q; ++i){
    Dense Z = gemm(A, Y, 1, true, false);
    Y = gemm(A, Z);
  }
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense QtA = gemm(Q, A, 1, true, false);
  Dense Ub, S, V;
  std::tie(Ub, S, V) = svd(QtA);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  Dense U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, Dense, Dense> rsvd_powOrtho(const Dense& A, int64_t sample_size, int64_t q) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense Y = gemm(A, RN);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense Wo(Y.dim[0], Y.dim[1]);
  for (int64_t i=0; i<q; ++i){
    Dense W = gemm(A, Q, 1, true, false);
    qr(W, Wo, R);
    Y = gemm(A, Wo);
    qr(Y, Q, R);
  }
  Dense QtA = gemm(Q, A, 1, true, false);
  Dense Ub, S, V;
  std::tie(Ub, S, V) = svd(QtA);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  Dense U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, Dense, Dense> rsvd_singlePass(const Dense& A, int64_t sample_size) {
  Dense RN1(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense RN2(
    random_uniform, std::vector<std::vector<double>>(), A.dim[0], sample_size);
  Dense Y1 = gemm(A, RN1);
  Dense Y2 = gemm(A, RN2, 1, true, false);
  Dense Q1(Y1.dim[0], Y1.dim[1]);
  Dense R1(Y1.dim[1], Y1.dim[1]);
  Dense Q2(Y2.dim[0], Y2.dim[1]);
  Dense R2(Y2.dim[1], Y2.dim[1]);
  qr(Y1, Q1, R1);
  qr(Y2, Q2, R2);
  //from (G2'Q1)X = Y2'Q2 aka Ax=B
  Dense RN2tQ1 = gemm(RN2, Q1, 1, true, false);
  Dense Y2tQ2 = gemm(Y2, Q2, 1, true, false);

  LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', sample_size, sample_size, sample_size, &RN2tQ1, RN2tQ1.stride, &Y2tQ2, Y2tQ2.stride);

  Dense Ub, S, Vb;
  std::tie(Ub, S, Vb) = svd(Y2tQ2);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  Dense U = gemm(Q1, Ub);
  Dense V = gemm(Q2, Vb);
  return {std::move(U), std::move(S), std::move(V)};
}

} // namespace hicma
