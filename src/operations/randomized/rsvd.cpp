#include "FRANK/operations/randomized_factorizations.h"

#include "FRANK/classes/dense.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/functions.h"
#include "FRANK/operations/BLAS.h"
#include "FRANK/operations/LAPACK.h"

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>


namespace FRANK
{

  std::tuple<Dense, Dense, Dense> rsvd(const Dense& A, const int64_t sample_size) {
  Dense RN(random_uniform, {}, A.dim[1], sample_size);
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

std::tuple<Dense, Dense, Dense> rsvd(const Hierarchical& A, const int64_t sample_size) {
  Dense RN(random_uniform, {}, A.dim[1], sample_size);
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

std::tuple<Dense, Dense, Dense> old_rsvd(const Dense& A, const int64_t sample_size) {
  Dense RN(random_uniform, {}, A.dim[1], sample_size);
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

} // namespace FRANK
