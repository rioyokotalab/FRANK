#include "hicma/operations/randomized_factorizations.h"

#include "hicma/classes/dense.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"

#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

std::tuple<Dense, Dense, Dense> rsvd(const Dense& A, int sample_size) {
  std::vector<double> x;
  Dense RN(random_uniform, x, A.dim[1], sample_size);
  Dense Y(A.dim[0], sample_size);
  gemm(A, RN, Y, 1, 0);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense QtA(Q.dim[1], A.dim[1]);
  gemm(Q, A, QtA, true, false, 1, 0);
  Dense Ub, S, V;
  std::tie(Ub, S, V) = svd(QtA);
  Dense U(A.dim[0], sample_size);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  gemm(Q, Ub, U, 1, 0);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, Dense, Dense> old_rsvd(const Dense& A, int sample_size) {
  std::vector<double> x;
  Dense RN(random_uniform, x, A.dim[1], sample_size);
  Dense Y(A.dim[0], sample_size);
  gemm(A, RN, Y, 1, 0);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense Bt(A.dim[1], sample_size);
  gemm(A, Q, Bt, true, false, 1, 0);
  Dense Qb(A.dim[1], sample_size);
  Dense Rb(sample_size, sample_size);
  qr(Bt, Qb, Rb);
  Dense Ur, S, Vr;
  std::tie(Ur, S, Vr) = svd(Rb);
  // TODO Resizing Ur (and thus U) before this operation might save some time!
  Dense U(A.dim[0], sample_size);
  gemm(Q, Ur, U, false, true, 1, 0);
  // TODO Resizing Vr (and thus V) before this operation might save some time!
  Dense V(sample_size, A.dim[1]);
  gemm(Vr, Qb, V, true, true, 1, 0);
  return {std::move(U), std::move(S), std::move(V)};
}

} // namespace hicma
