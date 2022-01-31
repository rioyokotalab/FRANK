#include "hicma/operations/randomized_factorizations.h"

#include "hicma/classes/dense.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

std::tuple<Dense<double>, Dense<double>, Dense<double>> rsvd(const Dense<double>& A, int64_t sample_size) {
  Dense<double> RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense<double> Y = gemm(A, RN);
  Dense<double> Q(Y.dim[0], Y.dim[1]);
  Dense<double> R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense<double> QtA = gemm(Q, A, 1, true, false);
  Dense<double> Ub, S, V;
  std::tie(Ub, S, V) = svd(QtA);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  Dense<double> U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense<double>, Dense<double>, Dense<double>> old_rsvd(const Dense<double>& A, int64_t sample_size) {
  Dense<double> RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense<double> Y = gemm(A, RN);
  Dense<double> Q(Y.dim[0], Y.dim[1]);
  Dense<double> R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense<double> Bt = gemm(A, Q, 1, true, false);
  Dense<double> Qb(A.dim[1], sample_size);
  Dense<double> Rb(sample_size, sample_size);
  qr(Bt, Qb, Rb);
  Dense<double> Ur, S, Vr;
  std::tie(Ur, S, Vr) = svd(Rb);
  // TODO Resizing Ur (and thus U) before this operation might save some time!
  Dense<double> U = gemm(Q, Ur, 1, false, true);
  // TODO Resizing Vr (and thus V) before this operation might save some time!
  Dense<double> V = gemm(Vr, Qb, 1, true, true);
  return {std::move(U), std::move(S), std::move(V)};
}

} // namespace hicma
