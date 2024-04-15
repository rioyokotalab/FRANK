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

// explicit template initialization (these are the only available types)
template std::tuple<Dense<float>, Dense<float>, Dense<float>> rsvd(const Dense<float>&, int64_t);
template std::tuple<Dense<double>, Dense<double>, Dense<double>> rsvd(const Dense<double>&, int64_t);
template std::tuple<Dense<float>, Dense<float>, Dense<float>> rsvd(const Hierarchical<float>&, int64_t);
template std::tuple<Dense<double>, Dense<double>, Dense<double>> rsvd(const Hierarchical<double>&, int64_t);

template<typename T>
std::tuple<Dense<T>, Dense<T>, Dense<T>> rsvd(const Dense<T>& A, int64_t sample_size) {
  Dense<T> RN(
    random_uniform, A.dim[1], sample_size);
  Dense<T> Y = gemm(A, RN);
  Dense<T> Q(Y.dim[0], Y.dim[1]);
  Dense<T> R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense<T> QtA = gemm(Q, A, 1, true, false);
  Dense<T> Ub, S, V;
  std::tie(Ub, S, V) = svd(QtA);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  Dense<T> U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}

template<typename T>
std::tuple<Dense<T>, Dense<T>, Dense<T>> rsvd(const Hierarchical<T>& A, int64_t sample_size) {
  Dense<T> RN(
    random_uniform, A.dim[1], sample_size);
  Dense<T> Y = gemm(A, RN);
  Dense<T> Q(Y.dim[0], Y.dim[1]);
  Dense<T> R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense<T> QtA = gemm(Q, A, 1, true, false);
  Dense<T> Ub, S, V;
  std::tie(Ub, S, V) = svd(QtA);
  // TODO Resizing Ub (and thus U) before this operation might save some time!
  Dense<T> U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}

template<typename T>
std::tuple<Dense<T>, Dense<T>, Dense<T>> old_rsvd(const Dense<T>& A, int64_t sample_size) {
  Dense<T> RN(
    random_uniform, A.dim[1], sample_size);
  Dense<T> Y = gemm(A, RN);
  Dense<T> Q(Y.dim[0], Y.dim[1]);
  Dense<T> R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense<T> Bt = gemm(A, Q, 1, true, false);
  Dense<T> Qb(A.dim[1], sample_size);
  Dense<T> Rb(sample_size, sample_size);
  qr(Bt, Qb, Rb);
  Dense<T> Ur, S, Vr;
  std::tie(Ur, S, Vr) = svd(Rb);
  // TODO Resizing Ur (and thus U) before this operation might save some time!
  Dense<T> U = gemm(Q, Ur, 1, false, true);
  // TODO Resizing Vr (and thus V) before this operation might save some time!
  Dense<T> V = gemm(Vr, Qb, 1, true, true);
  return {std::move(U), std::move(S), std::move(V)};
}

} // namespace hicma
