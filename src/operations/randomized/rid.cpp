#include "hicma/operations/randomized/rid.h"

#include "hicma/classes/dense.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/LAPACK/id.h"
#include "hicma/operations/LAPACK/qr.h"

#include "yorel/multi_methods.hpp"

#include <utility>
#include <vector>

namespace hicma
{

std::tuple<Dense, Dense, Dense> rid(const Dense& A, int sample_size, int rank) {
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
  std::tie(Ub, S, V) = two_sided_id(QtA, rank);
  Dense U(A.dim[0], rank);
  gemm(Q, Ub, U, 1, 0);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, std::vector<int>> one_sided_rid(
  const Dense& A, int sample_size, int rank, bool transA
) {
  std::vector<double> x;
  Dense RN(random_uniform, x, A.dim[transA? 0 : 1], sample_size);
  Dense Y(A.dim[transA? 1 : 0], sample_size);
  gemm(A, RN, Y, transA, false, 1, 0);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense QtA(Q.dim[1], A.dim[transA ? 0 : 1]);
  gemm(Q, A, QtA, true, transA, 1, 0);
  Dense V(rank, A.dim[transA ? 0 : 1]);
  std::vector<int> selected_cols = id(QtA, V, rank);
  return {std::move(V), std::move(selected_cols)};
}

} // namespace hicma
