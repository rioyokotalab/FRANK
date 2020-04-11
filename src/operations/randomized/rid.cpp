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

std::tuple<Dense, Dense, Dense> rid(const Dense& A, int sample_size, int rank) {
  std::vector<double> x;
  Dense RN(random_uniform, x, A.dim[1], sample_size);
  Dense Y = gemm(A, RN);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense QtA = gemm(Q, A, 1, true, false);
  Dense Ub, S, V;
  std::tie(Ub, S, V) = id(QtA, rank);
  Dense U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense, std::vector<int>> one_sided_rid(
  const Dense& A, int sample_size, int rank, bool transA
) {
  std::vector<double> x;
  Dense RN(random_uniform, x, A.dim[transA? 0 : 1], sample_size);
  Dense Y = gemm(A, RN, 1, transA, false);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense QtA = gemm(Q, A, 1, true, transA);
  Dense V;
  std::vector<int> selected_cols;
  std::tie(V, selected_cols) = one_sided_id(QtA, rank);
  return {std::move(V), std::move(selected_cols)};
}

} // namespace hicma
