#include "hicma/operations/randomized_factorizations.h"

#include "hicma/classes/dense.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/operations/misc.h"

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

std::tuple<Dense<double>, Dense<double>, Dense<double>> rid(
  const Dense<double>& A, int64_t sample_size, int64_t rank
) {
  Dense<double> RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense<double> Y = gemm(A, RN);
  Dense<double> Q(Y.dim[0], Y.dim[1]);
  Dense<double> R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense<double> QtA = gemm(Q, A, 1, true, false);
  Dense<double> Ub, S, V;
  std::tie(Ub, S, V) = id(QtA, rank);
  Dense<double> U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}

std::tuple<Dense<double>, std::vector<int64_t>> one_sided_rid(
  const Dense<double>& A, int64_t sample_size, int64_t rank, bool column
) {
  // Number of random samples: ColumnID -> m, RowID -> n
  Dense<double> RN(
    random_uniform, std::vector<std::vector<double>>(),
    A.dim[column? 0 : 1], sample_size
  );
  Dense<double> Y;
  if (column) {
    Y = gemm(RN, A, 1, true, false);
  }
  else {
    Y = transpose(gemm(A, RN));
  }
  Dense<double> V;
  std::vector<int64_t> selected_cols;
  std::tie(V, selected_cols) = one_sided_id(Y, rank);
  return {std::move(V), std::move(selected_cols)};
}

std::tuple<Dense<double>, std::vector<int64_t>> old_one_sided_rid(
  const Dense<double>& A, int64_t sample_size, int64_t rank, bool transA
) {
  Dense<double> RN(
    random_uniform, std::vector<std::vector<double>>(),
    A.dim[transA? 0 : 1], sample_size
  );
  Dense<double> Y = gemm(A, RN, 1, transA, false);
  Dense<double> Q(Y.dim[0], Y.dim[1]);
  Dense<double> R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense<double> QtA = gemm(Q, A, 1, true, transA);
  Dense<double> V;
  std::vector<int64_t> selected_cols;
  std::tie(V, selected_cols) = one_sided_id(QtA, rank);
  return {std::move(V), std::move(selected_cols)};
}

} // namespace hicma
