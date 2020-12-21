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

std::tuple<Dense, Dense, Dense> rid(
  const Dense& A, int64_t sample_size, int64_t rank
) {
  //Why can the sampling only be specified in the ID and not the SVD?
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
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

std::tuple<Dense, std::vector<int64_t>> one_sided_rid(
  const Dense& A, int64_t sample_size, int64_t rank, bool column
) {
  // Number of random samples: ColumnID -> m, RowID -> n
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(),
    A.dim[column? 0 : 1], sample_size
  );
  Dense Y;
  if (column) {
    Y = gemm(RN, A, 1, true, false);
  }
  else {
    Y = transpose(gemm(A, RN));
  }
  Dense V;
  std::vector<int64_t> selected_cols;
  std::tie(V, selected_cols) = one_sided_id(Y, rank);
  return {std::move(V), std::move(selected_cols)};
}

std::tuple<Dense, std::vector<int64_t>> old_one_sided_rid(
  const Dense& A, int64_t sample_size, int64_t rank, bool transA
) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(),
    A.dim[transA? 0 : 1], sample_size
  );
  Dense Y = gemm(A, RN, 1, transA, false);
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense QtA = gemm(Q, A, 1, true, transA);
  Dense V;
  std::vector<int64_t> selected_cols;
  std::tie(V, selected_cols) = one_sided_id(QtA, rank);
  return {std::move(V), std::move(selected_cols)};
}

} // namespace hicma
