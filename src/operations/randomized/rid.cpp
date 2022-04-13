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

// explicit template initialization (these are the only available types)
template std::tuple<Dense<float>, std::vector<int64_t>> one_sided_rid(
  const Dense<float>&, int64_t, int64_t, bool);
template std::tuple<Dense<double>, std::vector<int64_t>> one_sided_rid(
  const Dense<double>&, int64_t, int64_t, bool);
template std::tuple<Dense<float>, Dense<float>, Dense<float>> rid(
  const Dense<float>&, int64_t, int64_t);
template  std::tuple<Dense<double>, Dense<double>, Dense<double>> rid(
  const Dense<double>&, int64_t, int64_t);
template std::tuple<Dense<float>, std::vector<int64_t>> old_one_sided_rid(
  const Dense<float>&, int64_t, int64_t, bool);
std::tuple<Dense<double>, std::vector<int64_t>> old_one_sided_rid(
  const Dense<double>&, int64_t, int64_t, bool);


template<typename T>
std::tuple<Dense<T>, Dense<T>, Dense<T>> rid(
  const Dense<T>& A, int64_t sample_size, int64_t rank
) {
  Dense<T> RN(
    random_uniform<T>, A.dim[1], sample_size);
  Dense<T> Y = gemm(A, RN);
  Dense<T> Q(Y.dim[0], Y.dim[1]);
  Dense<T> R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense<T> QtA = gemm(Q, A, 1, true, false);
  Dense<T> Ub, S, V;
  std::tie(Ub, S, V) = id<T>(QtA, rank);
  Dense<T> U = gemm(Q, Ub);
  return {std::move(U), std::move(S), std::move(V)};
}

template<typename T>
std::tuple<Dense<T>, std::vector<int64_t>> one_sided_rid(
  const Dense<T>& A, int64_t sample_size, int64_t rank, bool column
) {
  // Number of random samples: ColumnID -> m, RowID -> n
  Dense<T> RN(
    random_uniform<T>,
    A.dim[column? 0 : 1], sample_size
  );
  Dense<T> Y;
  if (column) {
    Y = gemm(RN, A, 1, true, false);
  }
  else {
    Y = transpose(gemm(A, RN));
  }
  Dense<T> V;
  std::vector<int64_t> selected_cols;
  std::tie(V, selected_cols) = one_sided_id<T>(Y, rank);
  return {std::move(V), std::move(selected_cols)};
}

template<typename T>
std::tuple<Dense<T>, std::vector<int64_t>> old_one_sided_rid(
  const Dense<T>& A, int64_t sample_size, int64_t rank, bool transA
) {
  Dense<T> RN(
    random_uniform<T>,
    A.dim[transA? 0 : 1], sample_size
  );
  Dense<T> Y = gemm(A, RN, 1, transA, false);
  Dense<T> Q(Y.dim[0], Y.dim[1]);
  Dense<T> R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  Dense<T> QtA = gemm(Q, A, 1, true, transA);
  Dense<T> V;
  std::vector<int64_t> selected_cols;
  std::tie(V, selected_cols) = one_sided_id<T>(QtA, rank);
  return {std::move(V), std::move(selected_cols)};
}

} // namespace hicma
