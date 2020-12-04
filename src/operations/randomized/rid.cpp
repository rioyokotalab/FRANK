#include "hicma/operations/randomized_factorizations.h"

#include "hicma/classes/dense.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/util/print.h"

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

std::tuple<Dense, Dense, Dense> rid(
  const Dense& A, int64_t sample_size, int64_t rank
) {
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

std::tuple<Dense, Dense, Dense> rid_new(
  const Dense& A, int64_t sample_size, int64_t rank, int64_t q
) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(), A.dim[1], sample_size);
  Dense Y = gemm(A, RN);
  for (int64_t i=0; i<q; ++i){
    Dense Yhelp = gemm(A, Y, 1, true, false);
    Y = gemm(A, Yhelp);
  }
  Dense Q(Y.dim[0], Y.dim[1]);
  Dense R(Y.dim[1], Y.dim[1]);
  qr(Y, Q, R);
  //Dense QtA = gemm(Q, A, 1, true, false);
  //Dense U, S, V;
  //std::tie(U, S, V) = id(QtA, rank);
  Dense QtA(Q);
  Dense V(rank, QtA.dim[1]);
  Dense Awork(QtA);
  std::vector<int64_t> selected_cols;
  std::tie(V, selected_cols) = one_sided_id(Awork, rank);
  Dense AC = get_cols(QtA, selected_cols);
  Dense U(rank, QtA.dim[0]);
  AC.transpose();
  Dense ACwork(AC);
  std::tie(U, selected_cols) = one_sided_id(ACwork, rank);
  QtA = get_cols(AC, selected_cols);
  U.transpose();
  QtA.transpose();

  /*
  Dense V(rank, Y.dim[1]);
  Dense Ywork(QtA);
  std::vector<int64_t> selected_cols;
  std::tie(V, selected_cols) = one_sided_id(Ywork, rank);
  Dense YC = get_cols(Q, selected_cols);
  Dense U(rank, A.dim[0]);
  YC.transpose();
  Dense YCwork(YC);
  std::tie(U, selected_cols) = one_sided_id(YCwork, rank);
  Y = get_cols(YC, selected_cols);
  U.transpose();
  Y.transpose();*/
  //Dense Ub = gemm(Q, U);

  return {std::move(U), std::move(QtA), std::move(V)};
}

std::tuple<Dense, std::vector<int64_t>> one_sided_rid(
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

std::tuple<Dense, std::vector<int64_t>> one_sided_rid_new(
  const Dense& A, int64_t sample_size, int64_t rank, int64_t q, bool transA
) {
  Dense RN(
    random_uniform, std::vector<std::vector<double>>(),
    A.dim[transA? 0 : 1], sample_size
  );
  Dense Y;
  if (transA)
    Y = gemm(RN, A, 1, true, false);
  else
    Y = gemm(A, RN);//gemm(RN, A, 1, true, true);
  for (int64_t i=0; i<q; ++i){
    Dense Yhelp = gemm(A, Y, 1, true, false);
    Y = gemm(A, Yhelp);
  }
  Dense V;
  std::vector<int64_t> selected_cols;
  if (!transA)
    Y.transpose();
  std::tie(V, selected_cols) = one_sided_id(Y, rank);
  return {std::move(V), std::move(selected_cols)};
}

} // namespace hicma
