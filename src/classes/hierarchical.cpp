#include "hicma/classes/hierarchical.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/initialization_helpers/matrix_initializer.h"
#include "hicma/classes/initialization_helpers/matrix_initializer_block.h"
//#include "hicma/classes/initialization_helpers/matrix_initializer_kernel.h"
#include "hicma/classes/initialization_helpers/matrix_initializer_file.h"
#include "hicma/classes/initialization_helpers/matrix_initializer_function.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <utility>
#include <iostream>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class Hierarchical<float>;
template class Hierarchical<double>;
template Hierarchical<float>::Hierarchical(const Hierarchical<double>&, int64_t);
template Hierarchical<double>::Hierarchical(const Hierarchical<float>&, int64_t);
template Hierarchical<float>::Hierarchical(Dense<float>&&, int64_t, int64_t, double, int64_t,
  int64_t, int64_t, int64_t);
template Hierarchical<float>::Hierarchical(Dense<double>&&, int64_t, int64_t, double, int64_t,
  int64_t, int64_t, int64_t);
template Hierarchical<double>::Hierarchical(Dense<float>&&, int64_t, int64_t, double, int64_t,
  int64_t, int64_t, int64_t);
template Hierarchical<double>::Hierarchical(Dense<double>&&, int64_t, int64_t, double, int64_t,
  int64_t, int64_t, int64_t);
template Hierarchical<float>::Hierarchical(Dense<float>&&, const vec2d<double>&, int64_t, int64_t, double, int64_t,
  int64_t, int64_t, int64_t);
template Hierarchical<double>::Hierarchical(Dense<double>&&, const vec2d<double>&, int64_t, int64_t, double, int64_t,
  int64_t, int64_t, int64_t);

declare_method(Matrix&&, move_from_hierarchical, (virtual_<Matrix&>))

// static_cast is not safe because we don't know the template of MatrixProxy
// TODO add some type of error handling
template<typename T>
Hierarchical<T>::Hierarchical(MatrixProxy&& A)
: Hierarchical(dynamic_cast<Hierarchical<T>&&>(move_from_hierarchical(A))) {}

define_method(Matrix&&, move_from_hierarchical, (Hierarchical<double>& A)) {
  return std::move(A);
}

define_method(Matrix&&, move_from_hierarchical, (Hierarchical<float>& A)) {
  return std::move(A);
}

define_method(Matrix&&, move_from_hierarchical, (Matrix& A)) {
  omm_error_handler("move_from_hierarchical", {A}, __FILE__, __LINE__);
  std::abort();
}

template<typename T> template<typename U>
Hierarchical<T>::Hierarchical(const Hierarchical<U>& A, int64_t rank)
: dim(A.dim), data(dim[0]*dim[1]) {
  int64_t size = A.dim[0] * A.dim[1];
  for (int64_t i=0; i<size; ++i) {
    data[i] = convert_omm(A[i], rank);
  }
}

template<typename T>
Hierarchical<T>::Hierarchical(int64_t n_row_blocks, int64_t n_col_blocks)
: dim{n_row_blocks, n_col_blocks}, data(dim[0]*dim[1]) {}

template<typename T>
Hierarchical<T>::Hierarchical(int64_t n_row_blocks, int64_t n_col_blocks, std::vector<MatrixProxy>&& data)
: dim{n_row_blocks, n_col_blocks}, data(data) {}

template<typename T>
Hierarchical<T>::Hierarchical(
  const ClusterTree& node,
  MatrixInitializer& initializer
  ) : dim(node.block_dim), data(dim[0]*dim[1]) {
  for (const ClusterTree& child : node) {
    if (initializer.is_admissible(child)) {
      (*this)[child.rel_pos] = initializer.template get_compressed_representation<T>(child);
    } else {
      if (child.is_leaf()) {
        (*this)[child.rel_pos] = initializer.template get_dense_representation<T>(child);
      } else {
        (*this)[child.rel_pos] = Hierarchical<T>(child, initializer);
      }
    }
  }
}

template<typename T>
Hierarchical<T>::Hierarchical(
  void (*func)(
    T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const vec2d<double>& params,
    int64_t row_start, int64_t col_start
  ),
  int64_t n_rows, int64_t n_cols,
  int64_t rank,
  int64_t nleaf,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int64_t row_start, int64_t col_start
) {
  MatrixInitializerFunction<T> initializer(func, admis, rank);
  ClusterTree cluster_tree(
    {row_start, n_rows}, {col_start, n_cols}, n_row_blocks, n_col_blocks, nleaf
  );
  *this = Hierarchical<T>(cluster_tree, initializer);
}

template<typename T>
Hierarchical<T>::Hierarchical(
  void (*func)(
    T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const vec2d<double>& params,
    int64_t row_start, int64_t col_start
  ),
  const vec2d<double>& params,
  int64_t n_rows, int64_t n_cols,
  int64_t rank,
  int64_t nleaf,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int admis_type,
  int64_t row_start, int64_t col_start
) {
  MatrixInitializerFunction<T> initializer(func, admis, rank, admis_type, params);
  ClusterTree cluster_tree(
    {row_start, n_rows}, {col_start, n_cols}, n_row_blocks, n_col_blocks, nleaf
  );
  *this = Hierarchical<T>(cluster_tree, initializer);
}

//TODO extend to only use a subset at the beginning?
template<typename T> template<typename U>
Hierarchical<T>::Hierarchical(
  Dense<U>&& A,
  int64_t rank,
  int64_t nleaf,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int64_t row_start, int64_t col_start
) {
  ClusterTree cluster_tree(
    {row_start, A.dim[0]}, {col_start, A.dim[1]},
    n_row_blocks, n_col_blocks, nleaf
  );
  MatrixInitializerBlock<U> initializer(std::move(A), admis, rank);
  *this = Hierarchical<T>(cluster_tree, initializer);
}

template<typename T> template<typename U>
Hierarchical<T>::Hierarchical(
  Dense<U>&& A,
  const vec2d<double>& params,
  int64_t rank,
  int64_t nleaf,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int64_t row_start, int64_t col_start
) {
  ClusterTree cluster_tree(
    {row_start, A.dim[0]}, {col_start, A.dim[1]},
    n_row_blocks, n_col_blocks, nleaf
  );
  MatrixInitializerBlock<U> initializer(std::move(A), admis, rank, GEOMETRY_BASED_ADMIS, params);
  *this = Hierarchical<T>(cluster_tree, initializer);
}

template<typename T>
Hierarchical<T>::Hierarchical(
  std::string filename, MatrixLayout ordering,
  int64_t n_rows, int64_t n_cols,
  int64_t rank,
  int64_t nleaf,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int64_t row_start, int64_t col_start
) {
  MatrixInitializerFile initializer(filename, ordering, admis, rank);
  ClusterTree cluster_tree(
    {row_start, n_rows}, {col_start, n_cols},
    n_row_blocks, n_col_blocks, nleaf
  );
  *this = Hierarchical<T>(cluster_tree, initializer);
}

template<typename T>
Hierarchical<T>::Hierarchical(
  std::string filename, MatrixLayout ordering,
  const vec2d<double>& params,
  int64_t n_rows, int64_t n_cols,
  int64_t rank,
  int64_t nleaf,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int64_t row_start, int64_t col_start
) {
  MatrixInitializerFile initializer(filename, ordering, admis, rank, GEOMETRY_BASED_ADMIS, params);
  ClusterTree cluster_tree(
    {row_start, n_rows}, {col_start, n_cols},
    n_row_blocks, n_col_blocks, nleaf
  );
  *this = Hierarchical<T>(cluster_tree, initializer);
}

template<typename T>
const MatrixProxy& Hierarchical<T>::operator[](
  const std::array<int64_t, 2>& pos
) const {
  return (*this)(pos[0], pos[1]);
}

template<typename T>
MatrixProxy& Hierarchical<T>::operator[](const std::array<int64_t, 2>& pos) {
  return (*this)(pos[0], pos[1]);
}

template<typename T>
const MatrixProxy& Hierarchical<T>::operator[](int64_t i) const {
  assert(dim[0] == 1 || dim[1] == 1);
  assert(i < (dim[0] != 1 ? dim[0] : dim[1]));
  return data[i];
}

template<typename T>
MatrixProxy& Hierarchical<T>::operator[](int64_t i) {
  assert(dim[0] == 1 || dim[1] == 1);
  assert(i < (dim[0] != 1 ? dim[0] : dim[1]));
  return data[i];
}

template<typename T>
const MatrixProxy& Hierarchical<T>::operator()(int64_t i, int64_t j) const {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return data[i*dim[1]+j];
}

template<typename T>
MatrixProxy& Hierarchical<T>::operator()(int64_t i, int64_t j) {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return data[i*dim[1]+j];
}

} // namespace hicma
