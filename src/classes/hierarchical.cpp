#include "hicma/classes/hierarchical.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/initialization_helpers/matrix_initializer.h"
#include "hicma/classes/initialization_helpers/matrix_initializer_block.h"
#include "hicma/classes/initialization_helpers/matrix_initializer_kernel.h"
#include "hicma/classes/initialization_helpers/matrix_initializer_file.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

declare_method(Hierarchical&&, move_from_hierarchical, (virtual_<Matrix&>))

Hierarchical::Hierarchical(MatrixProxy&& A)
: Hierarchical(move_from_hierarchical(A)) {}

define_method(Hierarchical&&, move_from_hierarchical, (Hierarchical& A)) {
  return std::move(A);
}

define_method(Hierarchical&&, move_from_hierarchical, (Matrix& A)) {
  omm_error_handler("move_from_hierarchical", {A}, __FILE__, __LINE__);
  std::abort();
}

Hierarchical::Hierarchical(int64_t n_row_blocks, int64_t n_col_blocks)
: dim{n_row_blocks, n_col_blocks}, data(dim[0]*dim[1]) {}

Hierarchical::Hierarchical(
  const ClusterTree& node,
  MatrixInitializer& initializer,
  bool fixed_rank
) : dim(node.block_dim), data(dim[0]*dim[1]) {
  for (const ClusterTree& child : node) {
    if (initializer.is_admissible(child)) {
      (*this)[child.rel_pos] = initializer.get_compressed_representation(child, fixed_rank);
    } else {
      if (child.is_leaf()) {
        (*this)[child.rel_pos] = initializer.get_dense_representation(child);
      } else {
        (*this)[child.rel_pos] = Hierarchical(child, initializer, fixed_rank);
      }
    }
  }
}

Hierarchical::Hierarchical(
  void (*kernel)(
    double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const std::vector<std::vector<double>>& params,
    int64_t row_start, int64_t col_start
  ),
  std::vector<std::vector<double>> params,
  int64_t n_rows, int64_t n_cols,
  int64_t rank,
  int64_t nleaf,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int admis_type,
  int64_t row_start, int64_t col_start
) {
  MatrixInitializerKernel initializer(kernel, params, admis, 0, rank, admis_type);
  ClusterTree cluster_tree(
    {row_start, n_rows}, {col_start, n_cols}, n_row_blocks, n_col_blocks, nleaf
  );
  *this = Hierarchical(cluster_tree, initializer, true);
}

Hierarchical::Hierarchical(
  void (*kernel)(
    double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const std::vector<std::vector<double>>& params,
    int64_t row_start, int64_t col_start
  ),
  std::vector<std::vector<double>> params,
  int64_t n_rows, int64_t n_cols,
  int64_t nleaf,
  double eps,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int admis_type,
  int64_t row_start, int64_t col_start
) {
  MatrixInitializerKernel initializer(kernel, params, admis, eps, 0, admis_type);
  ClusterTree cluster_tree(
    {row_start, n_rows}, {col_start, n_cols}, n_row_blocks, n_col_blocks, nleaf
  );
  *this = Hierarchical(cluster_tree, initializer, false);
}


Hierarchical::Hierarchical(
  Dense&& A,
  int64_t rank,
  int64_t nleaf,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int64_t row_start, int64_t col_start,
  std::vector<std::vector<double>> params,
  int admis_type
) {
  MatrixInitializerBlock initializer(std::move(A), admis, 0, rank, params, admis_type);
  ClusterTree cluster_tree(
    {row_start, A.dim[0]}, {col_start, A.dim[1]},
    n_row_blocks, n_col_blocks, nleaf
  );
  *this = Hierarchical(cluster_tree, initializer, true);
}

Hierarchical::Hierarchical(
  Dense&& A,
  int64_t nleaf,
  double eps,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int64_t row_start, int64_t col_start,
  std::vector<std::vector<double>> params,
  int admis_type
) {
  MatrixInitializerBlock initializer(std::move(A), admis, eps, 0, params, admis_type);
  ClusterTree cluster_tree(
    {row_start, A.dim[0]}, {col_start, A.dim[1]},
    n_row_blocks, n_col_blocks, nleaf
  );
  *this = Hierarchical(cluster_tree, initializer, false);
}


Hierarchical::Hierarchical(
  std::string filename, MatrixLayout ordering,
  std::vector<std::vector<double>> params,
  int64_t n_rows, int64_t n_cols,
  int64_t rank,
  int64_t nleaf,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int admis_type,
  int64_t row_start, int64_t col_start
) {
  MatrixInitializerFile initializer(filename, ordering, admis, 0, rank, params, admis_type);
  ClusterTree cluster_tree(
    {row_start, n_rows}, {col_start, n_cols},
    n_row_blocks, n_col_blocks, nleaf
  );
  *this = Hierarchical(cluster_tree, initializer, true);
}

Hierarchical::Hierarchical(
  std::string filename, MatrixLayout ordering,
  std::vector<std::vector<double>> params,
  int64_t n_rows, int64_t n_cols,
  int64_t nleaf,
  double eps,
  double admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int admis_type,
  int64_t row_start, int64_t col_start
) {
  MatrixInitializerFile initializer(filename, ordering, admis, eps, 0, params, admis_type);
  ClusterTree cluster_tree(
    {row_start, n_rows}, {col_start, n_cols},
    n_row_blocks, n_col_blocks, nleaf
  );
  *this = Hierarchical(cluster_tree, initializer, false);
}

const MatrixProxy& Hierarchical::operator[](
  const std::array<int64_t, 2>& pos
) const {
  return (*this)(pos[0], pos[1]);
}

MatrixProxy& Hierarchical::operator[](const std::array<int64_t, 2>& pos) {
  return (*this)(pos[0], pos[1]);
}

const MatrixProxy& Hierarchical::operator[](int64_t i) const {
  assert(dim[0] == 1 || dim[1] == 1);
  assert(i < (dim[0] != 1 ? dim[0] : dim[1]));
  return data[i];
}

MatrixProxy& Hierarchical::operator[](int64_t i) {
  assert(dim[0] == 1 || dim[1] == 1);
  assert(i < (dim[0] != 1 ? dim[0] : dim[1]));
  return data[i];
}

const MatrixProxy& Hierarchical::operator()(int64_t i, int64_t j) const {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return data[i*dim[1]+j];
}

MatrixProxy& Hierarchical::operator()(int64_t i, int64_t j) {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return data[i*dim[1]+j];
}

} // namespace hicma
