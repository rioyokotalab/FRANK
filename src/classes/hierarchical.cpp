#include "hicma/classes/hierarchical.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/intitialization_helpers/basis_tracker.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"
#include "hicma/classes/intitialization_helpers/matrix_initializer.h"
#include "hicma/functions.h"
#include "hicma/gpu_batch/batch.h"
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


namespace hicma
{

Hierarchical::Hierarchical(const Hierarchical& A) : Matrix(A) {
  BasisTracker<BasisKey> tracker;
  *this = Hierarchical(A, tracker);
}

declare_method(
  MatrixProxy, copy_block, (virtual_<const Matrix&>, BasisTracker<BasisKey>&)
)

define_method(
  MatrixProxy, copy_block,
  (const Hierarchical& A, BasisTracker<BasisKey>& tracker)
) {
  return Hierarchical(A, tracker);
}

define_method(
  MatrixProxy, copy_block,
  (const Dense& A, [[maybe_unused]] BasisTracker<BasisKey>& tracker)
) {
  return A;
}

define_method(
  MatrixProxy, copy_block, (const LowRank& A, BasisTracker<BasisKey>& tracker)
) {
  if (!tracker.has_basis(A.U)) {
    tracker[A.U] = A.U;
  }
  if (!tracker.has_basis(A.V)) {
    tracker[A.V] = A.V;
  }
  return LowRank(tracker[A.U], A.S, tracker[A.V], true);
}

define_method(
  MatrixProxy, copy_block,
  (const Matrix& A, [[maybe_unused]] BasisTracker<BasisKey>& tracker)
) {
  omm_error_handler("copy_block", {A}, __FILE__, __LINE__);
  std::abort();
}

Hierarchical::Hierarchical(
  const Hierarchical& A, BasisTracker<BasisKey>& tracker
) : Matrix(A), dim(A.dim), data(dim[0]*dim[1]) {
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      (*this)(i, j) = copy_block(A(i, j), tracker);
    }
  }
}

Hierarchical& Hierarchical::operator=(const Hierarchical& A) {
  *this = Hierarchical(A);
  return *this;
}

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

Hierarchical::Hierarchical(
  const Matrix& A, int64_t n_row_blocks, int64_t n_col_blocks, bool copy
) : dim{n_row_blocks, n_col_blocks}, data(dim[0]*dim[1])
{
  ClusterTree node({0, get_n_rows(A)}, {0, get_n_cols(A)}, dim[0], dim[1]);
  for (const ClusterTree& child : node) {
    (*this)[child] = get_part(A, child, copy);
  }
}

Hierarchical::Hierarchical(const Matrix& A, const Hierarchical& like, bool copy)
: dim(like.dim), data(dim[0]*dim[1]) {
  assert(get_n_rows(A) == get_n_rows(like));
  assert(get_n_cols(A) == get_n_cols(like));
  ClusterTree node(like);
  for (const ClusterTree& child : node) {
    (*this)[child] = get_part(A, child, copy);
  }
}

Hierarchical::Hierarchical(int64_t n_row_blocks, int64_t n_col_blocks)
: dim{n_row_blocks, n_col_blocks}, data(dim[0]*dim[1]) {}

Hierarchical::Hierarchical(
  const ClusterTree& node,
  MatrixInitializer& initer
) : dim(node.block_dim), data(dim[0]*dim[1]) {
  for (const ClusterTree& child : node) {
    if (initer.is_admissible(child)) {
      (*this)[child] = initer.get_compressed_representation(child);
    } else {
      if (child.is_leaf()) {
        (*this)[child] = initer.get_dense_representation(child);
      } else {
        (*this)[child] = Hierarchical(child, initer);
      }
    }
  }
}

Hierarchical::Hierarchical(
  void (*func)(
    Dense& A,
    const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  const std::vector<std::vector<double>>& x,
  int64_t n_rows, int64_t n_cols,
  int64_t rank,
  int64_t nleaf,
  int64_t admis,
  int64_t n_row_blocks, int64_t n_col_blocks,
  int basis_type,
  int64_t row_start, int64_t col_start
) {
  MatrixInitializer initer(func, x, admis, rank, basis_type);
  ClusterTree cluster_tree(
    {row_start, n_rows}, {col_start, n_cols}, n_row_blocks, n_col_blocks, nleaf
  );
  if (basis_type == SHARED_BASIS) {
    // TODO Admissibility is checked later AGAIN (avoid?). Possible solutions:
    //  - Add appropirate booleans to ClusterTree
    //  - Use Tracker in MatrixInitializer
    initer.create_nested_basis(cluster_tree);
  }
  // TODO The following two should be combined into a single call
  *this = Hierarchical(cluster_tree, initer);
}

const MatrixProxy& Hierarchical::operator[](const ClusterTree& node) const {
  return (*this)(node.rel_pos[0], node.rel_pos[1]);
}

MatrixProxy& Hierarchical::operator[](const ClusterTree& node) {
  return (*this)(node.rel_pos[0], node.rel_pos[1]);
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

declare_method(void, unshare_omm, (virtual_<Matrix&>))

define_method(void, unshare_omm, (LowRank& A)) {
  A.U = Dense(A.U);
  A.V = Dense(A.V);
}

define_method(void, unshare_omm, (Matrix&)) {
  // Do nothing
}

void Hierarchical::unshare() {
  for (int64_t i=0; i<dim[0]; ++i) {
    for (int64_t j=0; j<dim[1]; ++j) {
      unshare_omm((*this)(i, j));
    }
  }
}

} // namespace hicma
