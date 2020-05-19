#include "hicma/classes/hierarchical.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
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
  const Matrix& A, int64_t n_row_blocks, int64_t n_col_blocks
) : dim{n_row_blocks, n_col_blocks}, data(dim[0]*dim[1])
{
  ClusterTree node(get_n_rows(A), get_n_cols(A), dim[0], dim[1]);
  fill_hierarchical_from(*this, A, node);
}

define_method(
  void, fill_hierarchical_from,
  (Hierarchical& H, const Dense& A, const ClusterTree& node)
) {
  timing::start("fill_hierarchical_from(D)");
  for (const ClusterTree& child : node) {
    H[child] = A.get_part(
      child.dim[0], child.dim[1], child.start[0], child.start[1]);
  }
  timing::stop("fill_hierarchical_from(D)");
}

define_method(
  void, fill_hierarchical_from,
  (Hierarchical& H, const LowRank& A, const ClusterTree& node)
) {
  timing::start("fill_hierarchical_from(LR)");
  for (const ClusterTree& child : node) {
    H[child] = A.get_part(
      child.dim[0], child.dim[1], child.start[0], child.start[1]);
  }
  timing::stop("fill_hierarchical_from(LR)");
}

define_method(
  void, fill_hierarchical_from,
  (Hierarchical& H, const Matrix& A, [[maybe_unused]] const ClusterTree& node)
) {
  omm_error_handler("fill_hierarchical_from", {H, A}, __FILE__, __LINE__);
  std::abort();
}

Hierarchical::Hierarchical(int64_t n_row_blocks, int64_t n_col_blocks)
: dim{n_row_blocks, n_col_blocks}, data(dim[0]*dim[1]) {}

Hierarchical::Hierarchical(
  const ClusterTree& node,
  const MatrixInitializer& initer,
  int64_t rank,
  int64_t admis
) : dim(node.block_dim), data(dim[0]*dim[1]) {
  for (const ClusterTree& child : node) {
    if (is_admissible(child, admis)) {
      (*this)[child] = LowRank(initer.get_dense_representation(child), rank);
    } else {
      if (child.is_leaf()) {
        (*this)[child] = initer.get_dense_representation(child);
      } else {
        (*this)[child] = Hierarchical(child, initer, rank, admis);
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
  int64_t row_start, int64_t col_start
) : Hierarchical(
      ClusterTree(
        n_rows, n_cols, n_row_blocks, n_col_blocks, row_start, col_start, nleaf
      ),
      MatrixInitializer(func, x),
      rank, admis
    )
{}

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

void Hierarchical::blr_col_qr(Hierarchical& Q, Hierarchical& R) {
  assert(dim[1] == 1);
  assert(Q.dim[0] == dim[0]);
  assert(Q.dim[1] == 1);
  assert(R.dim[0] == 1);
  assert(R.dim[1] == 1);
  Hierarchical Qu(dim[0], 1);
  Hierarchical B(dim[0], 1);
  for(int64_t i=0; i<dim[0]; i++) {
    std::tie(Qu(i, 0), B(i, 0)) = make_left_orthogonal((*this)(i, 0));
  }
  Dense DB(B);
  Dense Qb(DB.dim[0], DB.dim[1]);
  Dense Rb(DB.dim[1], DB.dim[1]);
  qr(DB, Qb, Rb);
  R(0, 0) = std::move(Rb);
  //Slice Qb based on B
  Hierarchical HQb(B.dim[0], B.dim[1]);
  int64_t rowOffset = 0;
  for(int64_t i=0; i<HQb.dim[0]; i++) {
    int64_t dim_Bi[2]{get_n_rows(B(i, 0)), get_n_cols(B(i, 0))};
    Dense Qbi(dim_Bi[0], dim_Bi[1]);
    for(int64_t row=0; row<dim_Bi[0]; row++) {
      for(int64_t col=0; col<dim_Bi[1]; col++) {
        Qbi(row, col) = Qb(rowOffset + row, col);
      }
    }
    // Moving should not make a difference. Why is this not auto-optimized?
    HQb(i, 0) = std::move(Qbi);
    rowOffset += dim_Bi[0];
  }
  for(int64_t i=0; i<dim[0]; i++) {
    gemm(Qu(i, 0), HQb(i, 0), Q(i, 0), 1, 0);
  }
}

void Hierarchical::split_col(Hierarchical& QL) {
  assert(dim[1] == 1);
  assert(QL.dim[0] == dim[0]);
  assert(QL.dim[1] == 1);
  int64_t rows = 0;
  int64_t cols = 1;
  for(int64_t i=0; i<dim[0]; i++) {
    update_splitted_size((*this)(i, 0), rows, cols);
  }
  Hierarchical spA(rows, cols);
  int64_t curRow = 0;
  for(int64_t i=0; i<dim[0]; i++) {
    QL(i, 0) = split_by_column((*this)(i, 0), spA, curRow);
  }
  *this = std::move(spA);
}

void Hierarchical::restore_col(
  const Hierarchical& Sp, const Hierarchical& QL
) {
  assert(dim[1] == 1);
  assert(dim[0] == QL.dim[0]);
  assert(QL.dim[1] == 1);
  Hierarchical restoredA(dim[0], dim[1]);
  int64_t curSpRow = 0;
  for(int64_t i=0; i<dim[0]; i++) {
    restoredA(i, 0) = concat_columns((*this)(i, 0), Sp, QL(i, 0), curSpRow);
  }
  *this = std::move(restoredA);
}

void Hierarchical::col_qr(int64_t j, Hierarchical& Q, Hierarchical &R) {
  assert(Q.dim[0] == dim[0]);
  assert(Q.dim[1] == 1);
  assert(R.dim[0] == 1);
  assert(R.dim[1] == 1);
  bool split = false;
  Hierarchical Aj(dim[0], 1);
  for(int64_t i=0; i<dim[0]; i++) {
    Aj(i, 0) = (*this)(i, j);
    split |= need_split(Aj(i, 0));
  }
  if(!split) {
    Aj.blr_col_qr(Q, R);
  }
  else {
    Hierarchical QL(dim[0], 1); //Stored Q for splitted lowrank blocks
    Aj.split_col(QL);
    Hierarchical SpQj(Aj);
    qr(Aj, SpQj, R(0, 0));
    Q.restore_col(SpQj, QL);
  }
}

bool Hierarchical::is_admissible(
  const ClusterTree& node, int64_t dist_to_diag
) {
  bool admissible = true;
  // Main admissibility condition
  admissible &= (node.dist_to_diag() > dist_to_diag);
  // Vectors are never admissible
  admissible &= (node.dim[0] > 1 && node.dim[1] > 1);
  return admissible;
}

} // namespace hicma
