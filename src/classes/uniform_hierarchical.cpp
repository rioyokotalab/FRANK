#include "hicma/classes/uniform_hierarchical.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"
#include "hicma/classes/intitialization_helpers/matrix_initializer.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/randomized_factorizations.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>


namespace hicma
{

UniformHierarchical::UniformHierarchical(const UniformHierarchical& A)
: Hierarchical(A) {
  col_basis.resize(A.dim[0]);
  copy_col_basis(A);
  row_basis.resize(A.dim[1]);
  copy_row_basis(A);
  for (int64_t i=0; i<dim[0]; i++) {
    for (int64_t j=0; j<dim[1]; j++) {
      set_col_basis(i, j);
      set_row_basis(i, j);
    }
  }
}

declare_method(
  UniformHierarchical, move_from_uniform_hierarchical, (virtual_<Node&>))

define_method(
  UniformHierarchical, move_from_uniform_hierarchical,
  (UniformHierarchical& A)
) {
  return std::move(A);
}

define_method(UniformHierarchical, move_from_uniform_hierarchical, (Node& A)) {
  omm_error_handler("move_from_unifor_hierarchical", {A}, __FILE__, __LINE__);
  std::abort();
}

UniformHierarchical::UniformHierarchical(NodeProxy&& A) {
  *this = move_from_uniform_hierarchical(A);
}

UniformHierarchical::UniformHierarchical(
  int64_t n_row_blocks, int64_t n_col_blocks
) : Hierarchical(n_row_blocks, n_col_blocks) {
  col_basis.resize(n_row_blocks);
  row_basis.resize(n_col_blocks);
}

Dense UniformHierarchical::make_block_row(
  const ClusterTree& node,
  const MatrixInitializer& initer,
  int64_t admis
) {
  // Create row block without admissible blocks
  int64_t n_cols = 0;
  std::vector<std::reference_wrapper<const ClusterTree>> admissible_blocks;
  for (const ClusterTree& block : node.get_block_row()) {
    if (is_admissible(block, admis)) {
      admissible_blocks.emplace_back(block);
      n_cols += block.dim[1];
    }
  }
  Dense block_row(node.dim[0], n_cols);
  int64_t col_start = 0;
  for (const ClusterTree& block : admissible_blocks) {
    Dense part(block.dim[0], block.dim[1], 0, col_start, block_row);
    initer.fill_dense_representation(part, block);
    col_start += block.dim[1];
  }
  return block_row;
}

Dense UniformHierarchical::make_block_col(
  const ClusterTree& node,
  const MatrixInitializer& initer,
  int64_t admis
) {
  // Create col block without admissible blocks
  int64_t n_rows = 0;
  std::vector<std::reference_wrapper<const ClusterTree>> admissible_blocks;
  for (const ClusterTree& block : node.get_block_col()) {
    if (is_admissible(block, admis)) {
      admissible_blocks.emplace_back(block);
      n_rows += block.dim[0];
    }
  }
  Dense block_col(n_rows, node.dim[1]);
  int64_t row_start = 0;
  for (const ClusterTree& block : admissible_blocks) {
    Dense part(block.dim[0], block.dim[1], row_start, 0, block_col);
    initer.fill_dense_representation(part, block);
    row_start += block.dim[0];
  }
  return block_col;
}

LowRankShared UniformHierarchical::construct_shared_block_id(
  const ClusterTree& node,
  const MatrixInitializer& initer,
  std::vector<std::vector<int64_t>>& selected_rows,
  std::vector<std::vector<int64_t>>& selected_cols,
  int64_t rank,
  int64_t admis
) {
  if (col_basis[node.rel_pos[0]].get() == nullptr) {
    Dense row_block = make_block_row(node, initer, admis);
    // Construct U using the ID and remember the selected rows
    Dense Ut;
    std::tie(Ut, selected_rows[node.rel_pos[0]]) = one_sided_rid(
      row_block, rank+5, rank, true);
    transpose(Ut);
    col_basis[node.rel_pos[0]] = std::make_shared<Dense>(std::move(Ut));
  }
  if (row_basis[node.rel_pos[1]].get() == nullptr) {
    Dense col_block = make_block_col(node, initer, admis);
    // Construct V using the ID and remember the selected cols
    Dense V;
    std::tie(V, selected_cols[node.rel_pos[1]]) = one_sided_rid(
      col_block, rank+5, rank);
    row_basis[node.rel_pos[1]] = std::make_shared<Dense>(std::move(V));
  }
  Dense D = initer.get_dense_representation(node);
  Dense S(rank, rank);
  for (int64_t ic=0; ic<rank; ++ic) {
    for (int64_t jc=0; jc<rank; ++jc) {
      S(ic, jc) = D(
        selected_rows[node.rel_pos[0]][ic], selected_cols[node.rel_pos[1]][jc]);
    }
  }
  return LowRankShared(
    S,
    col_basis[node.rel_pos[0]], row_basis[node.rel_pos[1]]
  );
}

LowRankShared UniformHierarchical::construct_shared_block_svd(
  const ClusterTree& node,
  const MatrixInitializer& initer,
  int64_t rank,
  int64_t admis
) {
  if (col_basis[node.rel_pos[0]].get() == nullptr) {
    Dense row_block = make_block_row(node, initer, admis);
    col_basis[node.rel_pos[0]] = std::make_shared<Dense>(
      LowRank(row_block, rank).U());
  }
  if (row_basis[node.rel_pos[1]].get() == nullptr) {
    Dense col_block = make_block_col(node, initer, admis);
    row_basis[node.rel_pos[1]] = std::make_shared<Dense>(LowRank(
      col_block, rank).V());
  }
  Dense D = initer.get_dense_representation(node);
  Dense S = gemm(
    gemm(*col_basis[node.rel_pos[0]], D, 1, true, false),
    *row_basis[node.rel_pos[1]],
    1, false ,true
  );
  return LowRankShared(
    S,
    col_basis[node.rel_pos[0]], row_basis[node.rel_pos[1]]
  );
}

UniformHierarchical::UniformHierarchical(
  const ClusterTree& node,
  const MatrixInitializer& initer,
  int64_t rank,
  int64_t admis,
  bool use_svd
) : Hierarchical(node.block_dim[0], node.block_dim[1]) {
  // TODO All dense UH not allowed for now!
  assert(dim[0] > admis + 1 && dim[1] > admis+1);
  // TODO Only single leve allowed for now. Constructions and some operations
  // work for more levels, but LU does not yet (LR+=LR issue).
  assert(node.dim[0]/node.block_dim[0] <= node.nleaf);
  assert(node.dim[1]/node.block_dim[1] <= node.nleaf);
  assert(rank <= node.nleaf);
  // TODO For now only admis 0! gemm(D, LR, LR) and gemm(LR, D, LR) needed for
  // more.
  assert(admis == 0);
  col_basis.resize(dim[0]);
  row_basis.resize(dim[1]);
  std::vector<std::vector<int64_t>> selected_rows(dim[0]);
  std::vector<std::vector<int64_t>> selected_cols(dim[1]);
  for (const ClusterTree& child : node) {
    if (is_admissible(child, admis)) {
      if (use_svd) {
        (*this)[child] = construct_shared_block_svd(
          child, initer, rank, admis);
      } else {
        (*this)[child] = construct_shared_block_id(
          child, initer, selected_rows, selected_cols, rank, admis);
      }
    } else {
      if (child.is_leaf()) {
        (*this)[child] = initer.get_dense_representation(child);
      } else {
        (*this)[child] = UniformHierarchical(child, initer, rank, admis);
      }
    }
  }
}

UniformHierarchical::UniformHierarchical(
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
  bool use_svd,
  int64_t row_start, int64_t col_start
) : UniformHierarchical(
    ClusterTree(
      n_rows, n_cols, n_row_blocks, n_col_blocks, row_start, col_start, nleaf),
    MatrixInitializer(func, x),
    rank, admis, use_svd
  )
{}

// declare_method(bool, is_LowRankShared, (virtual_<const Node&>));

// define_method(bool, is_LowRankShared, (const LowRankShared&)) {
//   return true;
// }

// define_method(bool, is_LowRankShared, (const Node&)) {
//   return false;
// }

// const UniformHierarchical& UniformHierarchical::operator+=(const UniformHierarchical& A) {
//   // Model after LR+=LR!
//   assert(dim[0] == A.dim[0]);
//   assert(dim[1] == A.dim[1]);

//   std::vector<std::shared_ptr<Dense>> new_col_basis(dim[0]), new_row_basis(dim[1]);

//   for (int64_t i=0; i<dim[0]; i++) {
//     for (int64_t j=0; j<dim[1]; j++) {
//       // Use LR merge on row and column basis to get Inner and Outers.
//       // Create stack of InnerUxSxInnerV for block column of *this.
//       // Compute SVD on the stack and use the row basis as shared InnerV.
//       // Do equivalent for block row and get shared InnerU.
//       // Use shared InnerU and InnerV to get S for all blocks.
//       // Use shared InnerU, InnerV, OuterU and OuterV to compute new shared row
//       // and column bases.

//       // if (is_LowRankShared((*this)(i, j))) {
//       //   int64_t n_non_admissible_blocks = 0;
//       //   for (int64_t i_b=0; i_b<dim[0]; ++i_b) {
//       //     if (is_LowRankShared((*this)(i_b, j))) n_non_admissible_blocks++;
//       //   }
//       //   Hierarchical col_block_h(n_non_admissible_blocks, 1);
//       //   // Note the ins counter!
//       //   for (int64_t i_b=0, ins=0; i_b<dim[0]; ++i_b) {
//       //     if (is_LowRankShared((*this)(i_b, j)))
//       //       col_block_h[ins++] = Dense();
//       //   }
//       //   Dense col_block(col_block_h);
//       // } else {

//       // }
//     }
//   }
//   return *this;
// }

Dense& UniformHierarchical::get_row_basis(int64_t i) {
  assert(i < dim[0]);
  return *row_basis[i];
}

const Dense& UniformHierarchical::get_row_basis(int64_t i) const {
  assert(i < dim[0]);
  return *row_basis[i];
}

Dense& UniformHierarchical::get_col_basis(int64_t j) {
  assert(j < dim[1]);
  return *col_basis[j];
}

const Dense& UniformHierarchical::get_col_basis(int64_t j) const {
  assert(j < dim[0]);
  return *col_basis[j];
}

void UniformHierarchical::copy_col_basis(const UniformHierarchical& A) {
  assert(dim[0] == A.dim[0]);
  for (int64_t i=0; i<dim[1]; i++) {
    col_basis[i] = std::make_shared<Dense>(A.get_col_basis(i));
  }
}

void UniformHierarchical::copy_row_basis(const UniformHierarchical& A) {
  assert(dim[1] == A.dim[1]);
  for (int64_t j=0; j<dim[1]; j++) {
    row_basis[j] = std::make_shared<Dense>(A.get_row_basis(j));
  }
}


declare_method(
  void, set_col_basis_omm,
  (virtual_<Node&>, std::shared_ptr<Dense>)
)

define_method(
  void, set_col_basis_omm,
  (LowRankShared& A, std::shared_ptr<Dense> basis)
) {
  A.U = basis;
}

define_method(
  void, set_col_basis_omm,
  ([[maybe_unused]] Dense& A, [[maybe_unused]] std::shared_ptr<Dense> basis)
) {
  // Do nothing
}

define_method(
  void, set_col_basis_omm,
  (
    [[maybe_unused]] UniformHierarchical& A,
    [[maybe_unused]] std::shared_ptr<Dense> basis
  )
) {
  // Do nothing
}

define_method(
  void, set_col_basis_omm,
  (Node& A, [[maybe_unused]] std::shared_ptr<Dense> basis)
) {
  omm_error_handler("set_col_basis", {A}, __FILE__, __LINE__);
  std::abort();
}

declare_method(
  void, set_row_basis_omm,
  (virtual_<Node&>, std::shared_ptr<Dense>)
)

define_method(
  void, set_row_basis_omm,
  (LowRankShared& A, std::shared_ptr<Dense> basis)
) {
  A.V = basis;
}

define_method(
  void, set_row_basis_omm,
  ([[maybe_unused]] Dense& A, [[maybe_unused]] std::shared_ptr<Dense> basis)
) {
  // Do nothing
}

define_method(
  void, set_row_basis_omm,
  (
    [[maybe_unused]] UniformHierarchical& A,
    [[maybe_unused]] std::shared_ptr<Dense> basis
  )
) {
  // Do nothing
}

define_method(
  void, set_row_basis_omm,
  (Node& A, [[maybe_unused]] std::shared_ptr<Dense> basis)
) {
  omm_error_handler("set_row_basis", {A}, __FILE__, __LINE__);
  std::abort();
}

void UniformHierarchical::set_col_basis(int64_t i, int64_t j) {
  hicma::set_col_basis_omm((*this)(i, j), col_basis[i]);
}

void UniformHierarchical::set_row_basis(int64_t i, int64_t j) {
  hicma::set_row_basis_omm((*this)(i, j), row_basis[j]);
}

} // namespace hicma
