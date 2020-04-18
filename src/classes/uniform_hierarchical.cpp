#include "hicma/classes/uniform_hierarchical.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/randomized_factorizations.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/operations/misc/transpose.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
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

std::unique_ptr<Node> UniformHierarchical::clone() const {
  return std::make_unique<UniformHierarchical>(*this);
}

std::unique_ptr<Node> UniformHierarchical::move_clone() {
  return std::make_unique<UniformHierarchical>(std::move(*this));
}

const char* UniformHierarchical::type() const { return "UniformHierarchical"; }

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
  abort();
}

UniformHierarchical::UniformHierarchical(NodeProxy&& A) {
  *this = move_from_uniform_hierarchical(A);
}

UniformHierarchical::UniformHierarchical(
  int64_t ni_level, int64_t nj_level
) : Hierarchical(ni_level, nj_level) {
  col_basis.resize(ni_level);
  row_basis.resize(nj_level);
}

Dense UniformHierarchical::make_block_row(
  const ClusterTree& node,
  void (*func)(
    Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
  ),
  std::vector<double>& x,
  int64_t admis
) {
  // Create row block without admissible blocks
  int64_t n_cols = 0;
  std::vector<const ClusterTree*> admissible_blocks;
  for (const ClusterTree* block : node.get_block_row()) {
    if (is_admissible(*block, admis)) {
      admissible_blocks.emplace_back(block);
      n_cols += block->dim[1];
    }
  }
  Dense block_row(node.dim[0], n_cols);
  int64_t col_begin = 0;
  for (const ClusterTree* block : admissible_blocks) {
    Dense part(
      ClusterTree(block->dim[0], block->dim[1], 0, col_begin), block_row);
    func(part, x, block->begin[0], block->begin[1]);
    col_begin += block->dim[1];
  }
  return block_row;
}

Dense UniformHierarchical::make_block_col(
  const ClusterTree& node,
  void (*func)(
    Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
  ),
  std::vector<double>& x,
  int64_t admis
) {
  // Create col block without admissible blocks
  int64_t n_rows = 0;
  std::vector<const ClusterTree*> admissible_blocks;
  for (const ClusterTree* block : node.get_block_col()) {
    if (is_admissible(*block, admis)) {
      admissible_blocks.emplace_back(block);
      n_rows += block->dim[0];
    }
  }
  Dense block_col(n_rows, node.dim[1]);
  int64_t row_begin = 0;
  for (const ClusterTree* block : admissible_blocks) {
    Dense part(
      ClusterTree(block->dim[0], block->dim[1], row_begin, 0), block_col);
    func(part, x, block->begin[0], block->begin[1]);
    row_begin += block->dim[0];
  }
  return block_col;
}

LowRankShared UniformHierarchical::construct_shared_block_id(
  const ClusterTree& node,
  void (*func)(
    Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
  ),
  std::vector<double>& x,
  std::vector<std::vector<int64_t>>& selected_rows,
  std::vector<std::vector<int64_t>>& selected_cols,
  int64_t rank,
  int64_t admis
) {
  if (col_basis[node.rel_pos[0]].get() == nullptr) {
    Dense row_block = make_block_row(node, func, x, admis);
    // Construct U using the ID and remember the selected rows
    Dense Ut;
    std::tie(Ut, selected_rows[node.rel_pos[0]]) = one_sided_rid(
      row_block, rank+5, rank, true);
    transpose(Ut);
    col_basis[node.rel_pos[0]] = std::make_shared<Dense>(std::move(Ut));
  }
  if (row_basis[node.rel_pos[1]].get() == nullptr) {
    Dense col_block = make_block_col(node, func, x, admis);
    // Construct V using the ID and remember the selected cols
    Dense V;
    std::tie(V, selected_cols[node.rel_pos[1]]) = one_sided_rid(
      col_block, rank+5, rank);
    row_basis[node.rel_pos[1]] = std::make_shared<Dense>(std::move(V));
  }
  Dense D(node, func, x);
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
  void (*func)(
    Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
  ),
  std::vector<double>& x,
  int64_t rank,
  int64_t admis
) {
  if (col_basis[node.rel_pos[0]].get() == nullptr) {
    Dense row_block = make_block_row(node, func, x, admis);
    col_basis[node.rel_pos[0]] = std::make_shared<Dense>(
      LowRank(row_block, rank).U());
  }
  if (row_basis[node.rel_pos[1]].get() == nullptr) {
    Dense col_block = make_block_col(node, func, x, admis);
    row_basis[node.rel_pos[1]] = std::make_shared<Dense>(LowRank(
      col_block, rank).V());
  }
  Dense D(node, func, x);
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
  ClusterTree& node,
  void (*func)(
    Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
  ),
  std::vector<double>& x,
  int64_t rank,
  int64_t nleaf,
  int64_t admis,
  int64_t ni_level, int64_t nj_level,
  bool use_svd
) : Hierarchical(ni_level, nj_level) {
  // TODO All dense UH not allowed for now!
  assert(dim[0] > admis + 1 && dim[1] > admis+1);
  // TODO Only single leve allowed for now. Constructions and some operations
  // work for more levels, but LU does not yet (LR+=LR issue).
  assert(node.dim[0]/ni_level <= nleaf);
  assert(node.dim[1]/nj_level <= nleaf);
  assert(rank <= nleaf);
  // TODO For now only admis 0! gemm(D, LR, LR) and gemm(LR, D, LR) needed for
  // more.
  assert(admis == 0);
  col_basis.resize(dim[0]);
  row_basis.resize(dim[1]);
  std::vector<std::vector<int64_t>> selected_rows(dim[0]);
  std::vector<std::vector<int64_t>> selected_cols(dim[1]);
  node.split(dim[0], dim[1]);
  for (ClusterTree& child_node : node.children) {
    if (is_admissible(child_node, admis)) {
      if (use_svd) {
        (*this)[child_node.rel_pos] = construct_shared_block_svd(
          child_node, func, x, rank, admis);
      } else {
        (*this)[child_node.rel_pos] = construct_shared_block_id(
          child_node, func, x, selected_rows, selected_cols, rank, admis);
      }
    } else {
      if (is_leaf(child_node, nleaf)) {
        (*this)[child_node.rel_pos] = Dense(child_node, func, x);
      } else {
        (*this)[child_node.rel_pos] = UniformHierarchical(
          child_node, func, x, rank, nleaf, admis, ni_level, nj_level);
      }
    }
  }
}

UniformHierarchical::UniformHierarchical(
  void (*func)(
    Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
  ),
  std::vector<double>& x,
  int64_t ni, int64_t nj,
  int64_t rank,
  int64_t nleaf,
  int64_t admis,
  int64_t ni_level, int64_t nj_level,
  bool use_svd,
  int64_t i_begin, int64_t j_begin
) {
  ClusterTree node(ni, nj, i_begin, j_begin);
  *this = UniformHierarchical(
    node, func, x, rank, nleaf, admis, ni_level, nj_level, use_svd);
}

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
  abort();
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
  abort();
}

void UniformHierarchical::set_col_basis(int64_t i, int64_t j) {
  hicma::set_col_basis_omm((*this)(i, j), col_basis[i]);
}

void UniformHierarchical::set_row_basis(int64_t i, int64_t j) {
  hicma::set_row_basis_omm((*this)(i, j), row_basis[j]);
}

} // namespace hicma
