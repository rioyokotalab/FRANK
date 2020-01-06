#include "hicma/classes/uniform_hierarchical.h"

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/LAPACK/id.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/operations/misc/transpose.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <utility>

#include "yorel/multi_methods.hpp"

namespace hicma
{

UniformHierarchical::UniformHierarchical() : Hierarchical() { MM_INIT(); }

UniformHierarchical::~UniformHierarchical() = default;

UniformHierarchical::UniformHierarchical(const UniformHierarchical& A){
  MM_INIT();
  *this = A;
}

UniformHierarchical&
UniformHierarchical::operator=(const UniformHierarchical& A) = default;

UniformHierarchical::UniformHierarchical(UniformHierarchical&& A) {
  MM_INIT();
  *this = std::move(A);
}

UniformHierarchical&
UniformHierarchical::operator=(UniformHierarchical&& A) = default;

std::unique_ptr<Node> UniformHierarchical::clone() const {
  return std::make_unique<UniformHierarchical>(*this);
}

std::unique_ptr<Node> UniformHierarchical::move_clone() {
  return std::make_unique<UniformHierarchical>(std::move(*this));
}

const char* UniformHierarchical::type() const { return "UniformHierarchical"; }

UniformHierarchical::UniformHierarchical(
  const Node& node,
  void (*func)(
    std::vector<double>& data,
    std::vector<double>& x,
    int ni, int nj,
    int i_begin, int j_begin
  ),
  std::vector<double>& x,
  int rank,
  int nleaf,
  int admis,
  int ni_level, int nj_level,
  bool use_svd
) : Hierarchical(node, nj_level, nj_level, true) {
  MM_INIT();
  if (!level) {
    assert(x.size() == std::max(node.row_range.length, node.col_range.length));
    std::sort(x.begin(),x.end());
  }
  col_basis.resize(dim[0]);
  row_basis.resize(dim[1]);
  std::vector<std::vector<int>> selected_rows(dim[0]);
  std::vector<std::vector<int>> selected_cols(dim[1]);
  create_children();
  for (NodeProxy& child : *this) {
    if (is_admissible(child, admis)) {
      if (is_leaf(child, nleaf)) {
        child = Dense(child, func, x);
      } else {
        child = UniformHierarchical(
          child, func, x, rank, nleaf, admis, ni_level, nj_level, use_svd);
      }
    } else {
      int i, j;
      std::tie(i, j) = get_rel_pos_child(child);
      if (col_basis[i].get() == nullptr) {
        // Create row block without admissible blocks
        int n_non_admissible_blocks = 0;
        for (int j_b=0; j_b<dim[1]; ++j_b) {
          if (!is_admissible((*this)(i, j_b), admis)) {
            n_non_admissible_blocks++;
          }
        }
        Hierarchical row_block_h(1, n_non_admissible_blocks);
        // Note the ins counter!
        // TODO j_b goes across entire matrix, outside of confines of *this.
        // Find way to combine this with create_children!
        for (int j_b=0, ins=0; j_b<dim[1]; ++j_b) {
          if (!is_admissible((*this)(i, j_b), admis))
            row_block_h[ins++] = Dense((*this)(i, j_b), func, x);
        }
        Dense row_block(row_block_h);
        if (use_svd) {
          col_basis[i] = std::make_shared<Dense>(LowRank(row_block, rank).U);
        } else {
          // Construct U using the ID and remember the selected rows
          Dense Ut(rank, get_n_cols(row_block));
          transpose(row_block);
          selected_rows[i] = id(row_block, Ut, rank);
          transpose(Ut);
          col_basis[i] = std::make_shared<Dense>(std::move(Ut));
        }
      }
      if (row_basis[j].get() == nullptr) {
        // Create col block without admissible blocks
        int n_non_admissible_blocks = 0;
        for (int i_b=0; i_b<dim[0]; ++i_b) {
          if (!is_admissible((*this)(i_b, j), admis)) n_non_admissible_blocks++;
        }
        Hierarchical col_block_h(n_non_admissible_blocks, 1);
        // Note the ins counter!
        for (int i_b=0, ins=0; i_b<dim[0]; ++i_b) {
          if (!is_admissible((*this)(i_b, j), admis))
            col_block_h[ins++] = Dense((*this)(i_b, j), func, x);
        }
        Dense col_block(col_block_h);
        if (use_svd) {
          row_basis[j] = std::make_shared<Dense>(LowRank(col_block, rank).V);
        } else {
          // Construct V using the ID and remember the selected cols
          Dense V(rank, get_n_cols(col_block));
          selected_cols[j] = id(col_block, V, rank);
          row_basis[j] = std::make_shared<Dense>(std::move(V));
        }
      }
      Dense D(child, func, x);
      Dense S(rank, rank);
      if (use_svd) {
        Dense UD(col_basis[i]->dim[1], D.dim[1]);
        gemm(*col_basis[i], D, UD, true, false, 1, 0);
        gemm(UD, *row_basis[j], S, false, true, 1, 0);
      } else {
        for (int ic=0; ic<rank; ++ic) {
          for (int jc=0; jc<rank; ++jc) {
            S(ic, jc) = D(selected_rows[i][ic], selected_cols[j][jc]);
          }
        }
      }
      (*this)(i, j) = LowRankShared(
        S,
        col_basis[i], row_basis[j]
      );
    }
  }
}

UniformHierarchical::UniformHierarchical(
  void (*func)(
    std::vector<double>& data,
    std::vector<double>& x,
    int ni, int nj,
    int i_begin, int j_begin
  ),
  std::vector<double>& x,
  int ni, int nj,
  int rank,
  int nleaf,
  int admis,
  int ni_level, int nj_level,
  bool use_svd,
  int i_begin, int j_begin,
  int i_abs, int j_abs,
  int level
) : UniformHierarchical(
  Node(i_abs, j_abs, level, IndexRange(i_begin, ni), IndexRange(j_begin, nj)),
  func, x, rank, nleaf, admis, ni_level, nj_level, use_svd
) {}

Dense& UniformHierarchical::get_row_basis(int i) {
  assert(i < dim[0]);
  return *row_basis[i];
}

const Dense& UniformHierarchical::get_row_basis(int i) const {
  assert(i < dim[0]);
  return *row_basis[i];
}

Dense& UniformHierarchical::get_col_basis(int j) {
  assert(j < dim[1]);
  return *col_basis[j];
}

const Dense& UniformHierarchical::get_col_basis(int j) const {
  assert(j < dim[0]);
  return *col_basis[j];
}

} // namespace hicma
