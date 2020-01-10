#include "hicma/classes/uniform_hierarchical.h"

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/randomized/rid.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/operations/misc/transpose.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <utility>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

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

MULTI_METHOD(
  move_from_uniform_hierarchical, UniformHierarchical, virtual_<Node>&);

BEGIN_SPECIALIZATION(
  move_from_uniform_hierarchical, UniformHierarchical,
  UniformHierarchical& A
) {
  return std::move(A);
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  move_from_uniform_hierarchical, UniformHierarchical,
  Node& A
) {
  std::cout << "Cannot move to UniformHierarchical from " << A.type() << "!" << std::endl;
  abort();
} END_SPECIALIZATION;

UniformHierarchical::UniformHierarchical(NodeProxy&& A) {
  *this = move_from_uniform_hierarchical(A);
}

UniformHierarchical::UniformHierarchical(
  const Node& node, int ni_level, int nj_level
) : Hierarchical(node, ni_level, nj_level, true) {
  MM_INIT();
  col_basis.resize(ni_level);
  row_basis.resize(nj_level);
}

UniformHierarchical::UniformHierarchical(
  const Node& node,
  void (*func)(Dense& A, std::vector<double>& x),
  std::vector<double>& x,
  int rank,
  int nleaf,
  int admis,
  int ni_level, int nj_level,
  bool use_svd
) : Hierarchical(node, ni_level, nj_level, true) {
  MM_INIT();
  // TODO All dense UH not allowed for now!
  assert(dim[0] > admis + 1 && dim[1] > admis+1);
  // TODO Only single leve allowed for now. Constructions and some operations
  // work for more levels, but LU does not yet (LR+=LR issue).
  assert(row_range.length/ni_level <= nleaf);
  assert(col_range.length/nj_level <= nleaf);
  assert(rank <= nleaf);
  // TODO For now only admis 0! gemm(D, LR, LR) and gemm(LR, D, LR) needed for
  // more.
  assert(admis == 0);
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
          Dense Ut;
          transpose(row_block);
          std::tie(Ut, selected_rows[i]) = one_sided_rid(row_block, rank+5, rank);
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
          Dense V;
          std::tie(V, selected_cols[j]) = one_sided_rid(col_block, rank+5, rank);
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
        child,
        S,
        col_basis[i], row_basis[j]
      );
    }
  }
}

UniformHierarchical::UniformHierarchical(
  void (*func)(Dense& A, std::vector<double>& x),
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

// MULTI_METHOD(is_LowRankShared, bool, const virtual_<Node>&);

// BEGIN_SPECIALIZATION(is_LowRankShared, bool, const LowRankShared&) {
//   return true;
// } END_SPECIALIZATION;

// BEGIN_SPECIALIZATION(is_LowRankShared, bool, const Node&) {
//   return false;
// } END_SPECIALIZATION;

// const UniformHierarchical& UniformHierarchical::operator+=(const UniformHierarchical& A) {
//   // Model after LR+=LR!
//   assert(dim[0] == A.dim[0]);
//   assert(dim[1] == A.dim[1]);

//   std::vector<std::shared_ptr<Dense>> new_col_basis(dim[0]), new_row_basis(dim[1]);

//   for (int i=0; i<dim[0]; i++) {
//     for (int j=0; j<dim[1]; j++) {
//       // Use LR merge on row and column basis to get Inner and Outers.
//       // Create stack of InnerUxSxInnerV for block column of *this.
//       // Compute SVD on the stack and use the row basis as shared InnerV.
//       // Do equivalent for block row and get shared InnerU.
//       // Use shared InnerU and InnerV to get S for all blocks.
//       // Use shared InnerU, InnerV, OuterU and OuterV to compute new shared row
//       // and column bases.

//       // if (is_LowRankShared((*this)(i, j))) {
//       //   int n_non_admissible_blocks = 0;
//       //   for (int i_b=0; i_b<dim[0]; ++i_b) {
//       //     if (is_LowRankShared((*this)(i_b, j))) n_non_admissible_blocks++;
//       //   }
//       //   Hierarchical col_block_h(n_non_admissible_blocks, 1);
//       //   // Note the ins counter!
//       //   for (int i_b=0, ins=0; i_b<dim[0]; ++i_b) {
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

void UniformHierarchical::copy_col_basis(const UniformHierarchical& A) {
  assert(dim[0] == A.dim[0]);
  for (int i=0; i<dim[1]; i++) {
    col_basis[i] = std::make_shared<Dense>(A.get_col_basis(i));
  }
}

void UniformHierarchical::copy_row_basis(const UniformHierarchical& A) {
  assert(dim[1] == A.dim[1]);
  for (int j=0; j<dim[1]; j++) {
    row_basis[j] = std::make_shared<Dense>(A.get_row_basis(j));
  }
}


MULTI_METHOD(
  set_col_basis_omm, void,
  virtual_<Node>& A, std::shared_ptr<Dense> basis
);

BEGIN_SPECIALIZATION(
  set_col_basis_omm, void,
  LowRankShared& A, std::shared_ptr<Dense> basis
) {
  A.U = basis;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  set_col_basis_omm, void,
  Dense& A, std::shared_ptr<Dense> basis
) {
  // Do nothing
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  set_col_basis_omm, void,
  UniformHierarchical& A, std::shared_ptr<Dense> basis
) {
  // Do nothing
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  set_col_basis_omm, void,
  Node& A, std::shared_ptr<Dense> basis
) {
  std::cout << "Cannot set column basis on " << A.type() << "!" << std::endl;
  abort();
} END_SPECIALIZATION;

MULTI_METHOD(
  set_row_basis_omm, void,
  virtual_<Node>& A, std::shared_ptr<Dense> basis
);

BEGIN_SPECIALIZATION(
  set_row_basis_omm, void,
  LowRankShared& A, std::shared_ptr<Dense> basis
) {
  A.V = basis;
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  set_row_basis_omm, void,
  Dense& A, std::shared_ptr<Dense> basis
) {
  // Do nothing
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  set_row_basis_omm, void,
  UniformHierarchical& A, std::shared_ptr<Dense> basis
) {
  // Do nothing
} END_SPECIALIZATION;

BEGIN_SPECIALIZATION(
  set_row_basis_omm, void,
  Node& A, std::shared_ptr<Dense> basis
) {
  std::cout << "Cannot set row basis on " << A.type() << "!" << std::endl;
  abort();
} END_SPECIALIZATION;

void UniformHierarchical::set_col_basis(int i, int j) {
  hicma::set_col_basis_omm((*this)(i, j), col_basis[i]);
}

void UniformHierarchical::set_row_basis(int i, int j) {
  hicma::set_row_basis_omm((*this)(i, j), row_basis[j]);
}

} // namespace hicma
