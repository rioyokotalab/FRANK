#include "hicma/classes/uniform_hierarchical.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/index_range.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/randomized_factorizations.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/operations/misc/transpose.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
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
  for (int i=0; i<dim[0]; i++) {
    for (int j=0; j<dim[1]; j++) {
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
  int ni_level, int nj_level
) : Hierarchical(ni_level, nj_level) {
  col_basis.resize(ni_level);
  row_basis.resize(nj_level);
}

Dense UniformHierarchical::make_block_row(
  int row, int i_abs, int j_abs,
  void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
  std::vector<double>& x,
  int admis,
  int i_begin, int j_begin
) {
  // Create row block without admissible blocks
  int n_non_admissible_blocks = 0;
  int row_abs = i_abs*dim[0]+row;
  for (int j_b=0; j_b<dim[1]; ++j_b) {
    if (!is_admissible(row, j_b, row_abs, j_abs*dim[1]+j_b, admis)) {
      n_non_admissible_blocks++;
    }
  }
  Hierarchical block_row_h(1, n_non_admissible_blocks);
  // Note the ins counter!
  // TODO j_b goes across entire matrix, outside of confines of *this.
  // Find way to combine this with create_children!
  for (int j_b=0, ins=0; j_b<dim[1]; ++j_b) {
    if (!is_admissible(row, j_b, row_abs, j_abs*dim[1]+j_b, admis)) {
      block_row_h[ins++] = Dense(
        row_range[row], col_range[j_b], func, x,
        i_begin+row_range[row].start, j_begin+col_range[j_b].start
      );
    }
  }
  return Dense(block_row_h);
}

Dense UniformHierarchical::make_block_col(
  int col, int i_abs, int j_abs,
  void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
  std::vector<double>& x,
  int admis,
  int i_begin, int j_begin
) {
  // Create col block without admissible blocks
  int n_non_admissible_blocks = 0;
  int col_abs = j_abs*dim[1]+col;
  for (int i_b=0; i_b<dim[0]; ++i_b) {
    if (!is_admissible(i_b, col, i_abs*dim[0]+i_b, col_abs, admis)) {
      n_non_admissible_blocks++;
    }
  }
  Hierarchical block_col_h(n_non_admissible_blocks, 1);
  // Note the ins counter!
  for (int i_b=0, ins=0; i_b<dim[0]; ++i_b) {
    if (!is_admissible(i_b, col, i_abs*dim[0]+i_b, col_abs, admis)) {
      block_col_h[ins++] = Dense(
        row_range[i_b], col_range[col], func, x,
        i_begin+row_range[i_b].start, j_begin+col_range[col].start
      );
    }
  }
  return Dense(block_col_h);
}

LowRankShared UniformHierarchical::construct_shared_block_id(
  int i, int j, int i_abs, int j_abs,
  std::vector<std::vector<int>>& selected_rows,
  std::vector<std::vector<int>>& selected_cols,
  void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
  std::vector<double>& x,
  int rank,
  int admis,
  int i_begin, int j_begin
) {
  if (col_basis[i].get() == nullptr) {
    Dense row_block = make_block_row(
      i, i_abs, j_abs, func, x, admis, i_begin, j_begin);
    // Construct U using the ID and remember the selected rows
    Dense Ut;
    std::tie(Ut, selected_rows[i]) = one_sided_rid(
      row_block, rank+5, rank, true);
    transpose(Ut);
    col_basis[i] = std::make_shared<Dense>(std::move(Ut));
  }
  if (row_basis[j].get() == nullptr) {
    Dense col_block = make_block_col(
      j, i_abs, j_abs, func, x, admis, i_begin, j_begin);
    // Construct V using the ID and remember the selected cols
    Dense V;
    std::tie(V, selected_cols[j]) = one_sided_rid(
      col_block, rank+5, rank);
    row_basis[j] = std::make_shared<Dense>(std::move(V));
  }
  Dense D(
    row_range[i], col_range[j], func, x,
    i_begin+row_range[i].start, j_begin+col_range[j].start
  );
  Dense S(rank, rank);
  for (int ic=0; ic<rank; ++ic) {
    for (int jc=0; jc<rank; ++jc) {
      S(ic, jc) = D(selected_rows[i][ic], selected_cols[j][jc]);
    }
  }
  return LowRankShared(
    S,
    col_basis[i], row_basis[j]
  );
}

LowRankShared UniformHierarchical::construct_shared_block_svd(
  int i, int j, int i_abs, int j_abs,
  void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
  std::vector<double>& x,
  int rank,
  int admis,
  int i_begin, int j_begin
) {
  if (col_basis[i].get() == nullptr) {
    Dense row_block = make_block_row(
      i, i_abs, j_abs, func, x, admis, i_begin, j_begin);
    col_basis[i] = std::make_shared<Dense>(LowRank(row_block, rank).U());
  }
  if (row_basis[j].get() == nullptr) {
    Dense col_block = make_block_col(
      j, i_abs, j_abs, func, x, admis, i_begin, j_begin);
    row_basis[j] = std::make_shared<Dense>(LowRank(col_block, rank).V());
  }
  Dense D(
    row_range[i], col_range[j], func, x,
    i_begin+row_range[i].start, j_begin+col_range[j].start
  );
  Dense S(rank, rank);
  Dense UD(col_basis[i]->dim[1], D.dim[1]);
  gemm(*col_basis[i], D, UD, true, false, 1, 0);
  gemm(UD, *row_basis[j], S, false, true, 1, 0);
  return LowRankShared(
    S,
    col_basis[i], row_basis[j]
  );
}

UniformHierarchical::UniformHierarchical(
  IndexRange row_range_, IndexRange col_range_,
  void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
  std::vector<double>& x,
  int rank,
  int nleaf,
  int admis,
  int ni_level, int nj_level,
  bool use_svd,
  int i_begin, int j_begin,
  int i_abs, int j_abs
) : Hierarchical(ni_level, nj_level) {
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
  row_range = row_range_;
  col_range = col_range_;
  col_basis.resize(dim[0]);
  row_basis.resize(dim[1]);
  std::vector<std::vector<int>> selected_rows(dim[0]);
  std::vector<std::vector<int>> selected_cols(dim[1]);
  create_children();
  for (int i=0; i<dim[0]; ++i) {
    for (int j=0; j<dim[1]; ++j) {
      // TODO Move into contsructor-helper-class?
      int i_abs_child = i_abs*dim[0]+i;
      int j_abs_child = j_abs*dim[1]+j;
      if (is_admissible(i, j, i_abs_child, j_abs_child, admis)) {
        if (is_leaf(i, j, nleaf)) {
          (*this)(i, j) = Dense(
            row_range[i], col_range[j], func, x,
            i_begin+row_range[i].start, j_begin+col_range[j].start
          );
        } else {
          (*this)(i, j) = UniformHierarchical(
            row_range[i], col_range[j],
            func, x, rank, nleaf, admis, ni_level, nj_level, use_svd,
            i_begin+row_range[i].start, j_begin+col_range[j].start,
            i_abs_child, j_abs_child
          );
        }
      } else {
        if (use_svd) {
          (*this)(i, j) = construct_shared_block_svd(
            i, j, i_abs, j_abs, func, x, rank, admis, i_begin, j_begin
          );
        } else {
          (*this)(i, j) = construct_shared_block_id(
            i, j, i_abs, j_abs,
            selected_rows, selected_cols, func, x, rank, admis, i_begin, j_begin
          );
        }
      }
    }
  }
}

UniformHierarchical::UniformHierarchical(
  void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
  std::vector<double>& x,
  int ni, int nj,
  int rank,
  int nleaf,
  int admis,
  int ni_level, int nj_level,
  bool use_svd,
  int i_begin, int j_begin,
  int i_abs, int j_abs
) : UniformHierarchical(
  IndexRange(0, ni), IndexRange(0, nj),
  func, x, rank, nleaf, admis, ni_level, nj_level, use_svd,
  i_begin, j_begin, i_abs, j_abs
) {}

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

void UniformHierarchical::set_col_basis(int i, int j) {
  hicma::set_col_basis_omm((*this)(i, j), col_basis[i]);
}

void UniformHierarchical::set_row_basis(int i, int j) {
  hicma::set_row_basis_omm((*this)(i, j), row_basis[j]);
}

} // namespace hicma
