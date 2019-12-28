#include "hicma/classes/uniform_hierarchical.h"

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/low_rank_shared.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/LAPACK/id.h"
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
  int i_begin, int j_begin,
  int i_abs, int j_abs,
  int level
) : Hierarchical(ni_level, nj_level, i_abs, j_abs, level) {
  MM_INIT();
  if (!level) {
    assert(int(x.size()) == std::max(ni,nj));
    std::sort(x.begin(),x.end());
  }
  col_basis.resize(dim[0]);
  row_basis.resize(dim[1]);
  std::vector<std::vector<int>> selected_rows(dim[0]);
  std::vector<std::vector<int>> selected_cols(dim[1]);
  for (int i=0; i<dim[0]; i++) {
    for (int j=0; j<dim[1]; j++) {
      int ni_child = ni/dim[0];
      if ( i == dim[0]-1 ) ni_child = ni - (ni/dim[0]) * (dim[0]-1);
      int nj_child = nj/dim[1];
      if ( j == dim[1]-1 ) nj_child = nj - (nj/dim[1]) * (dim[1]-1);
      int i_begin_child = i_begin + ni/dim[0] * i;
      int j_begin_child = j_begin + nj/dim[1] * j;
      int i_abs_child = i_abs * dim[0] + i;
      int j_abs_child = j_abs * dim[1] + j;
      if (
          std::abs(i_abs_child - j_abs_child) <= admis // Check regular admissibility
          || (nj == 1 || ni == 1) ) { // Check if vector, and if so do not use LowRank
        if ( ni_child/ni_level < nleaf && nj_child/nj_level < nleaf ) {
          (*this)(i,j) = Dense(
            func,
            x,
            ni_child, nj_child,
            i_begin_child, j_begin_child,
            i_abs_child, j_abs_child,
            level+1
          );
        }
        else {
          (*this)(i,j) = UniformHierarchical(
            func,
            x,
            ni_child, nj_child,
            rank,
            nleaf,
            admis,
            ni_level, nj_level,
            i_begin_child, j_begin_child,
            i_abs_child, j_abs_child,
            level+1
          );
        }
      }
      else {
        if (col_basis[i].get() == nullptr) {
          // Create row block without admissible blocks
          Hierarchical row_block_h(1, std::pow(nj_level, level+1)-1-2*admis);
          // Note the ins counter!
          for (int j_b=0, ins=0; j_b<std::pow(nj_level, level+1); ++j_b) {
            if (std::abs(j_b - i_abs_child) > admis)
              row_block_h[ins++] = Dense(
                func,
                x,
                ni_child, nj_child,
                i_begin_child, nj_child*j_b
              );
          }
          Dense row_block(row_block_h);
          // col_basis[i] = std::make_shared<Dense>(LowRank(row_block, rank).U);
          // Construct U using the ID and remember the selected rows
          Dense Ut(rank, ni_child);
          transpose(row_block);
          selected_rows[i] = id(row_block, Ut, rank);
          transpose(Ut);
          col_basis[i] = std::make_shared<Dense>(std::move(Ut));
        }
        if (row_basis[j].get() == nullptr) {
          // Create col block without admissible blocks
          // TODO The number of blocks actually only works for admis=0!!! the
          // first admis columns have a different number of admissable blocks
          Hierarchical col_block_h(std::pow(ni_level, level+1)-1-2*admis, 1);
          // Note the ins counter!
          for (int i_b=0, ins=0; i_b<std::pow(ni_level, level+1); ++i_b) {
            if (std::abs(i_b - j_abs_child) > admis)
              col_block_h[ins++] = Dense(
                func,
                x,
                ni_child, nj_child,
                ni_child*i_b, j_begin_child
              );
          }
          Dense col_block(col_block_h);
          // row_basis[j] = std::make_shared<Dense>(LowRank(col_block, rank).V);
          // Construct V using the ID and remember the selected cols
          Dense V(rank, nj_child);
          selected_cols[j] = id(col_block, V, rank);
          row_basis[j] = std::make_shared<Dense>(std::move(V));
        }
        Dense D(
          func,
          x,
          ni_child, nj_child,
          i_begin_child, j_begin_child
        );
        Dense S(rank, rank);
        // Dense UD(col_basis[i]->dim[1], D.dim[1]);
        // gemm(*col_basis[i], D, UD, true, false, 1, 0);
        // gemm(UD, *row_basis[j], S, false, true, 1, 0);
        for (int ic=0; ic<rank; ++ic) {
          for (int jc=0; jc<rank; ++jc) {
            S(ic, jc) = D(selected_rows[i][ic], selected_cols[j][jc]);
          }
        }
        (*this)(i, j) = LowRankShared(
          S,
          col_basis[i], row_basis[j]
        );
      }
    }
  }
}

} // namespace hicma
