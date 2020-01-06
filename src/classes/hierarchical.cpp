#include "hicma/classes/hierarchical.h"

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/functions.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/LAPACK/qr.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/util/print.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>

#include "yorel/multi_methods.hpp"

namespace hicma {

  Hierarchical::Hierarchical() : dim{0, 0} { MM_INIT(); }

  Hierarchical::~Hierarchical() = default;

  Hierarchical::Hierarchical(const Hierarchical& A) {
    MM_INIT();
    *this = A;
  }

  Hierarchical& Hierarchical::operator=(const Hierarchical& A) = default;

  Hierarchical::Hierarchical(Hierarchical&& A) {
    MM_INIT();
    *this = std::move(A);
  }

  Hierarchical& Hierarchical::operator=(Hierarchical&& A) = default;

  std::unique_ptr<Node> Hierarchical::clone() const {
    return std::make_unique<Hierarchical>(*this);
  }

  std::unique_ptr<Node> Hierarchical::move_clone() {
    return std::make_unique<Hierarchical>(std::move(*this));
  }

  const char* Hierarchical::type() const { return "Hierarchical"; }

  Hierarchical::Hierarchical(
    int ni_level, int nj_level,
    int i_abs, int j_abs,
    int level
  ) : Node(i_abs, j_abs, level), dim{ni_level, nj_level} {
    MM_INIT();
    data.resize(dim[0]*dim[1]);
  }

  Hierarchical::Hierarchical(
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
    int ni_level, int nj_level
  ) : Node(node) {
    MM_INIT();
    dim[0] = std::min(ni_level, row_range.length);
    dim[1] = std::min(nj_level, col_range.length);
    data.resize(dim[0]*dim[1]);
    create_children(dim[0], dim[1]);
    for (NodeProxy& child : data) {
      if (is_addmissible(child, admis)) {
        if (is_leaf(child, nleaf)) {
          child = Dense(child, func, x);
        } else {
          child = Hierarchical(
            child, func, x, rank, nleaf, admis, ni_level, nj_level);
        }
      } else {
        child = LowRank(Dense(child, func, x), rank);
      }
    }
  }

  Hierarchical::Hierarchical(
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
  ) : Hierarchical(
    Node(
      i_abs, j_abs, level,
      IndexRange(i_begin, ni), IndexRange(j_begin, nj)
    ),
    func, x, rank, nleaf, admis, ni_level, nj_level
  ) {}

  Hierarchical::Hierarchical(const Dense& A, int m, int n)
  : Node(A), dim{m, n} {
    MM_INIT();
    data.resize(dim[0]*dim[1]);
    create_children(dim[0], dim[1]);
    for (NodeProxy& child : data) {
      child = A.get_part(child);
    }
  }

  Hierarchical::Hierarchical(const LowRank& A, int m, int n)
  : Node(A), dim{m, n} {
    MM_INIT();
    data.resize(dim[0]*dim[1]);
    create_children(dim[0], dim[1]);
    for (NodeProxy& child : data) {
      child = A.get_part(child);
    }
  }

  const NodeProxy& Hierarchical::operator[](int i) const {
    assert(dim[0] == 1 || dim[1] == 1);
    assert(i < dim[0] != 1 ? dim[0] : dim[1]);
    return data[i];
  }

  NodeProxy& Hierarchical::operator[](int i) {
    assert(dim[0] == 1 || dim[1] == 1);
    assert(i < dim[0] != 1 ? dim[0] : dim[1]);
    return data[i];
  }

  const NodeProxy& Hierarchical::operator()(int i, int j) const {
    assert(i < dim[0]);
    assert(j < dim[1]);
    return data[i*dim[1]+j];
  }

  NodeProxy& Hierarchical::operator()(int i, int j) {
    assert(i < dim[0]);
    assert(j < dim[1]);
    return data[i*dim[1]+j];
  }

  std::vector<NodeProxy>::iterator Hierarchical::begin() {
    return data.begin();
  }

  std::vector<NodeProxy>::const_iterator Hierarchical::begin() const {
    return data.begin();
  }

  std::vector<NodeProxy>::iterator Hierarchical::end() {
    return data.end();
  }

  std::vector<NodeProxy>::const_iterator Hierarchical::end() const {
    return data.end();
  }

  void Hierarchical::blr_col_qr(Hierarchical& Q, Hierarchical& R) {
    assert(dim[1] == 1);
    assert(Q.dim[0] == dim[0]);
    assert(Q.dim[1] == 1);
    assert(R.dim[0] == 1);
    assert(R.dim[1] == 1);
    Hierarchical Qu(dim[0], 1);
    Hierarchical B(dim[0], 1);
    for(int i=0; i<dim[0]; i++) {
      std::tie(Qu(i, 0), B(i, 0)) = make_left_orthogonal((*this)(i, 0));
    }
    Dense DB(B);
    Dense Qb(DB.dim[0], DB.dim[1]);
    Dense Rb(DB.dim[1], DB.dim[1]);
    qr(DB, Qb, Rb);
    R(0, 0) = std::move(Rb);
    //Slice Qb based on B
    Hierarchical HQb(B.dim[0], B.dim[1]);
    int rowOffset = 0;
    for(int i=0; i<HQb.dim[0]; i++) {
      int nrows = get_n_rows(B(i, 0));
      int ncols = get_n_cols(B(i, 0));
      Dense Qbi(nrows, ncols);
      for(int row=0; row<nrows; row++) {
        for(int col=0; col<ncols; col++) {
          Qbi(row, col) = Qb(rowOffset + row, col);
        }
      }
      // Using move should now make a difference. Why is this not auto-optimized?
      HQb(i, 0) = std::move(Qbi);
      rowOffset += nrows;
    }
    for(int i=0; i<dim[0]; i++) {
      gemm(Qu(i, 0), HQb(i, 0), Q(i, 0), 1, 0);
    }
  }

  void Hierarchical::split_col(Hierarchical& QL) {
    assert(dim[1] == 1);
    assert(QL.dim[0] == dim[0]);
    assert(QL.dim[1] == 1);
    int rows = 0;
    int cols = 1;
    for(int i=0; i<dim[0]; i++) {
      update_splitted_size((*this)(i, 0), rows, cols);
    }
    Hierarchical spA(rows, cols);
    int curRow = 0;
    for(int i=0; i<dim[0]; i++) {
      QL(i, 0) = split_by_column((*this)(i, 0), spA, curRow);
    }
    *this = std::move(spA);
  }

  void Hierarchical::restore_col(const Hierarchical& Sp, const Hierarchical& QL) {
    assert(dim[1] == 1);
    assert(dim[0] == QL.dim[0]);
    assert(QL.dim[1] == 1);
    Hierarchical restoredA(dim[0], dim[1]);
    int curSpRow = 0;
    for(int i=0; i<dim[0]; i++) {
      restoredA(i, 0) = concat_columns((*this)(i, 0), Sp, curSpRow, QL(i, 0));
    }
    *this = std::move(restoredA);
  }

  void Hierarchical::col_qr(int j, Hierarchical& Q, Hierarchical &R) {
    assert(Q.dim[0] == dim[0]);
    assert(Q.dim[1] == 1);
    assert(R.dim[0] == 1);
    assert(R.dim[1] == 1);
    bool split = false;
    Hierarchical Aj(dim[0], 1);
    for(int i=0; i<dim[0]; i++) {
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

  void Hierarchical::create_children(int n_row_splits, int n_col_splits) {
    std::vector<IndexRange> child_row_ranges = row_range.split(n_row_splits);
    std::vector<IndexRange> child_col_ranges = col_range.split(n_col_splits);
    for (int i=0; i<n_row_splits; i++) for (int j=0; j<n_col_splits; j++) {
      int i_abs_child = i_abs * n_row_splits + i;
      int j_abs_child = j_abs * n_col_splits + j;
      (*this)(i, j) = Node(
        i_abs_child, j_abs_child, level+1,
        child_row_ranges[i], child_col_ranges[j]
      );
    }
  }

  bool Hierarchical::is_addmissible(const Node& node, int dist_to_diag) {
    bool admissible = false;
    // Main admissibility condition
    admissible |= (std::abs(node.i_abs - node.j_abs) <= dist_to_diag);
    // Vectors are never admissible
    admissible |= (node.row_range.length == 1 || node.col_range.length == 1);
    return admissible;
  }

  bool Hierarchical::is_leaf(const Node& node, int nleaf) {
    bool leaf = true;
    leaf &= (node.row_range.length/dim[0] < nleaf);
    leaf &= (node.col_range.length/dim[1] < nleaf);
    return leaf;
  }

} // namespace hicma
