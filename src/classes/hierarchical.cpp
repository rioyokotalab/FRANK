#include "hicma/classes/hierarchical.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/index_range.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/functions.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <tuple>
#include <utility>


namespace hicma
{

std::unique_ptr<Node> Hierarchical::clone() const {
  return std::make_unique<Hierarchical>(*this);
}

std::unique_ptr<Node> Hierarchical::move_clone() {
  return std::make_unique<Hierarchical>(std::move(*this));
}

const char* Hierarchical::type() const { return "Hierarchical"; }

declare_method(Hierarchical&&, move_from_hierarchical, (virtual_<Node&>))

Hierarchical::Hierarchical(NodeProxy&& A)
: Hierarchical(move_from_hierarchical(A)) {}

define_method(Hierarchical&&, move_from_hierarchical, (Hierarchical& A)) {
  return std::move(A);
}

define_method(Hierarchical&&, move_from_hierarchical, (Node& A)) {
  omm_error_handler("move_from_hierarchical", {A}, __FILE__, __LINE__);
  abort();
}

Hierarchical::Hierarchical(const Node& node, int ni_level, int nj_level)
: row_range(0, get_n_rows(node)), col_range(0, get_n_cols(node)),
  dim{ni_level, nj_level}
{
  create_children();
  fill_hierarchical_from(*this, node);
}

define_method(
  void, fill_hierarchical_from, (Hierarchical& H, const Dense& A)
) {
  timing::start("fill_hierarchical_from(D)");
  for (int i=0; i<H.dim[0]; ++i) {
    for (int j=0; j<H.dim[1]; ++j) {
      H(i, j) = A.get_part(H.row_range[i], H.col_range[j]);
    }
  }
  timing::stop("fill_hierarchical_from(D)");
}

define_method(
  void, fill_hierarchical_from, (Hierarchical& H, const LowRank& A)
) {
  timing::start("fill_hierarchical_from(LR)");
  for (int i=0; i<H.dim[0]; ++i) {
    for (int j=0; j<H.dim[1]; ++j) {
      H(i, j) = A.get_part(H.row_range[i], H.col_range[j]);
    }
  }
  timing::stop("fill_hierarchical_from(LR)");
}

define_method(
  void, fill_hierarchical_from, (Hierarchical& H, const Node& A)
) {
  omm_error_handler("fill_hierarchical_from", {H, A}, __FILE__, __LINE__);
  abort();
}

Hierarchical::Hierarchical(int ni_level, int nj_level)
: dim{ni_level, nj_level} {
  create_children();
}

Hierarchical::Hierarchical(
  IndexRange row_range_, IndexRange col_range_,
  void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
  std::vector<double>& x,
  int rank,
  int nleaf,
  int admis,
  int ni_level, int nj_level,
  int i_begin, int j_begin,
  int i_abs, int j_abs
) : row_range(row_range_), col_range(col_range_) {
  dim[0] = std::min(ni_level, row_range.length);
  dim[1] = std::min(nj_level, col_range.length);
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
          (*this)(i, j) = Hierarchical(
            row_range[i], col_range[j],
            func, x, rank, nleaf, admis, ni_level, nj_level,
            i_begin+row_range[i].start, j_begin+col_range[j].start,
            i_abs_child, j_abs_child
          );
        }
      } else {
        (*this)(i, j) = LowRank(
          Dense(
            row_range[i], col_range[j], func, x,
            i_begin+row_range[i].start, j_begin+col_range[j].start
          ),
          rank
        );
      }
    }
  }
}

Hierarchical::Hierarchical(
  void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
  std::vector<double>& x,
  int ni, int nj,
  int rank,
  int nleaf,
  int admis,
  int ni_level, int nj_level,
  int i_begin, int j_begin
) : Hierarchical(
  IndexRange(0, ni), IndexRange(0, nj),
  func, x, rank, nleaf, admis, ni_level, nj_level, i_begin, j_begin
) {}

const NodeProxy& Hierarchical::operator[](int i) const {
  assert(dim[0] == 1 || dim[1] == 1);
  assert(i < (dim[0] != 1 ? dim[0] : dim[1]));
  return data[i];
}

NodeProxy& Hierarchical::operator[](int i) {
  assert(dim[0] == 1 || dim[1] == 1);
  assert(i < (dim[0] != 1 ? dim[0] : dim[1]));
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

std::vector<NodeProxy>::iterator Hierarchical::begin() { return data.begin(); }

std::vector<NodeProxy>::const_iterator Hierarchical::begin() const {
  return data.begin();
}

std::vector<NodeProxy>::iterator Hierarchical::end() { return data.end(); }

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
    int dim_Bi[2]{get_n_rows(B(i, 0)), get_n_cols(B(i, 0))};
    Dense Qbi(dim_Bi[0], dim_Bi[1]);
    for(int row=0; row<dim_Bi[0]; row++) {
      for(int col=0; col<dim_Bi[1]; col++) {
        Qbi(row, col) = Qb(rowOffset + row, col);
      }
    }
    // Moving should not make a difference. Why is this not auto-optimized?
    HQb(i, 0) = std::move(Qbi);
    rowOffset += dim_Bi[0];
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

void Hierarchical::restore_col(
  const Hierarchical& Sp, const Hierarchical& QL
) {
  assert(dim[1] == 1);
  assert(dim[0] == QL.dim[0]);
  assert(QL.dim[1] == 1);
  Hierarchical restoredA(dim[0], dim[1]);
  int curSpRow = 0;
  for(int i=0; i<dim[0]; i++) {
    restoredA(i, 0) = concat_columns((*this)(i, 0), Sp, QL(i, 0), curSpRow);
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

void Hierarchical::create_children() {
  row_range.split(dim[0]);
  col_range.split(dim[1]);
  data.resize(dim[0]*dim[1]);
}

bool Hierarchical::is_admissible(
  int i, int j, int i_abs, int j_abs, int dist_to_diag
) {
  bool admissible = false;
  // Main admissibility condition
  admissible |= (std::abs(i_abs - j_abs) <= dist_to_diag);
  // Vectors are never admissible
  admissible |= (row_range[i].length == 1 || col_range[j].length == 1);
  return admissible;
}

bool Hierarchical::is_leaf(int i, int j, int nleaf) {
  bool leaf = true;
  leaf &= (row_range[i].length/dim[0] < nleaf);
  leaf &= (col_range[j].length/dim[1] < nleaf);
  return leaf;
}

} // namespace hicma
