#include "hicma/classes/hierarchical.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"
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
#include <cstdint>
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

Hierarchical::Hierarchical(const Node& A, int64_t ni_level, int64_t nj_level)
: dim{ni_level, nj_level}
{
  data.resize(dim[0]*dim[1]);
  ClusterTree node(get_n_rows(A), get_n_cols(A));
  node.split(dim[0], dim[1]);
  fill_hierarchical_from(*this, A, node);
}

define_method(
  void, fill_hierarchical_from,
  (Hierarchical& H, const Dense& A, const ClusterTree& node)
) {
  timing::start("fill_hierarchical_from(D)");
  for (const ClusterTree& child_node : node.children) {
    H[child_node.rel_pos] = A.get_part(child_node);
  }
  timing::stop("fill_hierarchical_from(D)");
}

define_method(
  void, fill_hierarchical_from,
  (Hierarchical& H, const LowRank& A, const ClusterTree& node)
) {
  timing::start("fill_hierarchical_from(LR)");
  for (const ClusterTree& child_node : node.children) {
    H[child_node.rel_pos] = A.get_part(child_node);
  }
  timing::stop("fill_hierarchical_from(LR)");
}

define_method(
  void, fill_hierarchical_from,
  (Hierarchical& H, const Node& A, [[maybe_unused]] const ClusterTree& node)
) {
  omm_error_handler("fill_hierarchical_from", {H, A}, __FILE__, __LINE__);
  abort();
}

Hierarchical::Hierarchical(int64_t ni_level, int64_t nj_level)
: dim{ni_level, nj_level} {
  data.resize(dim[0]*dim[1]);
}

Hierarchical::Hierarchical(
  ClusterTree& node,
  void (*func)(
    Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
  ),
  std::vector<double>& x,
  int64_t rank,
  int64_t nleaf,
  int64_t admis,
  int64_t ni_level, int64_t nj_level
) {
  dim[0] = std::min(ni_level, node.dim[0]);
  dim[1] = std::min(nj_level, node.dim[1]);
  data.resize(dim[0]*dim[1]);
  node.split(dim[0], dim[1]);
  for (ClusterTree& child_node : node.children) {
    if (is_admissible(child_node, admis)) {
      (*this)[child_node.rel_pos] = LowRank(child_node, func, x, rank);
    } else {
      if (is_leaf(child_node, nleaf)) {
        (*this)[child_node.rel_pos] = Dense(child_node, func, x);
      } else {
        (*this)[child_node.rel_pos] = Hierarchical(
          child_node, func, x, rank, nleaf, admis, ni_level, nj_level);
      }
    }
  }
}

Hierarchical::Hierarchical(
  void (*func)(
    Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
  ),
  std::vector<double>& x,
  int64_t ni, int64_t nj,
  int64_t rank,
  int64_t nleaf,
  int64_t admis,
  int64_t ni_level, int64_t nj_level,
  int64_t i_begin, int64_t j_begin
) {
  ClusterTree root(ni, nj, i_begin, j_begin);
  *this = Hierarchical(root, func, x, rank, nleaf, admis, ni_level, nj_level);
}

const NodeProxy& Hierarchical::operator[](std::array<int64_t, 2> pos) const {
  return (*this)(pos[0], pos[1]);
}

NodeProxy& Hierarchical::operator[](std::array<int64_t, 2> pos) {
  return (*this)(pos[0], pos[1]);
}

const NodeProxy& Hierarchical::operator[](int64_t i) const {
  assert(dim[0] == 1 || dim[1] == 1);
  assert(i < (dim[0] != 1 ? dim[0] : dim[1]));
  return data[i];
}

NodeProxy& Hierarchical::operator[](int64_t i) {
  assert(dim[0] == 1 || dim[1] == 1);
  assert(i < (dim[0] != 1 ? dim[0] : dim[1]));
  return data[i];
}

const NodeProxy& Hierarchical::operator()(int64_t i, int64_t j) const {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return data[i*dim[1]+j];
}

NodeProxy& Hierarchical::operator()(int64_t i, int64_t j) {
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
  admissible &= (std::abs(node.abs_pos[0] - node.abs_pos[1]) > dist_to_diag);
  // Vectors are never admissible
  admissible &= (node.dim[0] > 1 && node.dim[1] > 1);
  return admissible;
}

bool Hierarchical::is_leaf(const ClusterTree& node, int64_t nleaf) {
  bool leaf = true;
  leaf &= (node.dim[0] <= nleaf);
  leaf &= (node.dim[1] <= nleaf);
  return leaf;
}

} // namespace hicma
