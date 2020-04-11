#include "hicma/classes/dense.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/hierarchical.h"
#include "hicma/classes/index_range.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <memory>
#include <utility>
#include <vector>


namespace hicma
{

Dense::Dense(const Dense& A)
: Node(A), dim{A.dim[0], A.dim[1]}, stride(A.dim[1]) {
  // TODO Make this unnecessary by changing DenseView (sibling not child?).
  timing::start("Dense cctor");
  data.resize(dim[0]*dim[1], 0);
  for (int i=0; i<dim[0]; i++) {
    for (int j=0; j<dim[1]; j++) {
      (*this)(i, j) = A(i, j);
    }
  }
  timing::stop("Dense cctor");
}

std::unique_ptr<Node> Dense::clone() const {
  return std::make_unique<Dense>(*this);
}

std::unique_ptr<Node> Dense::move_clone() {
  return std::make_unique<Dense>(std::move(*this));
}

const char* Dense::type() const { return "Dense"; }

Dense::Dense(const Node& A)
: Node(A), dim{get_n_rows(A), get_n_cols(A)}, stride(dim[1]) {
  data.resize(dim[0]*dim[1], 0);
  fill_dense_from(A, *this);
}

define_method(void, fill_dense_from, (const Hierarchical& A, Dense& B)) {
  timing::start("make_dense(H)");
  NoCopySplit BH(B, A);
  for (int i=0; i<A.dim[0]; i++) {
    for (int j=0; j<A.dim[1]; j++) {
      fill_dense_from(A(i, j), BH(i, j));
    }
  }
  timing::stop("make_dense(H)");
}

define_method(void, fill_dense_from, (const LowRank& A, Dense& B)) {
  timing::start("make_dense(LR)");
  gemm(gemm(A.U(), A.S()), A.V(), B, 1, 0);
  timing::stop("make_dense(LR)");
}

define_method(void, fill_dense_from, (const Dense& A, Dense& B)) {
  assert(A.dim[0] == B.dim[0]);
  assert(A.dim[1] == B.dim[1]);
  for (int i=0; i<A.dim[0]; i++) {
    for (int j=0; j<A.dim[1]; j++) {
      B(i, j) = A(i, j);
    }
  }
}

define_method(void, fill_dense_from, (const Node& A, Node& B)) {
  omm_error_handler("fill_dense_from", {A, B}, __FILE__, __LINE__);
  abort();
}

declare_method(Dense&&, move_from_dense, (virtual_<Node&>))

Dense::Dense(NodeProxy&& A) : Dense(move_from_dense(A)) {}

define_method(Dense&&, move_from_dense, (Dense& A)) {
  return std::move(A);
}

define_method(Dense&&, move_from_dense, (Node& A)) {
  omm_error_handler("move_from_dense", {A}, __FILE__, __LINE__);
  abort();
}

Dense::Dense(int m, int n)
: dim{m, n}, stride(n) {
  timing::start("Dense alloc");
  data.resize(dim[0]*dim[1], 0);
  timing::stop("Dense alloc");
}

Dense::Dense(
  const IndexRange& row_range, const IndexRange& col_range,
  void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
  std::vector<double>& x,
  int i_begin, int j_begin
) : dim{row_range.length, col_range.length}, stride(dim[1]) {
  // TODO This function is still aware of hierarchical matrix idea. Change?
  // Related to making helper class for construction.
  data.resize(dim[0]*dim[1]);
  func(*this, x, i_begin, j_begin);
}

Dense::Dense(
  void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
  std::vector<double>& x,
  int ni, int nj,
  int i_begin, int j_begin
) : Dense(IndexRange(0, ni), IndexRange(0, nj), func, x, i_begin, j_begin) {}

Dense::Dense(
  void (*func)(
    std::vector<double>& data,
    std::vector<std::vector<double>>& x,
    const int& ni, const int& nj,
    const int& i_begin, const int& j_begin
  ),
  std::vector<std::vector<double>>& x,
  const int ni, const int nj,
  const int i_begin, const int j_begin
) : dim{ni, nj}, stride(nj), data(dim[0]*dim[1]) {
  func(data, x, ni, nj, i_begin, j_begin);
}

const Dense& Dense::operator=(const double a) {
  for (int i=0; i<dim[0]; i++) {
    for (int j=0; j<dim[1]; j++) {
      (*this)(i, j) = a;
    }
  }
  return *this;
}

double* Dense::get_pointer() { return &data[0]; }

const double* Dense::get_pointer() const { return &data[0]; }

double& Dense::operator[](int i) {
  assert(dim[0] == 1 || dim[1] == 1);
  if (dim[0] == 1) {
    assert(i < dim[1]);
    return get_pointer()[i];
  } else {
    assert(i < dim[0]);
    return get_pointer()[i*stride];
  }
}

const double& Dense::operator[](int i) const {
  assert(dim[0] == 1 || dim[1] == 1);
  if (dim[0] == 1) {
    assert(i < dim[1]);
    return get_pointer()[i];
  } else {
    assert(i < dim[0]);
    return get_pointer()[i*stride];
  }
}

double& Dense::operator()(int i, int j) {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return get_pointer()[i*stride+j];
}

const double& Dense::operator()(int i, int j) const {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return get_pointer()[i*stride+j];
}

double* Dense::operator&() { return get_pointer(); }

const double* Dense::operator&() const { return get_pointer(); }

int Dense::size() const { return dim[0] * dim[1]; }

void Dense::resize(int dim0, int dim1) {
  assert(dim0 <= dim[0]);
  assert(dim1 <= dim[1]);
  timing::start("Dense resize");
  if (dim0 == dim[0] && dim1 == dim[1]) {
    timing::stop("Dense resize");
    return;
  }
  for (int i=0; i<dim0; i++) {
    for (int j=0; j<dim1; j++) {
      // TODO this is the only place where data is used directly now. Would be
      // better not to use it and somehow use the regular index operator
      // efficiently.
      data[i*dim1+j] = (*this)(i, j);
    }
  }
  dim[0] = dim0;
  dim[1] = dim1;
  stride = dim[1];
  data.resize(dim[0]*dim[1]);
  timing::stop("Dense resize");
}

Dense Dense::transpose() const {
  Dense A(dim[1], dim[0]);
  for (int i=0; i<dim[0]; i++) {
    for (int j=0; j<dim[1]; j++) {
      A(j,i) = (*this)(i,j);
    }
  }
  return A;
}

void Dense::transpose() {
  assert(stride = dim[1]);
  Dense Copy(*this);
  std::swap(dim[0], dim[1]);
  stride = dim[1];
  for(int i=0; i<dim[0]; i++) {
    for(int j=0; j<dim[1]; j++) {
      (*this)(i, j) = Copy(j, i);
    }
  }
}

Dense Dense::get_part(
  const IndexRange& row_range, const IndexRange& col_range
) const {
  assert(row_range.start+row_range.length <= dim[0]);
  assert(col_range.start+col_range.length <= dim[1]);
  Dense A(row_range.length, col_range.length);
  for (int i=0; i<A.dim[0]; i++) {
    for (int j=0; j<A.dim[1]; j++) {
      A(i, j) = (*this)(i+row_range.start, j+col_range.start);
    }
  }
  return A;
}

} // namespace hicma
