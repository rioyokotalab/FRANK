#include "hicma/classes/dense.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

namespace hicma {

  Dense::Dense(const Dense& A)
  : Node(A), dim{A.dim[0], A.dim[1]}, stride(A.dim[1]) {
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

  Dense::Dense(const Node& A, bool only_node)
  : Node(A), dim{A.row_range.length, A.col_range.length}, stride(dim[1]) {
    if (!only_node) {
      *this = make_dense(A);
    }
  }

  define_method(Dense, make_dense, (const Hierarchical& A)) {
    timing::start("make_dense(H)");
    Dense B(get_n_rows(A), get_n_cols(A));
    // TODO This loop copies the data multiple times
    int i_begin = 0;
    for (int i=0; i<A.dim[0]; i++) {
      int j_begin = 0;
      for (int j=0; j<A.dim[1]; j++) {
        Dense AD = Dense(A(i,j));
        for (int ic=0; ic<AD.dim[0]; ic++) {
          for (int jc=0; jc<AD.dim[1]; jc++) {
            B(ic+i_begin, jc+j_begin) = AD(ic,jc);
          }
        }
        j_begin += AD.dim[1];
      }
      i_begin += get_n_rows(A(i, 0));
    }
    timing::stop("make_dense(H)");
    // TODO Consider return with std::move. Test if the copy is elided!!
    return B;
  }

  define_method(Dense, make_dense, (const LowRank& A)) {
    timing::start("make_dense(LR)");
    Dense B(A.dim[0], A.dim[1]);
    Dense UxS(A.dim[0], A.rank);
    gemm(A.U(), A.S(), UxS, 1, 0);
    gemm(UxS, A.V(), B, 1, 0);
    // TODO Consider return with std::move. Test if the copy is elided!!
    timing::stop("make_dense(LR)");
    return B;
  }

  define_method(Dense, make_dense, (const Dense& A)) {
    // TODO Consider return with std::move. Test if the copy is elided!!
    return Dense(A);
  }

  define_method(Dense, make_dense, (const Node& A)) {
    std::cout << "Cannot create Dense from " << A.type() << "!" << std::endl;
    abort();
  }

  declare_method(Dense, move_from_dense, (virtual_<Node&>));

  Dense::Dense(NodeProxy&& A) {
    *this = move_from_dense(A);
  }

  define_method(
    Dense, move_from_dense,
    (Dense& A)
  ) {
    return std::move(A);
  }

  define_method(
    Dense, move_from_dense,
    (Node& A)
  ) {
    std::cout << "Cannot move to Dense from " << A.type() << "!" << std::endl;
    abort();
  }

  Dense::Dense(
    int m, int n,
    int i_abs, int j_abs,
    int level
  ) : Dense(
    Node(i_abs, j_abs, level, IndexRange(0, m), IndexRange(0, n)),
    true
  ) {
    timing::start("Dense alloc");
    data.resize(dim[0]*dim[1], 0);
    timing::stop("Dense alloc");
  }

  Dense::Dense(
    const Node& node,
    void (*func)(Dense& A, std::vector<double>& x),
    std::vector<double>& x
  ) : Node(node), dim{node.row_range.length, node.col_range.length}, stride(dim[1]) {
    data.resize(dim[0]*dim[1]);
    func(*this, x);
  }

  Dense::Dense(
    void (*func)(Dense& A, std::vector<double>& x),
    std::vector<double>& x,
    int ni, int nj,
    int i_begin, int j_begin,
    int i_abs, int j_abs,
    int level
  ) : Dense(
    Node(i_abs, j_abs, level, IndexRange(i_begin, ni), IndexRange(j_begin, nj)),
    func, x
  ) {}

  Dense::Dense(
    void (*func)(
      std::vector<double>& data,
      std::vector<std::vector<double>>& x,
      const int& ni, const int& nj,
      const int& i_begin, const int& j_begin
    ),
    std::vector<std::vector<double>>& x,
    const int ni, const int nj,
    const int i_begin, const int j_begin,
    const int i_abs, const int j_abs,
    const int level
  ) : Node(i_abs,j_abs,level), dim{ni, nj}, stride(nj) {
    data.resize(dim[0]*dim[1]);
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

  Dense Dense::operator+(const Dense& A) const {
    Dense B(*this);
    B += A;
    return B;
  }

  Dense Dense::operator-(const Dense& A) const {
    timing::start("Dense - Dense");
    Dense B(*this);
    B -= A;
    timing::stop("Dense - Dense");
    return B;
  }

  const Dense& Dense::operator+=(const Dense& A) {
    assert(dim[0] == A.dim[0]);
    assert(dim[1] == A.dim[1]);
    timing::start("Dense -= Dense");
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        (*this)(i, j) += A(i, j);
      }
    }
    timing::stop("Dense -= Dense");
    return *this;
  }

  const Dense& Dense::operator-=(const Dense& A) {
    assert(dim[0] == A.dim[0]);
    assert(dim[1] == A.dim[1]);
    timing::start("Dense += Dense");
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        (*this)(i, j) -= A(i, j);
      }
    }
    timing::stop("Dense += Dense");
    return *this;
  }

  const Dense& Dense::operator*=(const double a) {
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        (*this)(i, j) *= a;
      }
    }
    return *this;
  }

  double* Dense::get_pointer() {
    return &data[0];
  }

  const double* Dense::get_pointer() const {
    return &data[0];
  }

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

  double* Dense::operator&() {
    return get_pointer();
  }

  const double* Dense::operator&() const {
    return get_pointer();
  }

  int Dense::size() const {
    return dim[0] * dim[1];
  }

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
    row_range.length = dim0;
    col_range.length = dim1;
    dim[0] = dim0;
    dim[1] = dim1;
    stride = dim[1];
    data.resize(dim[0]*dim[1]);
    timing::stop("Dense resize");
  }

  Dense Dense::transpose() const {
    Dense A(dim[1], dim[0], i_abs, j_abs, level);
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
    std::swap(row_range, col_range);
    for(int i=0; i<dim[0]; i++) {
      for(int j=0; j<dim[1]; j++) {
        (*this)(i, j) = Copy(j, i);
      }
    }
  }

  Dense Dense::get_part(const Node& node) const {
    assert(is_child(node));
    Dense A(node, true);
    A.data.resize(A.dim[0]*A.dim[1]);
    int rel_row_begin = A.row_range.start - row_range.start;
    int rel_col_begin = A.col_range.start - col_range.start;
    for (int i=0; i<A.dim[0]; i++) {
      for (int j=0; j<A.dim[1]; j++) {
        A(i, j) = (*this)(i+rel_row_begin, j+rel_col_begin);
      }
    }
    return A;
  }

} // namespace hicma
