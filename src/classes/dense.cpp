#include "hicma/classes/dense.h"

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/operations/BLAS/gemm.h"
#include "hicma/operations/misc/get_dim.h"
#include "hicma/gpu_batch/batch.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma {

  Dense::Dense() : dim{0, 0}, stride(dim[1]) { MM_INIT(); }

  Dense::~Dense() = default;

  Dense::Dense(const Dense& A) {
    MM_INIT();
    *this = A;
  }

  Dense& Dense::operator=(const Dense& A) = default;

  Dense::Dense(Dense&& A) {
    MM_INIT();
    *this = std::move(A);
  }

  Dense& Dense::operator=(Dense&& A) = default;

  Dense::Dense(int m) : dim{m, 1}, stride(dim[1]) {
    MM_INIT();
    data.resize(dim[0], 0);
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
    MM_INIT();
    if (!only_node) {
      *this = make_dense(A);
    }
  }

  MULTI_METHOD(move_from_dense, Dense, virtual_<Node>&);

  BEGIN_SPECIALIZATION(
    move_from_dense, Dense,
    Dense& A
  ) {
    return std::move(A);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(
    move_from_dense, Dense,
    Node& A
  ) {
    std::cout << "Cannot move to Dense from " << A.type() << "!" << std::endl;
    abort();
  } END_SPECIALIZATION;

  Dense::Dense(NodeProxy&& A) {
    *this = move_from_dense(A);
  }

  Dense::Dense(
    int m, int n,
    int i_abs, int j_abs,
    int level
  ) : Dense(
    Node(i_abs, j_abs, level, IndexRange(0, m), IndexRange(0, n)),
    true
  ) {
    data.resize(dim[0]*dim[1], 0);
  }

  Dense::Dense(
    const Node& node,
    void (*func)(Dense& A, std::vector<double>& x),
    std::vector<double>& x
  ) : Node(node), dim{node.row_range.length, node.col_range.length}, stride(dim[1]) {
    MM_INIT();
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
    MM_INIT();
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
    Dense B(*this);
    B -= A;
    return B;
  }

  const Dense& Dense::operator+=(const Dense& A) {
    assert(dim[0] == A.dim[0]);
    assert(dim[1] == A.dim[1]);
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        (*this)(i, j) += A(i, j);
      }
    }
    return *this;
  }

  const Dense& Dense::operator-=(const Dense& A) {
    assert(dim[0] == A.dim[0]);
    assert(dim[1] == A.dim[1]);
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        (*this)(i, j) -= A(i, j);
      }
    }
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

  double& Dense::operator[](int i) {
    assert(dim[0] == 1 || dim[1] == 1);
    if (dim[0] == 1) {
      assert(i < dim[1]);
      return data[i];
    } else {
      assert(i < dim[0]);
      return data[i*stride];
    }
  }

  const double& Dense::operator[](int i) const {
    assert(dim[0] == 1 || dim[1] == 1);
    if (dim[0] == 1) {
      assert(i < dim[1]);
      return data[i];
    } else {
      assert(i < dim[0]);
      return data[i*stride];
    }
  }

  double& Dense::operator()(int i, int j) {
    assert(i < dim[0]);
    assert(j < dim[1]);
    return data[i*stride+j];
  }

  const double& Dense::operator()(int i, int j) const {
    assert(i < dim[0]);
    assert(j < dim[1]);
    return data[i*stride+j];
  }

  double* Dense::operator&() {
    return &data[0];
  }

  const double* Dense::operator&() const {
    return &data[0];
  }

  int Dense::size() const {
    return dim[0] * dim[1];
  }

  void Dense::resize(int dim0, int dim1) {
    assert(dim0 <= dim[0]);
    assert(dim1 <= dim[1]);
    if (dim0 == dim[0] && dim1 == dim[1]) return;
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

  BEGIN_SPECIALIZATION(make_dense, Dense, const Hierarchical& A){
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
    // TODO Consider return with std::move. Test if the copy is elided!!
    return B;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const LowRank& A){
    Dense B(A.dim[0], A.dim[1]);
    Dense UxS(A.dim[0], A.rank);
    gemm(A.U, A.S, UxS, 1, 0);
    gemm(UxS, A.V, B, 1, 0);
    // TODO Consider return with std::move. Test if the copy is elided!!
    return B;
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const Dense& A){
    // TODO Consider return with std::move. Test if the copy is elided!!
    return Dense(A);
  } END_SPECIALIZATION;

  BEGIN_SPECIALIZATION(make_dense, Dense, const Node& A){
    std::cout << "Cannot create Dense from " << A.type() << "!" << std::endl;
    abort();
  } END_SPECIALIZATION;


  DenseView::DenseView() : Dense() { MM_INIT(); }

  DenseView::~DenseView() = default;

  DenseView::DenseView(const DenseView& A) {
    MM_INIT();
    *this = A;
  }
  DenseView& DenseView::operator=(const DenseView& A) = default;

  DenseView::DenseView(DenseView&& A) {
    MM_INIT();
    *this = std::move(A);
  }

  DenseView& DenseView::operator=(DenseView&& A) = default;

  std::unique_ptr<Node> DenseView::clone() const {
    return std::make_unique<DenseView>(*this);
  }

  std::unique_ptr<Node> DenseView::move_clone() {
    return std::make_unique<DenseView>(std::move(*this));
  }

  const char* DenseView::type() const {
    return "DenseView";
  }

  double& DenseView::operator[](int i) {
    assert(dim[0] == 1 || dim[1] == 1);
    assert(data != nullptr);
    if (dim[0] == 1) {
      assert(i < dim[1]);
      return data[i];
    } else {
      assert(i < dim[0]);
      return data[i*stride];
    }
  }

  const double& DenseView::operator[](int i) const {
    assert(dim[0] == 1 || dim[1] == 1);
    assert(data != nullptr || const_data != nullptr);
    if (dim[0] == 1) {
      assert(i < dim[1]);
      return data!=nullptr ? data[i] : const_data[i];
    } else {
      assert(i < dim[0]);
      return data!=nullptr ? data[i*stride] : const_data[i*stride];
    }
  }

  double& DenseView::operator()(int i, int j) {
    assert(i < dim[0]);
    assert(j < dim[1]);
    assert(data != nullptr);
    return data[i*stride+j];
  }

  const double& DenseView::operator()(int i, int j) const {
    assert(i < dim[0]);
    assert(j < dim[1]);
    assert(data != nullptr || const_data != nullptr);
    return data!=nullptr ? data[i*stride+j] : const_data[i*stride+j];
  }

  double* DenseView::operator&() {
    assert(data != nullptr);
    return data;
  }

  const double* DenseView::operator&() const {
    assert(data != nullptr || const_data != nullptr);
    return data!=nullptr ? data : const_data;
  }

  DenseView::DenseView(const Node& node, Dense& A)
  : Dense(node, true) {
    assert(A.is_child(node));
    stride = A.stride;
    int rel_row_begin = node.row_range.start - A.row_range.start;
    int rel_col_begin = node.col_range.start - A.col_range.start;
    data = &A(rel_row_begin, rel_col_begin);
    const_data = &A(rel_row_begin, rel_col_begin);
  }

  DenseView::DenseView(const Node& node, const Dense& A)
  : Dense(node, true) {
    assert(A.is_child(node));
    stride = A.stride;
    int rel_row_begin = node.row_range.start - A.row_range.start;
    int rel_col_begin = node.col_range.start - A.col_range.start;
    data = nullptr;
    const_data = &A(rel_row_begin, rel_col_begin);
  }

} // namespace hicma
