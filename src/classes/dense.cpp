#include "hicma/classes/dense.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>


namespace hicma
{

Dense::Dense(const Dense& A)
: Matrix(A), dim{A.dim[0], A.dim[1]}, stride(A.dim[1]), rel_start{0, 0}
{
  timing::start("Dense cctor");
  (*data).resize(dim[0]*dim[1], 0);
  data_ptr = &(*data)[0];
  for (int64_t i=0; i<dim[0]; i++) {
    for (int64_t j=0; j<dim[1]; j++) {
      (*this)(i, j) = A(i, j);
    }
  }
  timing::stop("Dense cctor");
}

Dense& Dense::operator=(const Dense& A) {
  timing::start("Dense copy assignment");
  Matrix::operator=(A);
  dim = A.dim;
  stride = A.stride;
  (*data).resize(dim[0]*dim[1], 0);
  rel_start = {0, 0};
  data_ptr = &(*data)[0];
  for (int64_t i=0; i<dim[0]; i++) {
    for (int64_t j=0; j<dim[1]; j++) {
      (*this)(i, j) = A(i, j);
    }
  }
  timing::stop("Dense copy assignment");
  return *this;
}

Dense::Dense(const Matrix& A)
: Matrix(A), dim{get_n_rows(A), get_n_cols(A)}, stride(dim[1]),
  data(std::make_shared<std::vector<double>>(dim[0]*dim[1], 0)),
  rel_start{0, 0}, data_ptr(&(*data)[0])
{
  fill_dense_from(A, *this);
}

define_method(void, fill_dense_from, (const Hierarchical& A, Dense& B)) {
  timing::start("make_dense(H)");
  NoCopySplit BH(B, A);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      fill_dense_from(A(i, j), BH(i, j));
    }
  }
  timing::stop("make_dense(H)");
}

define_method(void, fill_dense_from, (const LowRank& A, Dense& B)) {
  timing::start("make_dense(LR)");
  gemm(gemm(A.U, A.S), A.V, B, 1, 0);
  timing::stop("make_dense(LR)");
}

define_method(void, fill_dense_from, (const Dense& A, Dense& B)) {
  assert(A.dim[0] == B.dim[0]);
  assert(A.dim[1] == B.dim[1]);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      B(i, j) = A(i, j);
    }
  }
}

define_method(void, fill_dense_from, (const Matrix& A, Matrix& B)) {
  omm_error_handler("fill_dense_from", {A, B}, __FILE__, __LINE__);
  std::abort();
}

Dense::Dense(int64_t n_rows, int64_t n_cols)
: dim{n_rows, n_cols}, stride(dim[1]),
  data(std::make_shared<std::vector<double>>(dim[0]*dim[1])),
  rel_start{0, 0}, data_ptr(&(*data)[0])
{
  timing::start("Dense alloc");
  (*data).resize(dim[0]*dim[1], 0);
  timing::stop("Dense alloc");
}

Dense::Dense(
  void (*func)(
    Dense& A,
    const std::vector<std::vector<double>>& x,
    int64_t row_start, int64_t col_start
  ),
  const std::vector<std::vector<double>>& x,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start
) : Dense(n_rows, n_cols) {
  func(*this, x, row_start, col_start);
}

Dense::Dense(
  const Dense& A,
  int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
  bool copy
) : dim{n_rows, n_cols} {
  if (copy) {
    stride = dim[1];
    data = std::make_shared<std::vector<double>>(n_rows*n_cols);
    rel_start = {0, 0};
    data_ptr = &(*data)[0];
    for (int64_t i=0; i<dim[0]; i++) {
      for (int64_t j=0; j<dim[1]; j++) {
        (*this)(i, j) = A(row_start+i, col_start+j);
      }
    }
  } else {
    assert(row_start+dim[0] <= A.dim[0]);
    assert(col_start+dim[1] <= A.dim[1]);
    stride = A.stride;
    data = A.data;
    rel_start = {A.rel_start[0]+row_start, A.rel_start[1]+col_start};
    data_ptr = &(*data)[rel_start[0]*stride+rel_start[1]];
  }
}

const Dense& Dense::operator=(const double a) {
  for (int64_t i=0; i<dim[0]; i++) {
    for (int64_t j=0; j<dim[1]; j++) {
      (*this)(i, j) = a;
    }
  }
  return *this;
}

double& Dense::operator[](int64_t i) {
  assert(dim[0] == 1 || dim[1] == 1);
  if (dim[0] == 1) {
    assert(i < dim[1]);
    return data_ptr[i];
  } else {
    assert(i < dim[0]);
    return data_ptr[i*stride];
  }
}

const double& Dense::operator[](int64_t i) const {
  assert(dim[0] == 1 || dim[1] == 1);
  if (dim[0] == 1) {
    assert(i < dim[1]);
    return data_ptr[i];
  } else {
    assert(i < dim[0]);
    return data_ptr[i*stride];
  }
}

double& Dense::operator()(int64_t i, int64_t j) {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return data_ptr[i*stride+j];
}

const double& Dense::operator()(int64_t i, int64_t j) const {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return data_ptr[i*stride+j];
}

double* Dense::operator&() { return data_ptr; }

const double* Dense::operator&() const { return data_ptr; }

int64_t Dense::size() const { return dim[0] * dim[1]; }

void Dense::resize(int64_t dim0, int64_t dim1) {
  assert(data.use_count() == 1);
  assert(dim0 <= dim[0]);
  assert(dim1 <= dim[1]);
  timing::start("Dense resize");
  if (dim0 == dim[0] && dim1 == dim[1]) {
    timing::stop("Dense resize");
    return;
  }
  for (int64_t i=0; i<dim0; i++) {
    for (int64_t j=0; j<dim1; j++) {
      // TODO this is the only place where data is used directly now. Would be
      // better not to use it and somehow use the regular index operator
      // efficiently.
      // TODO This might/will cause issues when rel_start != {0, 0}
      (*data)[i*dim1+j] = (*this)(i, j);
    }
  }
  dim = {dim0, dim1};
  stride = dim[1];
  (*data).resize(dim[0]*dim[1]);
  timing::stop("Dense resize");
}

Dense Dense::transpose() const {
  Dense A(dim[1], dim[0]);
  for (int64_t i=0; i<dim[0]; i++) {
    for (int64_t j=0; j<dim[1]; j++) {
      A(j,i) = (*this)(i,j);
    }
  }
  return A;
}

void Dense::transpose() {
  // TODO Consider removing this function!
  assert(data.use_count() == 1);
  assert(stride == dim[1]);
  Dense Copy(*this);
  std::swap(dim[0], dim[1]);
  stride = dim[1];
  for(int64_t i=0; i<dim[0]; i++) {
    for(int64_t j=0; j<dim[1]; j++) {
      (*this)(i, j) = Copy(j, i);
    }
  }
}

} // namespace hicma
