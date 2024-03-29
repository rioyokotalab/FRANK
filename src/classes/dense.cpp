#include "FRANK/classes/dense.h"

#include "FRANK/classes/empty.h"
#include "FRANK/classes/hierarchical.h"
#include "FRANK/classes/low_rank.h"
#include "FRANK/classes/matrix.h"
#include "FRANK/classes/matrix_proxy.h"
#include "FRANK/classes/initialization_helpers/index_range.h"
#include "FRANK/classes/initialization_helpers/matrix_initializer_file.h"
#include "FRANK/operations/BLAS.h"
#include "FRANK/operations/misc.h"
#include "FRANK/util/omm_error_handler.h"
#include "FRANK/util/print.h"
#include "FRANK/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>


namespace FRANK
{

uint64_t next_unique_id = 0;

declare_method(
  void, fill_dense_from, (virtual_<const Matrix&>, virtual_<Matrix&>)
)

Dense::Dense(const Dense& A)
: Matrix(A), dim{A.dim[0], A.dim[1]}, stride(A.dim[1]), rel_start{0, 0},
  unique_id(next_unique_id++)
{
  data = std::make_shared<std::vector<double>>(dim[0]*dim[1], 0);
  data_ptr = (*data).data();
  fill_dense_from(A, *this);
}

Dense& Dense::operator=(const Dense& A) {
  Matrix::operator=(A);
  dim = A.dim;
  stride = A.stride;
  data = std::make_shared<std::vector<double>>(dim[0]*dim[1], 0);
  rel_start = {0, 0};
  data_ptr = (*data).data();
  fill_dense_from(A, *this);
  unique_id = next_unique_id++;
  return *this;
}

Dense::Dense(const Matrix& A)
: Matrix(A), dim{get_n_rows(A), get_n_cols(A)}, stride(dim[1]),
  data(std::make_shared<std::vector<double>>(dim[0]*dim[1], 0)),
  rel_start{0, 0}, data_ptr(&(*data)[0]), unique_id(next_unique_id++)
{
  fill_dense_from(A, *this);
}

define_method(void, fill_dense_from, (const Hierarchical& A, Dense& B)) {
  Hierarchical BH = split(B, A);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      fill_dense_from(A(i, j), BH(i, j));
    }
  }
}

define_method(void, fill_dense_from, (const LowRank& A, Dense& B)) {
  gemm(gemm(A.U, A.S), A.V, B, 1, 0);
}

define_method(void, fill_dense_from, (const Dense& A, Dense& B)) {
  assert(A.dim[0] == B.dim[0]);
  assert(A.dim[1] == B.dim[1]);
  A.copy_to(B);
}

define_method(void, fill_dense_from, ([[maybe_unused]] const Empty& A, Dense& B)) {
  assert(A.dim[0] == B.dim[0]);
  assert(A.dim[1] == B.dim[1]);
  B = 0.0;
}

define_method(void, fill_dense_from, (const Matrix& A, Matrix& B)) {
  omm_error_handler("fill_dense_from", {A, B}, __FILE__, __LINE__);
  std::abort();
}

declare_method(Dense&&, move_from_dense, (virtual_<Matrix&>))

Dense::Dense(MatrixProxy&& A)
: Dense(move_from_dense(A)) {}

define_method(Dense&&, move_from_dense, (Dense& A)) {
  return std::move(A);
}

define_method(Dense&&, move_from_dense, (Matrix& A)) {
  omm_error_handler("move_from_dense", {A}, __FILE__, __LINE__);
  std::abort();
}

Dense::Dense(const int64_t n_rows, const int64_t n_cols)
: dim{n_rows, n_cols}, stride(dim[1]), unique_id(next_unique_id++) {
  data = std::make_shared<std::vector<double>>(dim[0]*dim[1], 0);
  rel_start = {0, 0};
  data_ptr = (*data).data();
}

Dense::Dense(
  void (*kernel)(
    double* A, const uint64_t A_rows, const uint64_t A_cols, const uint64_t A_stride,
    const std::vector<std::vector<double>>& params,
    const int64_t row_start, const int64_t col_start
  ),
  const std::vector<std::vector<double>>& params,
  const int64_t n_rows, const int64_t n_cols,
  const int64_t row_start, const int64_t col_start
) : Dense(n_rows, n_cols) {
    kernel(
      &(*this), dim[0], dim[1], stride, params, row_start, col_start
    );
}

Dense::Dense(
  const std::string filename, const MatrixLayout ordering,
  const int64_t n_rows, const int64_t n_cols,
  const int64_t row_start, const int64_t col_start
) : Dense(n_rows, n_cols) {
  MatrixInitializerFile initializer(filename, ordering);
  initializer.fill_dense_representation(*this,
                                        {row_start, n_rows},
                                        {col_start, n_cols});
}

void Dense::copy_to(Dense &A, const int64_t row_start, const int64_t col_start) const {
  assert(dim[0]-row_start >= A.dim[0]);
  assert(dim[1]-col_start >= A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      A(i, j) = (*this)(row_start+i, col_start+j);
    }
  }
}

Dense& Dense::operator=(const double a) {
  for (int64_t i=0; i<dim[0]; i++) {
    for (int64_t j=0; j<dim[1]; j++) {
      (*this)(i, j) = a;
    }
  }
  return *this;
}

double& Dense::operator[](const int64_t i) {
  assert(dim[0] == 1 || dim[1] == 1);
  if (dim[0] == 1) {
    assert(i < dim[1]);
    return data_ptr[i];
  } else {
    assert(i < dim[0]);
    return data_ptr[i*stride];
  }
}

const double& Dense::operator[](const int64_t i) const {
  assert(dim[0] == 1 || dim[1] == 1);
  if (dim[0] == 1) {
    assert(i < dim[1]);
    return data_ptr[i];
  } else {
    assert(i < dim[0]);
    return data_ptr[i*stride];
  }
}

double& Dense::operator()(const int64_t i, const int64_t j) {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return data_ptr[i*stride+j];
}

const double& Dense::operator()(const int64_t i, const int64_t j) const {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return data_ptr[i*stride+j];
}

double* Dense::operator&() { return data_ptr; }

const double* Dense::operator&() const { return data_ptr; }

Dense Dense::shallow_copy() const {
  Dense out;
  out.dim = dim;
  out.stride = stride;
  out.data = data;
  out.rel_start = rel_start;
  out.data_ptr = data_ptr;
  out.unique_id = unique_id;
  return out;
}

bool Dense::is_submatrix() const {
  bool out = (rel_start == std::array<int64_t, 2>{0, 0});
  // TODO Think about int64_t!
  out &= (data->size() == uint64_t(dim[0] * dim[1]));
  return !out;
}

uint64_t Dense::id() const { return unique_id; }

std::vector<Dense> Dense::split(
  const std::vector<IndexRange>& row_ranges,
  const std::vector<IndexRange>& col_ranges,
  const bool copy
) const {
  std::vector<Dense> out(row_ranges.size()*col_ranges.size());
  if (copy) {
    for (uint64_t i=0; i<row_ranges.size(); ++i) {
      for (uint64_t j=0; j<col_ranges.size(); ++j) {
        Dense child(row_ranges[i].n, col_ranges[j].n);
        (*this).copy_to(child, row_ranges[i].start, col_ranges[j].start);
        out[i*col_ranges.size()+j] = std::move(child);
      }
    }
  } else {
    for (uint64_t i=0; i<row_ranges.size(); ++i) {
      for (uint64_t j=0; j<col_ranges.size(); ++j) {
        Dense child;
        child.dim = {row_ranges[i].n, col_ranges[j].n};
        child.stride = stride;
        child.data = data;
        child.rel_start[0] = rel_start[0] + row_ranges[i].start;
        child.rel_start[1] = rel_start[1] + col_ranges[j].start;
        child.data_ptr = (*child.data).data() +
          child.rel_start[0]*child.stride + child.rel_start[1];
        child.unique_id = next_unique_id++;
        out[i*col_ranges.size()+j] = std::move(child);
      }
    }
  }
  return out;
}

std::vector<Dense> Dense::split(
  const uint64_t n_row_splits,
  const uint64_t n_col_splits,
  const bool copy
) const {
  IndexRange row_index(0, dim[0]), col_index(0, dim[1]);
  return split(
    row_index.split(n_row_splits), col_index.split(n_col_splits), copy
  );
}

} // namespace FRANK
