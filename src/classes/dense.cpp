#include "hicma/classes/dense.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/empty.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/classes/initialization_helpers/matrix_initializer_file.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <utility>


namespace hicma
{

// explicit template initialization (these are the only available types)
template class Dense<float>;
template class Dense<double>;
template void Dense<float>::copy_to(Dense<float>&, int64_t, int64_t) const;
template void Dense<float>::copy_to(Dense<double>&, int64_t, int64_t) const;
template void Dense<double>::copy_to(Dense<float>&, int64_t, int64_t) const;
template void Dense<double>::copy_to(Dense<double>&, int64_t, int64_t) const;


//TODO no implicit type conversion
template<typename T>
Dense<T>::Dense(const Dense<T>& A)
: Matrix(A), dim{A.dim[0], A.dim[1]}, stride(A.dim[1]), rel_start{0, 0}
{
  timing::start("Dense cctor");
  (*data).resize(dim[0]*dim[1], 0);
  data_ptr = &(*data)[0];
  fill_dense_from(A, *this);
  timing::stop("Dense cctor");
}

// TODO extend for type conversion?
template<typename T>
Dense<T>& Dense<T>::operator=(const Dense<T>& A) {
  timing::start("Dense copy assignment");
  Matrix::operator=(A);
  dim = A.dim;
  stride = A.stride;
  (*data).resize(dim[0]*dim[1], 0);
  rel_start = {0, 0};
  data_ptr = &(*data)[0];
  fill_dense_from(A, *this);
  timing::stop("Dense copy assignment");
  return *this;
}

// Also enables switching between template types
template<typename T>
Dense<T>::Dense(const Matrix& A)
: Matrix(A), dim{get_n_rows(A), get_n_cols(A)}, stride(dim[1]),
  data(std::make_shared<std::vector<T>>(dim[0]*dim[1], 0)),
  rel_start{0, 0}, data_ptr(&(*data)[0])
{
  fill_dense_from(A, *this);
}

template<typename T>
void fill_dense_from_hierarchical(const Hierarchical<T>& A, Dense<T>& B) {
  timing::start("make_dense(H)");
  Hierarchical<T> BH = split(B, A);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      fill_dense_from(A(i, j), BH(i, j));
    }
  }
  timing::stop("make_dense(H)"); 
}

define_method(void, fill_dense_from, (const Hierarchical<double>& A, Dense<double>& B, int64_t, int64_t)) {
  fill_dense_from_hierarchical(A, B);
}
define_method(void, fill_dense_from, (const Hierarchical<float>& A, Dense<float>& B, int64_t, int64_t)) {
  fill_dense_from_hierarchical(A, B);
}

template<typename T>
void fill_dense_from_low_rank(const LowRank<T>& A, Dense<T>& B) {
  timing::start("make_dense(LR)");
  gemm(gemm(A.U, A.S), A.V, B, 1, 0);
  timing::stop("make_dense(LR)");
}

define_method(void, fill_dense_from, (const LowRank<double>& A, Dense<double>& B, int64_t, int64_t)) {
  fill_dense_from_low_rank(A, B);
}
define_method(void, fill_dense_from, (const LowRank<float>& A, Dense<float>& B, int64_t, int64_t)) {
  fill_dense_from_low_rank(A, B);
}

define_method(void, fill_dense_from, (const Dense<double>& A, Dense<double>& B, int64_t row_start, int64_t col_start)) {
  A.copy_to(B, row_start, col_start);
}
define_method(void, fill_dense_from, (const Dense<float>& A, Dense<float>& B, int64_t row_start, int64_t col_start)) {
  A.copy_to(B, row_start, col_start);
}
define_method(void, fill_dense_from, (const Dense<double>& A, Dense<float>& B, int64_t row_start, int64_t col_start)) {
  A.copy_to(B, row_start, col_start);
}
define_method(void, fill_dense_from, (const Dense<float>& A, Dense<double>& B, int64_t row_start, int64_t col_start)) {
  A.copy_to(B, row_start, col_start);
}

template<typename T>
void fill_dense_from_empty(const Empty&, Dense<T>& B) {
  assert(A.dim[0] == B.dim[0]);
  assert(A.dim[1] == B.dim[1]);
  B = 0.0;
}

define_method(void, fill_dense_from, (const Empty& A, Dense<double>& B, int64_t, int64_t)) {
  fill_dense_from_empty(A, B);
}
define_method(void, fill_dense_from, (const Empty& A, Dense<float>& B, int64_t, int64_t)) {
  fill_dense_from_empty(A, B);
}

define_method(void, fill_dense_from, (const Matrix& A, Matrix& B, int64_t, int64_t)) {
  omm_error_handler("fill_dense_from", {A, B}, __FILE__, __LINE__);
  std::abort();
}

declare_method(Matrix&&, move_from_dense, (virtual_<Matrix&>))

// static_cast is not safe because we don't know the template of MatrixProxy
// TODO add some type of error handling
template<typename T>
Dense<T>::Dense(MatrixProxy&& A)
: Dense<T>(dynamic_cast<Dense<T>&&>(move_from_dense(A))) {}

define_method(Matrix&&, move_from_dense, (Dense<double>& A)) {
  return std::move(A);
}

define_method(Matrix&&, move_from_dense, (Dense<float>& A)) {
  return std::move(A);
}

define_method(Matrix&&, move_from_dense, (Matrix& A)) {
  omm_error_handler("move_from_dense", {A}, __FILE__, __LINE__);
  std::abort();
}

template<typename T>
Dense<T>::Dense(int64_t n_rows, int64_t n_cols)
: dim{n_rows, n_cols}, stride(dim[1]) {
  timing::start("Dense alloc");
  (*data).resize(dim[0]*dim[1], 0);
  rel_start = {0, 0};
  data_ptr = &(*data)[0];
  timing::stop("Dense alloc");
}

template<typename T>
Dense<T>::Dense(
  void (*func)(
    T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const vec2d<double>& params,
    int64_t row_start, int64_t col_start
  ),
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start
) : Dense(n_rows, n_cols) {
    vec2d<double> params;
    func(
      &(*this), dim[0], dim[1], stride, params, row_start, col_start
    );
}

template<typename T>
Dense<T>::Dense(
  void (*func)(
    T* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
    const vec2d<double>& params,
    int64_t row_start, int64_t col_start
  ),
  const vec2d<double>& params,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start
) : Dense(n_rows, n_cols) {
    func(
      &(*this), dim[0], dim[1], stride, params, row_start, col_start
    );
}

template<typename T>
Dense<T>::Dense(
  std::string filename, MatrixLayout ordering,
  int64_t n_rows, int64_t n_cols,
  int64_t row_start, int64_t col_start
) : Dense(n_rows, n_cols) {
  MatrixInitializerFile initializer(filename, ordering);
  initializer.fill_dense_representation(*this,
					{row_start, n_rows},
					{col_start, n_cols});
}

template<typename T> template<typename U>
void Dense<T>::copy_to(Dense<U> &A, int64_t row_start, int64_t col_start) const {
  assert(dim[0]-row_start >= A.dim[0]);
  assert(dim[1]-col_start >= A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; i++) {
    for (int64_t j=0; j<A.dim[1]; j++) {
      // relies on implicit conversion
      A(i, j) = (*this)(row_start+i, col_start+j);
    }
  }
}

// relies on automatic type conversion
template<typename T>
Dense<T>& Dense<T>::operator=(const T a) {
  for (int64_t i=0; i<dim[0]; i++) {
    for (int64_t j=0; j<dim[1]; j++) {
      (*this)(i, j) = a;
    }
  }
  return *this;
}

template<typename T>
T& Dense<T>::operator[](int64_t i) {
  assert(dim[0] == 1 || dim[1] == 1);
  if (dim[0] == 1) {
    assert(i < dim[1]);
    return data_ptr[i];
  } else {
    assert(i < dim[0]);
    return data_ptr[i*stride];
  }
}

template<typename T>
const T& Dense<T>::operator[](int64_t i) const {
  assert(dim[0] == 1 || dim[1] == 1);
  if (dim[0] == 1) {
    assert(i < dim[1]);
    return data_ptr[i];
  } else {
    assert(i < dim[0]);
    return data_ptr[i*stride];
  }
}

template<typename T>
T& Dense<T>::operator()(int64_t i, int64_t j) {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return data_ptr[i*stride+j];
}

template<typename T>
const T& Dense<T>::operator()(int64_t i, int64_t j) const {
  assert(i < dim[0]);
  assert(j < dim[1]);
  return data_ptr[i*stride+j];
}

template<typename T>
T* Dense<T>::operator&() { return data_ptr; }

template<typename T>
const T* Dense<T>::operator&() const { return data_ptr; }

template<typename T>
Dense<T> Dense<T>::shallow_copy() const {
  Dense out;
  out.dim = dim;
  out.stride = stride;
  out.data = data;
  out.rel_start = rel_start;
  out.data_ptr = data_ptr;
  return out;
}

template<typename T>
bool Dense<T>::is_submatrix() const {
  bool out = (rel_start == std::array<int64_t, 2>{0, 0});
  // TODO Think about int64_t!
  out &= (data->size() == uint64_t(dim[0] * dim[1]));
  return !out;
}

template<typename T>
std::vector<Dense<T>> Dense<T>::split(
  const std::vector<IndexRange>& row_ranges,
  const std::vector<IndexRange>& col_ranges,
  bool copy
) const {
  std::vector<Dense<T>> out(row_ranges.size()*col_ranges.size());
  if (copy) {
    for (uint64_t i=0; i<row_ranges.size(); ++i) {
      for (uint64_t j=0; j<col_ranges.size(); ++j) {
        Dense<T> child(row_ranges[i].n, col_ranges[j].n);
        (*this).copy_to(child, row_ranges[i].start, col_ranges[j].start);
        out[i*col_ranges.size()+j] = std::move(child);
      }
    }
  } else {
    for (uint64_t i=0; i<row_ranges.size(); ++i) {
      for (uint64_t j=0; j<col_ranges.size(); ++j) {
        Dense<T> child;
        child.dim = {row_ranges[i].n, col_ranges[j].n};
        child.stride = stride;
        child.data = data;
        child.rel_start[0] = rel_start[0] + row_ranges[i].start;
        child.rel_start[1] = rel_start[1] + col_ranges[j].start;
        child.data_ptr = &(*child.data)[
          child.rel_start[0]*child.stride + child.rel_start[1]
        ];
        out[i*col_ranges.size()+j] = std::move(child);
      }
    }
  }
  return out;
}

template<typename T>
std::vector<Dense<T>> Dense<T>::split(
  uint64_t n_row_splits, uint64_t n_col_splits, bool copy
) const {
  IndexRange row_index(0, dim[0]), col_index(0, dim[1]);
  return split(
    row_index.split(n_row_splits), col_index.split(n_col_splits), copy
  );
}

//TODO move somewhere else?
void fill_dense_from(const Matrix& A, Matrix &B){
  fill_dense_from(A, B, 0, 0);
}

//TODO remove
/* YOMM EXAMPLES */
/*
template Dense<double> add_one(const Matrix &A);
template Dense<double>& add_one2(Matrix &A);
template Dense<float> add_one(const Matrix &A);
template Dense<float>& add_one2(Matrix &A);

declare_method(
  MatrixProxy, add_one_omm,
  (virtual_<const Matrix&>)
)

template<typename T>
Dense<T> add_one(const Matrix &A){
  return add_one_omm(A);
}

template<typename T>
MatrixProxy add1 (const Dense<T>& A){
  Dense<T> result(A);
  for (int64_t i=0; i<A.dim[0]; ++i)
    for (int64_t j=0; j<A.dim[0]; ++j)
      result(i,j) += 1;
  return result;
}

define_method(MatrixProxy, add_one_omm, (const Dense<double>& A)) {
  return add1(A);
}
define_method(MatrixProxy, add_one_omm, (const Dense<float>& A)) {
  return add1(A);
}
define_method(MatrixProxy, add_one_omm, (const LowRank<double>& A)) {
  return add1(Dense<double>(A));
}
define_method(MatrixProxy, add_one_omm, (const Matrix& A)) {
  std::abort();
}

declare_method(
  Matrix&, add_one_omm2,
  (virtual_<Matrix&>)
)

template<typename T>
Dense<T>& add_one2(Matrix& A){
  return dynamic_cast<Dense<T>&>(add_one_omm2(A));
}

template<typename T>
Matrix& add2 (Dense<T>& A){
  for (int64_t i=0; i<A.dim[0]; ++i)
    for (int64_t j=0; j<A.dim[0]; ++j)
      A(i,j) += 1;
  return A;
}

define_method(Matrix&, add_one_omm2, (Dense<double>& A)) {
  return add2(A);
}
define_method(Matrix&, add_one_omm2, (Dense<float>& A)) {
  return add2(A);
}
define_method(Matrix&, add_one_omm2, (Matrix& A)) {
  std::abort();
}*/

/*
auto deduce_type (Matrix& A) ->decltype(dynamic_cast<Dense<double>*>(&A)?dynamic_cast<Dense<double>*>(&A):(dynamic_cast<Dense<float>*>(&A)))
) {
  if (auto d_type = dynamic_cast<Dense<double>*>(&A))
    return d_type;
  else if (auto d_type = dynamic_cast<Dense<float>*>(&A))
    return d_type;
  else if (auto d_type = dynamic_cast<LowRank<double>*>(&A))
    return d_type;
  else if (auto d_type = dynamic_cast<Hierarchical<double>*>(&A))
    return d_type;
  else
    std::abort();
}*/

} // namespace hicma
