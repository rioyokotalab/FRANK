#include "hicma/classes/dense_view.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/index_range.h"

#include <cassert>
#include <memory>
#include <utility>


namespace hicma
{

std::unique_ptr<Node> DenseView::clone() const {
  return std::make_unique<DenseView>(*this);
}

std::unique_ptr<Node> DenseView::move_clone() {
  return std::make_unique<DenseView>(std::move(*this));
}

const char* DenseView::type() const { return "DenseView"; }

DenseView::DenseView(Dense& A)
: DenseView(IndexRange(0, A.dim[0]), IndexRange(0, A.dim[1]), A) {}

DenseView::DenseView(const Dense& A)
: DenseView(IndexRange(0, A.dim[0]), IndexRange(0, A.dim[1]), A) {}

DenseView::DenseView(
  const IndexRange& row_range, const IndexRange& col_range, Dense& A
) {
  assert(row_range.start+row_range.length <= A.dim[0]);
  assert(col_range.start+col_range.length <= A.dim[1]);
  dim[0] = row_range.length;
  dim[1] = col_range.length;
  stride = A.stride;
  data = &A(row_range.start, col_range.start);
  const_data = &A(row_range.start, col_range.start);
}

DenseView::DenseView(
  const IndexRange& row_range, const IndexRange& col_range, const Dense& A
) {
  assert(row_range.start+row_range.length <= A.dim[0]);
  assert(col_range.start+col_range.length <= A.dim[1]);
  dim[0] = row_range.length;
  dim[1] = col_range.length;
  stride = A.stride;
  data = nullptr;
  const_data = &A(row_range.start, col_range.start);
}

double* DenseView::get_pointer() {
  assert(data != nullptr);
  return data;
}

const double* DenseView::get_pointer() const {
  assert(data != nullptr || const_data != nullptr);
  return data!=nullptr ? data : const_data;
}

} // namespace hicma
