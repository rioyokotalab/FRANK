#include "hicma/classes/dense_view.h"

#include "hicma/classes/dense.h"

#include <cassert>
#include <memory>
#include <utility>


namespace hicma {

  DenseView::DenseView() = default;

  DenseView::~DenseView() = default;

  DenseView::DenseView(const DenseView& A) = default;

  DenseView& DenseView::operator=(const DenseView& A) = default;

  DenseView::DenseView(DenseView&& A) = default;

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

  DenseView& DenseView::operator=(Dense& A) {
    *this = DenseView(A, A);
    return *this;
  }

  DenseView& DenseView::operator=(const Dense& A) {
    *this = DenseView(A, A);
    return *this;
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
