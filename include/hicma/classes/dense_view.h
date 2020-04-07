#ifndef hicma_classes_dense_view_h
#define hicma_classes_dense_view_h

#include "hicma/classes/dense.h"

#include "yorel/yomm2/cute.hpp"

#include <memory>


namespace hicma
{

class IndexRange;
class Node;

class DenseView : public Dense {
 private:
  double* data;
  const double* const_data;
 protected:
  virtual double* get_pointer() override;

  virtual const double* get_pointer() const override;
 public:
  // Special member functions
  DenseView() = default;

  ~DenseView() = default;

  DenseView(const DenseView& A) = default;

  DenseView& operator=(const DenseView& A) = default;

  DenseView(DenseView&& A) = default;

  DenseView& operator=(DenseView&& A) = default;

  // Overridden functions from Node
  std::unique_ptr<Node> clone() const override;

  std::unique_ptr<Node> move_clone() override;

  const char* type() const override;

  // Additional constructors
  DenseView(Dense& A);

  DenseView(const Dense& A);

  DenseView(
    const IndexRange& row_range, const IndexRange& col_range, Dense& A);

  DenseView(
    const IndexRange& row_range, const IndexRange& col_range, const Dense& A);

  // Delete methods that cannot be used from Dense
  void tranpose() = delete;

  void resize(int dim0, int dim1) = delete;
};

register_class(DenseView, Dense)

} // namespace hicma

#endif // hicma_classes_dense_view_h
