#ifndef hicma_classes_dense_view_h
#define hicma_classes_dense_view_h

#include "hicma/classes/dense.h"

#include "yorel/yomm2/cute.hpp"

#include <memory>

namespace hicma {

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
    DenseView();

    ~DenseView();

    DenseView(const DenseView& A);

    DenseView& operator=(const DenseView& A);

    DenseView(DenseView&& A);

    DenseView& operator=(DenseView&& A);

    // Overridden functions from Node
    std::unique_ptr<Node> clone() const override;

    std::unique_ptr<Node> move_clone() override;

    const char* type() const override;

    // Additional constructors
    DenseView(const Node& node, Dense& A);

    DenseView(const Node& node, const Dense& A);

    // Additional operators
    DenseView& operator=(const Dense& A);

    // Delete methods that cannot be used from Dense
    void tranpose() = delete;

    void resize(int dim0, int dim1) = delete;
  };

  register_class(DenseView, Dense);

} // namespace hicma

#endif // hicma_classes_dense_view_h
