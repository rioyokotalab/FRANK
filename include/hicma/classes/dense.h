#ifndef hicma_classes_dense_h
#define hicma_classes_dense_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"

#include <vector>
#include <memory>


namespace hicma
{

class IndexRange;

class Dense : public Node {
 private:
  std::vector<double> data;
 protected:
  virtual double* get_pointer();

  virtual const double* get_pointer() const;
 public:
  int dim[2] = {0, 0};
  int stride = 0;

  // Special member functions
  Dense() = default;

  virtual ~Dense() = default;

  Dense(const Dense& A);

  Dense& operator=(const Dense& A) = default;

  Dense(Dense&& A) = default;

  Dense& operator=(Dense&& A) = default;

  // Overridden functions from Node
  virtual std::unique_ptr<Node> clone() const override;

  virtual std::unique_ptr<Node> move_clone() override;

  virtual const char* type() const override;

  // Explicit conversions using multiple-dispatch function.
  explicit Dense(const Node& A);

  // Additional constructors
  Dense(int m, int n=1);

  Dense(
    const IndexRange& row_range,
    const IndexRange& col_range,
    void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
    std::vector<double>& x,
    int i_begin, int j_begin
  );

  Dense(
    void (*func)(Dense& A, std::vector<double>& x, int i_begin, int j_begin),
    std::vector<double>& x,
    int ni, int nj=1,
    int i_begin=0, int j_begin=0
  );

  Dense(
    void (*func)(
      std::vector<double>& data,
      std::vector<std::vector<double>>& x,
      const int& ni, const int& nj,
      const int& i_begin, const int& j_begin
    ),
    std::vector<std::vector<double>>& x,
    const int ni, const int nj,
    const int i_begin=0, const int j_begin=0
  );

  // Additional operators
  const Dense& operator=(const double a);

  double& operator[](int i);

  const double& operator[](int i) const;

  double& operator()(int i, int j);

  const double& operator()(int i, int j) const;

  double* operator&();

  const double* operator&() const;

  // Utility methods
  int size() const;

  void resize(int dim0, int dim1);

  Dense transpose() const;

  void transpose();

  // Get part of other Dense
  Dense get_part(
    const IndexRange& row_range, const IndexRange& col_range) const;
};

register_class(Dense, Node)

} // namespace hicma

#endif // hicma_classes_dense_h
