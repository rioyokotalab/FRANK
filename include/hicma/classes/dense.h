#ifndef hicma_classes_dense_h
#define hicma_classes_dense_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>


namespace hicma
{

// TODO Remove awareness of ClusterTree?
class ClusterTree;
class NodeProxy;

class Dense : public Node {
 private:
  std::vector<double> data;
  double* data_ptr = nullptr;
  const double* const_data_ptr = nullptr;
  bool owning = true;
 protected:
  virtual double* get_pointer();

  virtual const double* get_pointer() const;
 public:
  std::array<int64_t, 2> dim = {0, 0};
  int64_t stride = 0;

  // Special member functions
  Dense() = default;

  virtual ~Dense() = default;

  Dense(const Dense& A);

  Dense& operator=(const Dense& A);

  Dense(Dense&& A) = default;

  Dense& operator=(Dense&& A) = default;

  // Overridden functions from Node
  virtual std::unique_ptr<Node> clone() const override;

  virtual std::unique_ptr<Node> move_clone() override;

  virtual const char* type() const override;

  // Explicit conversions using multiple-dispatch function.
  explicit Dense(const Node& A);

  // Additional constructors
  Dense(int64_t m, int64_t n=1);

  Dense(
    const ClusterTree& node,
    void (*func)(
      Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin),
    std::vector<double>& x
  );

  Dense(
    void (*func)(
      Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin),
    std::vector<double>& x,
    int64_t ni, int64_t nj=1,
    int64_t i_begin=0, int64_t j_begin=0
  );

  Dense(
    void (*func)(
      std::vector<double>& data,
      std::vector<std::vector<double>>& x,
      int64_t ni, int64_t nj,
      int64_t i_begin, int64_t j_begin
    ),
    std::vector<std::vector<double>>& x,
    const int64_t ni, const int64_t nj,
    const int64_t i_begin=0, const int64_t j_begin=0
  );

  Dense(const ClusterTree& node, Dense& A);

  Dense(const ClusterTree& node, const Dense& A);

  // Additional operators
  const Dense& operator=(const double a);

  double& operator[](int64_t i);

  const double& operator[](int64_t i) const;

  double& operator[](std::array<int64_t, 2> pos);

  const double& operator[](std::array<int64_t, 2> pos) const;

  double& operator()(int64_t i, int64_t j);

  const double& operator()(int64_t i, int64_t j) const;

  double* operator&();

  const double* operator&() const;

  // Utility methods
  int64_t size() const;

  void resize(int64_t dim0, int64_t dim1);

  Dense transpose() const;

  void transpose();

  // Get part of other Dense
  Dense get_part(const ClusterTree& node) const;
};

register_class(Dense, Node)

} // namespace hicma

#endif // hicma_classes_dense_h
