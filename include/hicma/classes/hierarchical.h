#ifndef hicma_classes_hierarchical_h
#define hicma_classes_hierarchical_h

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class Dense;
class ClusterTree;
class MatrixInitializer;

class Hierarchical : public Node {
 public:
  std::array<int64_t, 2> dim = {0, 0};
 private:
  std::vector<NodeProxy> data;

 public:
  // Special member functions
  Hierarchical() = default;

  virtual ~Hierarchical() = default;

  Hierarchical(const Hierarchical& A) = default;

  Hierarchical& operator=(const Hierarchical& A) = default;

  Hierarchical(Hierarchical&& A) = default;

  Hierarchical& operator=(Hierarchical&& A) = default;

  // Overridden functions from Node
  virtual const char* type() const override;

  // Conversion constructors
  Hierarchical(NodeProxy&&);

  Hierarchical(const Node& A, int64_t n_row_blocks, int64_t n_col_blocks);

  // Additional constructors
  Hierarchical(int64_t n_row_blocks, int64_t n_col_blocks=1);

  Hierarchical(
    const ClusterTree& node,
    const MatrixInitializer& initer,
    int64_t rank,
    int64_t admis
  );

  Hierarchical(
    void (*func)(
      Dense& A, const std::vector<std::vector<double>>& x,
      int64_t row_start, int64_t col_start
    ),
    const std::vector<std::vector<double>>& x,
    int64_t n_rows, int64_t n_cols,
    int64_t rank,
    int64_t nleaf,
    int64_t admis=1,
    int64_t n_row_blocks=2, int64_t n_col_blocks=2,
    int64_t row_start=0, int64_t col_start=0
  );

  // Additional operators
  const NodeProxy& operator[](int64_t i) const;

  NodeProxy& operator[](int64_t i);

  const NodeProxy& operator[](const ClusterTree& node) const;

  NodeProxy& operator[](const ClusterTree& node);

  const NodeProxy& operator()(int64_t i, int64_t j) const;

  NodeProxy& operator()(int64_t i, int64_t j);

  // Utility methods
  void blr_col_qr(Hierarchical& Q, Hierarchical& R);

  void split_col(Hierarchical& QL);

  void restore_col(const Hierarchical& Sp, const Hierarchical& QL);

  void col_qr(int64_t j, Hierarchical& Q, Hierarchical &R);

  bool is_admissible(const ClusterTree& node, int64_t dist_to_diag);
};

} // namespace hicma

#endif // hicma_classes_hierarchical_h
