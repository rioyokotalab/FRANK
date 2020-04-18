#ifndef hicma_classes_hierarchical_h
#define hicma_classes_hierarchical_h

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"

#include "yorel/yomm2/cute.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>


namespace hicma
{

class Dense;
class ClusterTree;

class Hierarchical : public Node {
 private:
  std::vector<NodeProxy> data;
 public:
  std::array<int64_t, 2> dim = {0, 0};

  // Special member functions
  Hierarchical() = default;

  virtual ~Hierarchical() = default;

  Hierarchical(const Hierarchical& A) = default;

  Hierarchical& operator=(const Hierarchical& A) = default;

  Hierarchical(Hierarchical&& A) = default;

  Hierarchical& operator=(Hierarchical&& A) = default;

  // Overridden functions from Node
  virtual std::unique_ptr<Node> clone() const override;

  virtual std::unique_ptr<Node> move_clone() override;

  virtual const char* type() const override;

  // Conversion constructors
  Hierarchical(NodeProxy&&);

  Hierarchical(const Node& A, int64_t ni_level, int64_t nj_level);

  // Additional constructors
  Hierarchical(int64_t ni_level, int64_t nj_level=1);

  Hierarchical(
    ClusterTree& node,
    void (*func)(
      Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
    ),
    std::vector<double>& x,
    int64_t rank,
    int64_t nleaf,
    int64_t admis,
    int64_t ni_level, int64_t nj_level
  );

  Hierarchical(
    void (*func)(
      Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
    ),
    std::vector<double>& x,
    int64_t ni, int64_t nj,
    int64_t rank,
    int64_t nleaf,
    int64_t admis=1,
    int64_t ni_level=2, int64_t nj_level=2,
    int64_t i_begin=0, int64_t j_begin=0
  );

  // Additional operators
  const NodeProxy& operator[](int64_t i) const;

  NodeProxy& operator[](int64_t i);

  const NodeProxy& operator[](std::array<int64_t, 2> pos) const;

  NodeProxy& operator[](std::array<int64_t, 2> pos);

  const NodeProxy& operator()(int64_t i, int64_t j) const;

  NodeProxy& operator()(int64_t i, int64_t j);

  // Make class usable as range
  std::vector<NodeProxy>::iterator begin();

  std::vector<NodeProxy>::const_iterator begin() const;

  std::vector<NodeProxy>::iterator end();

  std::vector<NodeProxy>::const_iterator end() const;

  // Utility methods
  void blr_col_qr(Hierarchical& Q, Hierarchical& R);

  void split_col(Hierarchical& QL);

  void restore_col(const Hierarchical& Sp, const Hierarchical& QL);

  void col_qr(int64_t j, Hierarchical& Q, Hierarchical &R);

  bool is_admissible(const ClusterTree& node, int64_t dist_to_diag);

  bool is_leaf(const ClusterTree& node, int64_t nleaf);
};

register_class(Hierarchical, Node)

} // namespace hicma

#endif // hicma_classes_hierarchical_h
