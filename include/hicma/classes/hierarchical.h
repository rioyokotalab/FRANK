#ifndef hicma_classes_hierarchical_h
#define hicma_classes_hierarchical_h

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"

#include <vector>
#include <memory>
#include <tuple>

#include "yorel/yomm2/cute.hpp"

namespace hicma {

  class Dense;
  class LowRank;

  class Hierarchical : public Node {
  private:
    std::vector<NodeProxy> data;
  public:
    int dim[2] = {0, 0};

    // Special member functions
    Hierarchical();

    virtual ~Hierarchical();

    Hierarchical(const Hierarchical& A);

    Hierarchical& operator=(const Hierarchical& A);

    Hierarchical(Hierarchical&& A);

    Hierarchical& operator=(Hierarchical&& A);

    // Overridden functions from Node
    virtual std::unique_ptr<Node> clone() const override;

    virtual std::unique_ptr<Node> move_clone() override;

    virtual const char* type() const override;

    // Conversion constructors
    Hierarchical(NodeProxy&&);

    Hierarchical(
      const Node& node, int ni_level, int nj_level, bool node_only=false);

    // Additional constructors
    Hierarchical(
      int ni_level, int nj_level=1,
      int i_abs=0, int j_abs=0,
      int level=0
    );

    Hierarchical(
      const Node& node,
      void (*func)(Dense& A, std::vector<double>& x),
      std::vector<double>& x,
      int rank,
      int nleaf,
      int admis=1,
      int ni_level=2, int nj_level=2
    );

    Hierarchical(
      void (*func)(Dense& A, std::vector<double>& x),
      std::vector<double>& x,
      int ni, int nj,
      int rank,
      int nleaf,
      int admis=1,
      int ni_level=2, int nj_level=2,
      int i_begin=0, int j_begin=0,
      int i_abs=0, int j_abs=0,
      int level=0
    );

    // Additional operators
    const NodeProxy& operator[](int i) const;

    NodeProxy& operator[](int i);

    const NodeProxy& operator()(int i, int j) const;

    NodeProxy& operator()(int i, int j);

    // Make class usable as range
    std::vector<NodeProxy>::iterator begin();

    std::vector<NodeProxy>::const_iterator begin() const;

    std::vector<NodeProxy>::iterator end();

    std::vector<NodeProxy>::const_iterator end() const;

    // Utility methods
    void blr_col_qr(Hierarchical& Q, Hierarchical& R);

    void split_col(Hierarchical& QL);

    void restore_col(const Hierarchical& Sp, const Hierarchical& QL);

    void col_qr(int j, Hierarchical& Q, Hierarchical &R);

    void create_children();

    bool is_admissible(const Node& node, int dist_to_diag);

    bool is_leaf(const Node& node, int nleaf);

    std::tuple<int, int> get_rel_pos_child(const Node& node);
  };

  register_class(Hierarchical, Node);

} // namespace hicma

#endif // hicma_classes_hierarchical_h
