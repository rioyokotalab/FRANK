#ifndef hicma_classes_hierarchical_h
#define hicma_classes_hierarchical_h

#include "hicma/classes/node.h"
#include "hicma/classes/node_proxy.h"

#include <vector>
#include <memory>
#include <tuple>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma {

  class Dense;
  class LowRank;

  class Hierarchical : public Node {
  private:
    std::vector<NodeProxy> data;
  public:
    MM_CLASS(Hierarchical, Node);
    int dim[2];

    // Special member functions
    Hierarchical();

    ~Hierarchical();

    Hierarchical(const Hierarchical& A);

    Hierarchical& operator=(const Hierarchical& A);

    Hierarchical(Hierarchical&& A);

    Hierarchical& operator=(Hierarchical&& A);

    // Overridden functions from Node
    std::unique_ptr<Node> clone() const override;

    std::unique_ptr<Node> move_clone() override;

    const char* type() const override;

    // Conversion constructors
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
      void (*func)(
        std::vector<double>& data,
        std::vector<double>& x,
        int ni, int nj,
        int i_begin, int j_begin
      ),
      std::vector<double>& x,
      int rank,
      int nleaf,
      int admis=1,
      int ni_level=2, int nj_level=2
    );

    Hierarchical(
      void (*func)(
        std::vector<double>& data,
        std::vector<double>& x,
        int ni, int nj,
        int i_begin, int j_begin
      ),
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

  Hierarchical make_hierarchical(
    const Node& A, int ni_level, int nj_level);

  MULTI_METHOD(
    make_hierarchical_omm, Hierarchical,
    const virtual_<Node>&, int ni_level, int nj_level
  );

} // namespace hicma

#endif // hicma_classes_hierarchical_h
