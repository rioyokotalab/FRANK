#ifndef hicma_classes_node_h
#define hicma_classes_node_h

#include "yorel/multi_methods.hpp"

#include <memory>

namespace hicma {

  class Node : public yorel::multi_methods::selector {
  public:
    MM_CLASS(Node);
    int i_abs; //! Row number of the node on the current recursion level
    int j_abs; //! Column number of the node on the current recursion level
    int level; //! Recursion level of the node

    Node();

    Node(const int i_abs, const int j_abs, const int level);

    Node(const Node& A);

    virtual ~Node();

    virtual std::unique_ptr<Node> clone() const;

    virtual std::unique_ptr<Node> move_clone();

    const Node& operator=(Node&& A);

    virtual const char* type() const;

  };

} // namespace hicma

#endif // hicma_classes_node_h
