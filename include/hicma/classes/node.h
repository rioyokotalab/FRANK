#ifndef hicma_classes_node_h
#define hicma_classes_node_h
#include "hicma/classes/index_range.h"

#include "yorel/multi_methods.hpp"

#include <memory>
#include <vector>

namespace hicma {

  class Node : public yorel::multi_methods::selector {
  public:
    MM_CLASS(Node);
    int i_abs = 0; //! Row number of the node on the current recursion level
    int j_abs = 0; //! Column number of the node on the current recursion level
    int level = 0; //! Recursion level of the node
    IndexRange row_range, col_range;

    // Special member functions
    Node();

    virtual ~Node();

    Node(const Node& A);

    Node& operator=(const Node& A);

    Node(Node&& A);

    Node& operator=(Node&& A);

    // Virtual functions for inheritance
    // TODO Consider moving these three into multi-methods.
    // That would make inheritance a pure formality (which is good)
    // TODO Or consider making them protected members!
    virtual std::unique_ptr<Node> clone() const;

    virtual std::unique_ptr<Node> move_clone();

    virtual const char* type() const;

    // Additional constructors
    Node(int i_abs, int j_abs, int level);

    Node(
      int i_abs, int j_abs, int level,
      IndexRange row_range, IndexRange col_range
    );

    // Utility methods
    bool is_child(const Node& node) const;
  };

} // namespace hicma

#endif // hicma_classes_node_h
