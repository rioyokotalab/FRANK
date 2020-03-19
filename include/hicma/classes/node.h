#ifndef hicma_classes_node_h
#define hicma_classes_node_h

#include "hicma/classes/index_range.h"

#include "yorel/yomm2/cute.hpp"

#include <memory>


namespace hicma
{

class Node {
 public:
  int i_abs = 0; //! Row number of the node on the current recursion level
  int j_abs = 0; //! Column number of the node on the current recursion level
  int level = 0; //! Recursion level of the node
  IndexRange row_range, col_range;

  // Special member functions
  Node() = default;

  virtual ~Node() = default;

  Node(const Node& A) = default;

  Node& operator=(const Node& A) = default;

  Node(Node&& A) = default;

  Node& operator=(Node&& A) = default;

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

register_class(Node)

} // namespace hicma

#endif // hicma_classes_node_h
