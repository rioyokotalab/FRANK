#ifndef hicma_classes_node_h
#define hicma_classes_node_h

#include "hicma/classes/index_range.h"

#include "yorel/yomm2/cute.hpp"

#include <memory>


namespace hicma
{

class Node {
 public:
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
  virtual std::unique_ptr<Node> clone() const = 0;

  virtual std::unique_ptr<Node> move_clone() = 0;

  virtual const char* type() const = 0;
};

register_class(Node)

} // namespace hicma

#endif // hicma_classes_node_h
