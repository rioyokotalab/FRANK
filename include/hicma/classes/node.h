#ifndef hicma_classes_node_h
#define hicma_classes_node_h


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
};

} // namespace hicma

#endif // hicma_classes_node_h
