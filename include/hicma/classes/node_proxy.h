#ifndef hicma_classes_node_proxy_h
#define hicma_classes_node_proxy_h

#include <memory>

namespace hicma {

  class Node;

  class NodeProxy {
  private:
    std::unique_ptr<Node> ptr;
  public:
    // Special member functions
    NodeProxy() = default;

    ~NodeProxy() = default;

    NodeProxy(const NodeProxy& A);

    NodeProxy& operator=(const NodeProxy& A);

    NodeProxy(NodeProxy&& A) = default;

    NodeProxy& operator=(NodeProxy&& A) = default;

    // Additional constructors from Node to allow implicit conversion
    NodeProxy(const Node& A);

    NodeProxy(Node&& A);

    // Conversion operator to Node&. We want to write our operations as
    // operation(Node&, Node&) and not have to write a list of overloads that
    // cover cases where we pass operation(H(0, 0), H(0, 1)).
    // If we define an implicit copy/move constructor on the Node class, the
    // derived types get cut short since we would need a copy/move to implement
    // them and would have no way of knowing which of the types we derived from
    // Node is actually pointed to by ptr of NodeProxy.
    operator const Node&() const;

    operator Node&();

    // Utility methods
    const char* type() const;

  };

} // namespace hicma

#endif // hicma_classes_node_proxy_h
