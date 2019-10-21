#ifndef any_h
#define any_h
#include <memory>

namespace hicma {

  class Node;

  class NodeProxy {
  public:
    std::unique_ptr<Node> ptr;

    NodeProxy();

    NodeProxy(const NodeProxy& A);

    NodeProxy(const Node& A);

    ~NodeProxy();

    friend void swap(NodeProxy&, NodeProxy&);

    const NodeProxy& operator=(NodeProxy A);

    const char* type() const;

    double norm() const;

    void print() const;

    void transpose();

  };
}

#endif
