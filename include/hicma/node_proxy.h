#ifndef any_h
#define any_h

#include <memory>

namespace hicma {

  class Node;

  class NodeProxy {
  private:
    std::unique_ptr<Node> ptr;
  public:
    NodeProxy();

    NodeProxy(const NodeProxy& A);

    NodeProxy(const Node& A);

    NodeProxy(Node&& A);

    ~NodeProxy();

    const Node& operator*() const;

    Node& operator*();

    const Node* operator->() const;

    Node* operator->();

    operator const Node& () const;

    operator Node& ();

    friend void swap(NodeProxy&, NodeProxy&);

    const NodeProxy& operator=(NodeProxy A);

    const char* type() const;

    double norm() const;

  };
}

#endif
