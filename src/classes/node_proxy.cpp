#include "hicma/classes/node_proxy.h"

#include "hicma/classes/node.h"

#include <cassert>
#include <iostream>
#include <utility>

namespace hicma {

  NodeProxy::NodeProxy() = default;

  NodeProxy::NodeProxy(const NodeProxy& A) : ptr(A->clone()) {}

  NodeProxy::NodeProxy(const Node& A) : ptr(A.clone()) {}

  NodeProxy::NodeProxy(Node&& A) : ptr(A.move_clone()) {}

  NodeProxy::~NodeProxy() = default;

  const Node& NodeProxy::operator*() const {
    assert(ptr.get() != nullptr);
    return *ptr.get();
  }

  Node& NodeProxy::operator*() {
    assert(ptr.get() != nullptr);
    return *ptr.get();
  }

  const Node* NodeProxy::operator->() const {
    return ptr.get();
  }

  Node* NodeProxy::operator->() {
    return ptr.get();
  }

  NodeProxy::operator const Node& () const {
    assert(ptr.get() != nullptr);
    return *ptr;
  }

  NodeProxy::operator Node& () {
    assert(ptr.get() != nullptr);
    return *ptr;
  }

  void swap(NodeProxy& A, NodeProxy& B){
    using std::swap;
    swap(A.ptr, B.ptr);
  }

  const NodeProxy& NodeProxy::operator=(NodeProxy A) {
    swap(*this, A);
    return *this;
  }

  const char* NodeProxy::type() const { return ptr->type(); }

} // namespace hicma
