#include "hicma/classes/node_proxy.h"

#include "hicma/classes/node.h"

#include <cassert>


namespace hicma
{

// Reconsider these constructors. Performance testing needed!!
NodeProxy::NodeProxy(const NodeProxy& A) : ptr(A.ptr->clone()) {}

// TODO This might not play nice with NoCopySplit (pointer replacing doesn't cut
// it!). Consider NodeProxyView?
NodeProxy& NodeProxy::operator=(const NodeProxy& A) {
  ptr = A.ptr->clone();
  return *this;
}

NodeProxy::NodeProxy(const Node& A) : ptr(A.clone()) {}

NodeProxy::NodeProxy(Node&& A) : ptr(A.move_clone()) {}

NodeProxy::operator const Node&() const {
  assert(ptr.get() != nullptr);
  return *ptr;
}

NodeProxy::operator Node&() {
  assert(ptr.get() != nullptr);
  return *ptr;
}

const char* NodeProxy::type() const { return ptr->type(); }

} // namespace hicma
