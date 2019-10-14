#include "hicma/node_proxy.h"
#include "hicma/low_rank.h"
#include "hicma/hierarchical.h"
#include "hicma/operations.h"

#include <iostream>

namespace hicma {

  NodeProxy::NodeProxy() = default;

  NodeProxy::NodeProxy(const NodeProxy& A) : ptr(A.ptr->clone()) {}

  NodeProxy::NodeProxy(const Node& A) : ptr(A.clone()) {}

  NodeProxy::~NodeProxy() = default;

  void swap(NodeProxy& A, NodeProxy& B){
    A.ptr.swap(B.ptr);
  }

  const NodeProxy& NodeProxy::operator=(NodeProxy A) {
    this->ptr = std::move(A.ptr);
    return *this;
  }

  const char* NodeProxy::type() const { return ptr->type(); }

  double NodeProxy::norm() const {
    return ptr->norm();
  }

  void NodeProxy::print() const {
    ptr->print();
  }

  void NodeProxy::transpose() {
    ptr->transpose();
  }

  // void NodeProxy::geqrt(NodeProxy& T) {
  //   if(T.is(HICMA_DENSE)) {
  //     ptr->geqrt(static_cast<Dense&>(*T.ptr));
  //   }
  //   else if(T.is(HICMA_HIERARCHICAL)) {
  //     ptr->geqrt(static_cast<Hierarchical&>(*T.ptr));
  //   }
  //   else {
  //     std::cerr << "Input matrix for geqrt must be Hierarchical or Dense." << std::endl;
  //     abort();
  //   }
  // }

  // void NodeProxy::larfb(const NodeProxy& Y, const NodeProxy& T, const bool trans) {
  //   if(Y.is(HICMA_DENSE) && T.is(HICMA_DENSE)) {
  //     ptr->larfb(static_cast<Dense&>(*Y.ptr), static_cast<Dense&>(*T.ptr), trans);
  //   }
  //   else if(Y.is(HICMA_HIERARCHICAL) && T.is(HICMA_HIERARCHICAL)) {
  //     ptr->larfb(static_cast<Hierarchical&>(*Y.ptr), static_cast<Hierarchical&>(*T.ptr), trans);
  //   }
  //   else {
  //     std::cerr << "Input matrix for larfb must be (Dense,Dense) or (Hierarchical,Hierarchical)." << std::endl;
  //     abort();
  //   }
  // }

  // void NodeProxy::tpqrt(NodeProxy& A, NodeProxy& T) {
  //   if(A.is(HICMA_DENSE) && T.is(HICMA_DENSE)) {
  //     ptr->tpqrt(static_cast<Dense&>(*A.ptr), static_cast<Dense&>(*T.ptr));
  //   }
  //   else if(A.is(HICMA_HIERARCHICAL) && T.is(HICMA_DENSE)) {
  //     ptr->tpqrt(static_cast<Hierarchical&>(*A.ptr), static_cast<Dense&>(*T.ptr));
  //   }
  //   else if(A.is(HICMA_HIERARCHICAL) && T.is(HICMA_HIERARCHICAL)) {
  //     ptr->tpqrt(static_cast<Hierarchical&>(*A.ptr), static_cast<Hierarchical&>(*T.ptr));
  //   }
  //   else {
  //     std::cerr << "Input matrix for tpqrt must be (Dense,Dense), (Hierarchical, Dense) or (Hierarchical,Hierarchical)." << std::endl;
  //     abort();
  //   }
  // }

}
