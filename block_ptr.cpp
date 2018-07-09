#include "block_ptr.h"
#include "node.h"
#include "dense.h"
#include "low_rank.h"
#include "hierarchical.h"

namespace hicma {
  template <typename T>
  BlockPtr<T>::BlockPtr() : std::shared_ptr<T>() {};

  template <typename T>
  BlockPtr<T>::BlockPtr(nullptr_t ptr) : std::shared_ptr<T>(ptr) {};

  template <typename T>
  BlockPtr<T>::BlockPtr(std::shared_ptr<T> ptr) : std::shared_ptr<T>(ptr) {};

  template <typename T>
  BlockPtr<T>::BlockPtr(T* ptr) : std::shared_ptr<T>(ptr) {};

  template <typename T>
  double BlockPtr<T>::norm() {
    return this->get()->norm();
  }

  template <typename T>
  void BlockPtr<T>::print() {
    return this->get()->print();
  }

  template <typename T>
  void BlockPtr<T>::getrf() {
    return this->get()->getrf();
  }

  template <typename T>
  void BlockPtr<T>::trsm(const Node& A, const char& uplo) {
    return this->get()->trsm(A, uplo);
  }

  template <typename T>
  void BlockPtr<T>::gemm(const Node& A, const Node& B) {
    return this->get()->gemm(A, B);
  }

  template <typename T>
  void BlockPtr<T>::gemm(const Node& A, BlockPtr<T> B) {
    return this->get()->gemm(A, *B);
  }


  template class BlockPtr<Node>;
  template class BlockPtr<Dense>;
  template class BlockPtr<LowRank>;
  template class BlockPtr<Hierarchical>;
}
