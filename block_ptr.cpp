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

  // template <typename T>
  // const BlockPtr<T> BlockPtr<T>::operator=(const BlockPtr<T>& ptr) {
  //   *(this->get()) = *(ptr.get());
  //   return *this;
  // };

  template <typename T>
  const BlockPtr<T>& BlockPtr<T>::operator=(int i) {
    *(this->get()) = i;
    return *this;
  }

  template <>
  const BlockPtr<Dense>& BlockPtr<Dense>::operator-() const {
    *(this->get()) = this->get()->operator-();
    return *this;
  }

  template <>
  double& BlockPtr<Dense>::operator()(int i, int j) {
    return (*this)(i, j);
  }

  template <>
  const double& BlockPtr<Dense>::operator()(int i, int j) const {
    return (*this)(i, j);
  }

  template <typename T>
  const bool BlockPtr<T>::is(const int enum_id) const {
    return this->get()->is(enum_id);
  }

  template <typename T>
  const char* BlockPtr<T>::is_string() const {
    return this->get()->is_string();
  }

  template <>
  void BlockPtr<Dense>::resize(int i) {
    this->get()->resize(i);
  }

  template <>
  void BlockPtr<Dense>::resize(int i, int j) {
    this->get()->resize(i, j);
  }

  template <typename T>
  double BlockPtr<T>::norm() const {
    return this->get()->norm();
  }

  template <typename T>
  void BlockPtr<T>::print() const {
    return this->get()->print();
  }

  template <typename T>
  void BlockPtr<T>::getrf() {
    return this->get()->getrf();
  }

  // template <typename T>
  // void BlockPtr<T>::trsm(const BlockPtr<T>& A, const char& uplo) {
  //   return this->get()->trsm(A, uplo);
  // }

  template <typename T>
  void BlockPtr<T>::gemm(const BlockPtr<T>& A, const BlockPtr<T>& B) {
    return this->get()->gemm(A, B);
  }


  template class BlockPtr<_Node>;
  template class BlockPtr<Dense>;
  template class BlockPtr<LowRank>;
  template class BlockPtr<Hierarchical>;
}
