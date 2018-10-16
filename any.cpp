#include "hierarchical.h"

namespace hicma {

  Any::Any() = default;

  Any::Any(const Any& A) : ptr(A.ptr->clone()) {}

  Any::Any(const Node& A) : ptr(A.clone()) {}

  Any::~Any() = default;

  const Any& Any::operator=(Any A) {
    this->ptr = std::move(A.ptr);
    return *this;
  }

  bool Any::is(const int i) const { return ptr->is(i); }

  void Any::getrf() {
    return ptr->getrf();
  }

  void Any::trsm(const Any& A, const char& uplo) {
    return ptr->trsm(*A.ptr, uplo);
  }

  void Any::gemm(const Any& A, const Any& B, const int& alpha, const int& beta) {
    return ptr->gemm(*A.ptr, *B.ptr, alpha, beta);
  }

}
