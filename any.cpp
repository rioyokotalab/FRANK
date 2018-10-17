#include "any.h"
#include "low_rank.h"
#include "hierarchical.h"

#include <iostream>

namespace hicma {

  Any::Any() = default;

  Any::Any(const Any& A) : ptr(A.ptr->clone()) {}

  Any::Any(const Node& A) : ptr(A.clone()) {}

  Any::Any(Node&& A) {
    if (A.is(HICMA_HIERARCHICAL)) {
      ptr = std::unique_ptr<Hierarchical>(new Hierarchical);
      swap(static_cast<Hierarchical&>(*ptr), static_cast<Hierarchical&>(A));
    } else if (A.is(HICMA_LOWRANK)) {
      ptr = std::unique_ptr<LowRank>(new LowRank);
      swap(static_cast<LowRank&>(*ptr), static_cast<LowRank&>(A));
    } else if (A.is(HICMA_DENSE)) {
      ptr = std::unique_ptr<Dense>(new Dense);
      swap(static_cast<Dense&>(*ptr), static_cast<Dense&>(A));
    } else {
      std::cerr << "Node is of an undefined type." << std::endl;
      abort();
    }
  }

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
