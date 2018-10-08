#include "hierarchical.h"

namespace hicma {

  Block::Block() = default;

  Block::Block(const Block& A) : ptr(A.ptr->clone()) {}

  Block::Block(const Node& A) : ptr(A.clone()) {}

  Block::~Block() = default;

  const Block& Block::operator=(Block A) {
    swap(this->ptr, A.ptr);
    return *this;
  }

  const bool Block::is(const int i) const { return ptr->is(i); }

  void Block::getrf() {
    return ptr->getrf();
  }

  void Block::trsm(const Block& A, const char& uplo) {
    return ptr->trsm(*A.ptr, uplo);
  }

  void Block::trsm(const Node& A, const char& uplo) {
    return ptr->trsm(A, uplo);
  }

  void Block::gemm(const Block& A, const Block& B) {
    return ptr->gemm(*A.ptr, *B.ptr);
  }

}
