#include "hierarchical.h"

namespace hicma {

  Block::Block() = default;

  Block::Block(const Block& A) : ptr(A.ptr->clone()) {}

  Block::Block(const Node& A) : ptr(A.clone()) {}

  Block::~Block() = default;

  void swap(Block& A, Block& B) {
    std::swap(A.ptr, B.ptr);
  }

  const Block& Block::operator=(Block A) {
    swap(*this, A);
    return *this;
  }

  const Node& Block::operator[](const int i) const {
    if (is(HICMA_HIERARCHICAL)) {
      return static_cast<const Hierarchical&>(*ptr)[i];
    } else return *ptr;
  }

  Block& Block::operator[](const int i) {
    if (is(HICMA_HIERARCHICAL)) {
      return static_cast<Hierarchical&>(*ptr)[i];
    } else return *this;
  }

  const Node& Block::operator()(const int i, const int j) const {
    if (is(HICMA_HIERARCHICAL)) {
      return static_cast<const Hierarchical&>(*ptr)(i, j);
    } else return *ptr;
  }

  Block& Block::operator()(const int i, const int j) {
    if (is(HICMA_HIERARCHICAL)) {
      return static_cast<Hierarchical&>(*ptr)(i, j);
    } else return *this;
  }

  const char* Block::type() const { return ptr->type(); }

  const bool Block::is(const int i) const { return ptr->is(i); }

  double Block::norm() const {
    return ptr->norm();
  }

  void Block::print() const {
    return ptr->print();
  }

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

  void Block::gemm(const Node& A, const Node& B) {
    return ptr->gemm(A, B);
  }

}
