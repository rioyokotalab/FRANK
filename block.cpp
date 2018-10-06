#include "hierarchical.h"

namespace hicma {

  Block::Block() = default;

  Block::Block(const Block& A) : ptr(A.ptr->clone()) {}

  Block::Block(Block&& A) : ptr() {
    swap(*this, A);
  }

  Block::Block(const Node& A) : ptr(A.clone()) {}

  Block::Block(Node&& A) {
    if (A.is(HICMA_HIERARCHICAL)) {
      ptr = std::make_unique<Hierarchical>();
      swap(static_cast<Hierarchical&>(*ptr), static_cast<Hierarchical&>(A));
    } else if (A.is(HICMA_LOWRANK)) {
      ptr = std::make_unique<LowRank>();
      swap(static_cast<LowRank&>(*ptr), static_cast<LowRank&>(A));
    } else if (A.is(HICMA_DENSE)) {
      ptr = std::make_unique<Dense>();
      swap(static_cast<Dense&>(*ptr), static_cast<Dense&>(A));
    } else {
      std::cerr << "Node is of an undefined type." << std::endl;
      abort();
    }
  }

  Block::~Block() = default;

  void swap(Block& A, Block& B) {
    std::swap(A.ptr, B.ptr);
  }

  const Block& Block::operator=(Block A) {
    swap(*this, A);
    return *this;
  }

  const Block& Block::operator=(Node&& A) {
    if (A.is(HICMA_HIERARCHICAL)) {
      ptr = std::make_unique<Hierarchical>();
      swap(static_cast<Hierarchical&>(*ptr), static_cast<Hierarchical&>(A));
    } else if (A.is(HICMA_LOWRANK)) {
      ptr = std::make_unique<LowRank>();
      swap(static_cast<LowRank&>(*ptr), static_cast<LowRank&>(A));
    } else if (A.is(HICMA_DENSE)) {
      ptr = std::make_unique<Dense>();
      swap(static_cast<Dense&>(*ptr), static_cast<Dense&>(A));
    } else {
      std::cerr << "Node is of an undefined type." << std::endl;
      abort();
    }
    return *this;
  }

  const Block& Block::operator=(double a) {
    *ptr = a;
    return *this;
  }

  Block Block::operator+(const Block& A) const {
    return *ptr + *A.ptr;
  }

  const Node& Block::operator+=(const Block& A) {
    return *ptr += *A.ptr;
  }

  Block Block::operator-(const Block& A) const {
    return *ptr - *A.ptr;
  }

  const Node& Block::operator-=(const Block& A) {
    return *ptr -= *A.ptr;
  }

  Block Block::operator+(const Node& A) const {
    return *ptr + A;
  }

  const Node& Block::operator+=(const Node& A) {
    return *ptr += A;
  }

  Block Block::operator-(const Node& A) const {
    return *ptr - A;
  }

  const Node& Block::operator-=(const Node& A) {
    return *ptr -= A;
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
