#include "block.h"

#include "node.h"
#include "hierarchical.h"
#include "low_rank.h"
#include "dense.h"

#include <iostream>
#include <cassert>
#include <memory>

namespace hicma {

Block::Block() = default;

Block::Block(const Block& lvalue) : ptr(lvalue.ptr->clone()) {}
Block::Block(Block&& rvalue) : ptr() {
  swap(*this, rvalue);
}
Block::Block(const Node& ref) : ptr(ref.clone()) {}
Block::Block(Node&& ref) {
  if (ref.is(HICMA_HIERARCHICAL)) {
    ptr = std::make_unique<Hierarchical>();
    swap(static_cast<Hierarchical&>(*ptr), static_cast<Hierarchical&>(ref));
  } else if (ref.is(HICMA_LOWRANK)) {
    ptr = std::make_unique<LowRank>();
    swap(static_cast<LowRank&>(*ptr), static_cast<LowRank&>(ref));
  } else if (ref.is(HICMA_DENSE)) {
    ptr = std::make_unique<Dense>();
    swap(static_cast<Dense&>(*ptr), static_cast<Dense&>(ref));
  } else {
    std::cout << "Not implemented!!" << std::endl; abort();
  }
}

Block::~Block() = default;

void swap(Block& first, Block& second) {
  using std::swap;
  swap(first.ptr, second.ptr);
}

const Block& Block::operator=(Block lvalue) {
  swap(*this, lvalue);
  return *this;
}

const Block& Block::operator=(Node&& ref) {
  if (ref.is(HICMA_HIERARCHICAL)) {
    ptr = std::make_unique<Hierarchical>();
    swap(static_cast<Hierarchical&>(*ptr), static_cast<Hierarchical&>(ref));
  } else if (ref.is(HICMA_LOWRANK)) {
    ptr = std::make_unique<LowRank>();
    swap(static_cast<LowRank&>(*ptr), static_cast<LowRank&>(ref));
  } else if (ref.is(HICMA_DENSE)) {
    ptr = std::make_unique<Dense>();
    swap(static_cast<Dense&>(*ptr), static_cast<Dense&>(ref));
  } else {
    std::cout << "Not implemented!!" << std::endl; abort();
  }
  return *this;
}

const Block& Block::operator=(double a) {
  *ptr = a;
  return *this;
}

Block Block::operator+(const Block& rhs) const {
  return *ptr + *rhs.ptr;
}
const Node& Block::operator+=(const Block& rhs) {
  return *ptr += *rhs.ptr;
}
Block Block::operator-(const Block& rhs) const {
  return *ptr - *rhs.ptr;
}
const Node& Block::operator-=(const Block& rhs) {
  return *ptr -= *rhs.ptr;
}
Block Block::operator*(const Block& rhs) const {
  return *ptr * *rhs.ptr;
}

Block Block::operator+(const Node& rhs) const {
  return *ptr + rhs;
}
const Node& Block::operator+=(const Node& rhs) {
  return *ptr += rhs;
}
Block Block::operator-(const Node& rhs) const {
  return *ptr - rhs;
}
const Node& Block::operator-=(const Node& rhs) {
  return *ptr -= rhs;
}
Block Block::operator*(const Node& rhs) const {
  return *ptr * rhs;
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

const char* Block::is_string() const { return ptr->is_string(); }
const bool Block::is(const int i) const { return ptr->is(i); }

void Block::resize(int i) {
  assert(is(HICMA_DENSE));
  static_cast<Dense*>(ptr.get())->resize(i);
}

void Block::resize(int i, int j) {
  assert(is(HICMA_DENSE));
  static_cast<Dense*>(ptr.get())->resize(i, j);
}

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
