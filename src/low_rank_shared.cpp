#include "hicma/low_rank_shared.h"

#include "hicma/node.h"
#include "hicma/dense.h"

#include <memory>

#include "yorel/multi_methods.hpp"

namespace hicma
{
LowRankShared::LowRankShared() {
  MM_INIT();
}

LowRankShared::LowRankShared(
  const Dense& S,
  std::shared_ptr<Dense> U, std::shared_ptr<Dense> V
) : Node(S.i_abs, S.j_abs, S.level), U(U), S(S), V(V)
{
  MM_INIT();
}

LowRankShared::LowRankShared(const LowRankShared& A)
  : Node(A.i_abs,A.j_abs,A.level), U(A.U), S(A.S), V(A.V)
{
  MM_INIT();
}

LowRankShared::LowRankShared(LowRankShared&& A) {
  MM_INIT();
  swap(*this, A);
}

LowRankShared* LowRankShared::clone() const {
  return new LowRankShared(*this);
}

void swap(LowRankShared& A, LowRankShared& B) {
  using std::swap;
  swap(static_cast<Node&>(A), static_cast<Node&>(B));
  swap(A.U, B.U);
  swap(A.S, B.S);
  swap(A.V, B.V);
}

const LowRankShared& LowRankShared::operator=(LowRankShared A) {
  swap(*this, A);
  return *this;
}

const char* LowRankShared::type() const { return "LowRankShared"; }

} // namespace hicma
