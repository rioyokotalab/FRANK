#include "hicma/classes/low_rank_shared.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/extension_headers/classes.h"
#include "hicma/operations/BLAS/gemm.h"

#include <memory>
#include <utility>

#include "yorel/multi_methods.hpp"

namespace hicma
{

LowRankShared::LowRankShared() : Node() { MM_INIT(); }

LowRankShared::~LowRankShared() = default;

LowRankShared::LowRankShared(const LowRankShared& A) {
  MM_INIT();
  *this = A;
}

LowRankShared& LowRankShared::operator=(const LowRankShared& A) = default;

LowRankShared::LowRankShared(LowRankShared&& A) {
  MM_INIT();
  *this = std::move(A);
}

LowRankShared& LowRankShared::operator=(LowRankShared&& A) = default;

std::unique_ptr<Node> LowRankShared::clone() const {
  return std::make_unique<LowRankShared>(*this);
}

std::unique_ptr<Node> LowRankShared::move_clone() {
  return std::make_unique<LowRankShared>(std::move(*this));
}

const char* LowRankShared::type() const { return "LowRankShared"; }

LowRankShared::LowRankShared(
  const Node& node,
  const Dense& S,
  std::shared_ptr<Dense> U, std::shared_ptr<Dense> V
) : Node(node), U(U), S(S), V(V), dim{U->dim[0], V->dim[1]}, rank(S.dim[0])
{
  MM_INIT();
}

BEGIN_SPECIALIZATION(make_dense, Dense, const LowRankShared& A){
  // TODO exactly the same as the LowRank method. Consider inheritance!
  Dense B(A.dim[0], A.dim[1]);
  Dense UxS(A.dim[0], A.rank);
  gemm(A.U, A.S, UxS, 1, 0);
  gemm(UxS, A.V, B, 1, 0);
  return B;
} END_SPECIALIZATION;

} // namespace hicma
