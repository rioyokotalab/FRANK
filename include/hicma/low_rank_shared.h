#ifndef low_rank_shared_h
#define low_rank_shared_h

#include "hicma/node.h"
#include "hicma/dense.h"

#include <memory>

#include "yorel/multi_methods.hpp"

namespace hicma {

class LowRankShared : public Node {
 public:
  MM_CLASS(LowRankShared, Node);
  std::shared_ptr<Dense> U, V;
  Dense S;

  LowRankShared();

  LowRankShared(
    const Dense& S,
    std::shared_ptr<Dense> U, std::shared_ptr<Dense> V
  );

  LowRankShared(const LowRankShared& A);

  LowRankShared(LowRankShared&& A);

  LowRankShared* clone() const override;

  friend void swap(LowRankShared& A, LowRankShared& B);

  const LowRankShared& operator=(LowRankShared A);

  const char* type() const override;
};

} // namespace hicma

#endif // low_rank_shared_h