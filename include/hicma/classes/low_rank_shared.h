#ifndef hicma_classes_low_rank_shared_h
#define hicma_classes_low_rank_shared_h

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"

#include <memory>

#include "yorel/multi_methods.hpp"

namespace hicma {

class LowRankShared : public Node {
 public:
  MM_CLASS(LowRankShared, Node);
  int dim[2];
  int rank;

  class SharedBasis {
  public:
    // operator Dense& () { return *value; }
    operator const Dense& () const { return *value; }
    SharedBasis() = default;
    SharedBasis(std::shared_ptr<Dense> A) : value(A) {};
  private:
    std::shared_ptr<Dense> value;
  } U, V;
  Dense S;

  LowRankShared();

  LowRankShared(
    const Dense& S,
    std::shared_ptr<Dense> U, std::shared_ptr<Dense> V
  );

  LowRankShared(const LowRankShared& A);

  LowRankShared(LowRankShared&& A);

  std::unique_ptr<Node> clone() const override;

  std::unique_ptr<Node> move_clone() override;

  friend void swap(LowRankShared& A, LowRankShared& B);

  const LowRankShared& operator=(LowRankShared A);

  const char* type() const override;
};

} // namespace hicma

#endif // hicma_classes_low_rank_shared_h