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
  private:
    std::shared_ptr<Dense> basis;
  public:
    // Special member functions
    SharedBasis() = default;
    ~SharedBasis() = default;
    // TODO Not that these do deep copies - is that in line with the idea of a
    // shared basis?
    SharedBasis(const SharedBasis& A) : basis(std::make_shared<Dense>(A)) {}
    SharedBasis& operator=(const SharedBasis& A) {
      basis = std::make_shared<Dense>(A);
      return *this;
    }
    // TODO Should these be implemented? If we create a temporary from the
    // custom constructor and then move from it, it will share the basis.
    SharedBasis(SharedBasis&& A) = default;
    SharedBasis& operator=(SharedBasis&& A) = default;

    // Additional constructors
    SharedBasis(std::shared_ptr<Dense> A) : basis(A) {};

    // Implicit conversion operators
    operator Dense& () { return *basis; }
    operator const Dense&() const { return *basis; }
  } U, V;
  Dense S;

  // Special member functions
  LowRankShared();

  ~LowRankShared();

  LowRankShared(const LowRankShared& A);

  LowRankShared& operator=(const LowRankShared& A);

  LowRankShared(LowRankShared&& A);

  LowRankShared& operator=(LowRankShared&& A);

  // Overridden functions from Node
  std::unique_ptr<Node> clone() const override;

  std::unique_ptr<Node> move_clone() override;

  const char* type() const override;

  // Additional construcors
  LowRankShared(
    const Dense& S,
    std::shared_ptr<Dense> U, std::shared_ptr<Dense> V
  );
};

} // namespace hicma

#endif // hicma_classes_low_rank_shared_h