#ifndef hicma_classes_low_rank_shared_h
#define hicma_classes_low_rank_shared_h

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"

#include <array>
#include <cstdint>
#include <memory>


namespace hicma
{

class LowRankShared : public Matrix {
 public:
  // TODO Make these members private just like in LowRank
  class SharedBasis {
   private:
    std::shared_ptr<Dense> basis;
   public:
    // Special member functions
    SharedBasis() = default;
    ~SharedBasis() = default;
    // TODO Not that these do not copy - they simply share the same basis with
    // the copied from object.
    SharedBasis(const SharedBasis& A) = default;
    SharedBasis& operator=(const SharedBasis& A) = default;
    // TODO Should these be implemented? If we create a temporary from the
    // custom constructor and then move from it, it will share the basis.
    SharedBasis(SharedBasis&& A) = default;
    SharedBasis& operator=(SharedBasis&& A) = default;

    // Additional constructors
    SharedBasis(std::shared_ptr<Dense> A) : basis(A) {};

    // Additional constructors
    SharedBasis& operator=(std::shared_ptr<Dense> A) {
      basis = A;
      return *this;
    };

    // Implicit conversion operators
    operator Dense& () { return *basis; }
    operator const Dense&() const { return *basis; }

    // Comparison operator
    bool operator==(const SharedBasis& B) const {
      return basis == B.basis;
    }
  } U, V;
  Dense S;
  std::array<int64_t, 2> dim = {0, 0};
  int64_t rank = 0;

  // Special member functions
  LowRankShared() = default;

  ~LowRankShared() = default;

  LowRankShared(const LowRankShared& A) = default;

  LowRankShared& operator=(const LowRankShared& A) = default;

  LowRankShared(LowRankShared&& A) = default;

  LowRankShared& operator=(LowRankShared&& A) = default;

  // Additional construcors
  LowRankShared(
    const Dense& S, std::shared_ptr<Dense> U, std::shared_ptr<Dense> V);
};

} // namespace hicma

#endif // hicma_classes_low_rank_shared_h
