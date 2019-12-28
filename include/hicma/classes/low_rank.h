#ifndef hicma_classes_low_rank_h
#define hicma_classes_low_rank_h

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"

#include "yorel/multi_methods.hpp"

#include <memory>

namespace hicma {

  class LowRank : public Node {
  public:
    MM_CLASS(LowRank, Node);
    Dense U, S, V;
    int dim[2];
    int rank;

    // Special member functions
    LowRank();

    ~LowRank();

    LowRank(const LowRank& A);

    LowRank& operator=(const LowRank& A);

    LowRank(LowRank&& A);

    LowRank& operator=(LowRank&& A);

    // Overridden functions from Node
    std::unique_ptr<Node> clone() const override;

    std::unique_ptr<Node> move_clone() override;

    const char* type() const override;

    // Additional constructors
    LowRank(
      int m, int n,
      int k,
      int i_abs=0, int j_abs=0,
      int level=0
    );

    LowRank(const Dense& A, int k);

    // Additional operators
    const LowRank& operator+=(const LowRank& A);

    // Utility methods
    void mergeU(const LowRank& A, const LowRank& B);

    void mergeS(const LowRank& A, const LowRank& B);

    void mergeV(const LowRank& A, const LowRank& B);

  };

} // namespace hicma

#endif // hicma_classes_low_rank_h
