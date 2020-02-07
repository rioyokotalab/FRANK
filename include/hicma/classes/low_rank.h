#ifndef hicma_classes_low_rank_h
#define hicma_classes_low_rank_h

#include "hicma/classes/node.h"
#include "hicma/classes/dense.h"

#include "yorel/multi_methods.hpp"

#include <memory>

namespace hicma {

  class LowRank : public Node {
  private:
    Dense _U, _S, _V;
  public:
    MM_CLASS(LowRank, Node);
    int dim[2] = {0, 0};
    int rank = 0;

    // Special member functions
    LowRank();

    virtual ~LowRank();

    LowRank(const LowRank& A);

    LowRank& operator=(const LowRank& A);

    LowRank(LowRank&& A);

    LowRank& operator=(LowRank&& A);

    // Overridden functions from Node
    virtual std::unique_ptr<Node> clone() const override;

    virtual std::unique_ptr<Node> move_clone() override;

    virtual const char* type() const override;

    // Getters and setters
    virtual Dense& U();
    virtual const Dense& U() const;

    virtual Dense& S();
    virtual const Dense& S() const;

    virtual Dense& V();
    virtual const Dense& V() const;

    // Additional constructors
    LowRank(const Node& node, int k, bool node_only=false);

    LowRank(
      int m, int n,
      int k,
      int i_abs=0, int j_abs=0,
      int level=0
    );

    LowRank(const Dense& A, int k);

    // Utility methods
    void mergeU(const LowRank& A, const LowRank& B);

    void mergeS(const LowRank& A, const LowRank& B);

    void mergeV(const LowRank& A, const LowRank& B);

    LowRank get_part(const Node& node) const;
  };

} // namespace hicma

#endif // hicma_classes_low_rank_h
