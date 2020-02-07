#ifndef hicma_classes_low_rank_view_h
#define hicma_classes_low_rank_view_h

#include "hicma/classes/dense_view.h"
#include "hicma/classes/low_rank.h"

#include "yorel/multi_methods.hpp"

#include <memory>

namespace hicma {

  class Node;

  class LowRankView : public LowRank {
  private:
    DenseView _U, _S, _V;
  public:
    MM_CLASS(LowRankView, LowRank);

    // Special member functions
    LowRankView();

    ~LowRankView();

    LowRankView(const LowRankView& A);

    LowRankView& operator=(const LowRankView& A);

    LowRankView(LowRankView&& A);

    LowRankView& operator=(LowRankView&& A);

    // Overridden functions from Node
    std::unique_ptr<Node> clone() const override;

    std::unique_ptr<Node> move_clone() override;

    const char* type() const override;

    DenseView& U() override;
    const DenseView& U() const override;

    DenseView& S() override;
    const DenseView& S() const override;

    DenseView& V() override;
    const DenseView& V() const override;

    LowRankView(const Node& node, const LowRank& A);

  };

} // namespace hicma

#endif // hicma_classes_low_rank_view_h
