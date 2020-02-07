#include "hicma/classes/low_rank_view.h"

#include "hicma/classes/node.h"
#include "hicma/classes/dense_view.h"
#include "hicma/classes/low_rank.h"

#include "yorel/multi_methods.hpp"

#include <memory>
#include <utility>

namespace hicma {

  LowRankView::LowRankView() : LowRank() { MM_INIT(); }

  LowRankView::~LowRankView() = default;

  LowRankView::LowRankView(const LowRankView& A) {
    MM_INIT();
    *this = A;
  }
  LowRankView& LowRankView::operator=(const LowRankView& A) = default;

  LowRankView::LowRankView(LowRankView&& A) {
    MM_INIT();
    *this = std::move(A);
  }

  LowRankView& LowRankView::operator=(LowRankView&& A) = default;

  std::unique_ptr<Node> LowRankView::clone() const {
    return std::make_unique<LowRankView>(*this);
  }

  std::unique_ptr<Node> LowRankView::move_clone() {
    return std::make_unique<LowRankView>(std::move(*this));
  }

  const char* LowRankView::type() const {
    return "LowRankView";
  }

  // TODO write safe setters!
  // A.U() = B is a shitty way to write things. A.setU(B) is better.
  DenseView& LowRankView::U() { return _U; }
  const DenseView& LowRankView::U() const { return _U; }

  DenseView& LowRankView::S() { return _S; }
  const DenseView& LowRankView::S() const { return _S; }

  DenseView& LowRankView::V() { return _V; }
  const DenseView& LowRankView::V() const { return _V; }

  LowRankView::LowRankView(const Node& node, const LowRank& A)
  : LowRank(node, A.rank, true) {
    MM_INIT();
    int rel_row_start = (
      node.row_range.start-A.row_range.start + A.U().row_range.start);
    U() = DenseView(Node(
      0, 0, A.U().level+1,
      IndexRange(rel_row_start, node.row_range.length),
      IndexRange(0, A.rank)
    ), A.U());
    S() = DenseView(
      Node(0, 0, A.S().level+1, IndexRange(0, A.rank), IndexRange(0, A.rank)),
      A.S()
    );
    int rel_col_start = (
      node.col_range.start-A.col_range.start + A.V().col_range.start);
    V() = DenseView(Node(
      0, 0, A.V().level+1,
      IndexRange(0, A.rank),
      IndexRange(rel_col_start, node.col_range.length)
    ), A.V());
  }

} // namespace hicma
