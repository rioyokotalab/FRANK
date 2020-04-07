#include "hicma/classes/low_rank_view.h"

#include "hicma/classes/dense_view.h"
#include "hicma/classes/index_range.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <memory>
#include <utility>


namespace hicma
{

std::unique_ptr<Node> LowRankView::clone() const {
  return std::make_unique<LowRankView>(*this);
}

std::unique_ptr<Node> LowRankView::move_clone() {
  return std::make_unique<LowRankView>(std::move(*this));
}

const char* LowRankView::type() const { return "LowRankView"; }

// TODO write safe setters!
// A.U() = B is a shitty way to write things. A.setU(B) is better.
DenseView& LowRankView::U() { return _U; }
const DenseView& LowRankView::U() const { return _U; }

DenseView& LowRankView::S() { return _S; }
const DenseView& LowRankView::S() const { return _S; }

DenseView& LowRankView::V() { return _V; }
const DenseView& LowRankView::V() const { return _V; }

LowRankView::LowRankView(const LowRank& A)
: LowRankView(IndexRange(0, A.dim[0]), IndexRange(0, A.dim[1]), A) {}

LowRankView::LowRankView(
  const IndexRange& row_range, const IndexRange& col_range, const LowRank& A
) {
  assert(row_range.start+row_range.length <= A.dim[0]);
  assert(col_range.start+col_range.length <= A.dim[1]);
  dim[0] = row_range.length;
  dim[1] = col_range.length;
  rank = A.rank;
  U() = DenseView(
    IndexRange(row_range.start, row_range.length), IndexRange(0, A.rank),
    A.U()
  );
  S() = A.S();
  V() = DenseView(
    IndexRange(0, A.rank), IndexRange(col_range.start, col_range.length),
    A.V()
  );
}

} // namespace hicma
