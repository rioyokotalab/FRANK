#include "hicma/classes/initialization_helpers/index_range.h"

#include "hicma/classes/hierarchical.h"
#include "hicma/operations/misc.h"

#include <cstdint>
#include <vector>


namespace hicma
{

IndexRange::IndexRange(int64_t start, int64_t n) : start(start), n(n) {}

std::vector<IndexRange> IndexRange::split(int64_t n_splits) const {
  std::vector<IndexRange> children(n_splits);
  for (int64_t i=0; i<n_splits; ++i) {
    int64_t child_n = (n+n_splits-1) / n_splits;
    int64_t child_start = start + child_n * i;
    if (i == n_splits-1) child_n = n - child_n * (n_splits-1);
    children[i] = IndexRange(child_start, child_n);
  }
  return children;
}

std::vector<IndexRange> IndexRange::split_at(int64_t index) const {
  return {{start, index}, {start+index, n-index}};
}

std::vector<IndexRange> IndexRange::split_like(
  const Hierarchical& like, int along
) const {
  std::vector<IndexRange> children(like.dim[along == ALONG_ROW ? 1 : 0]);
  int64_t child_start = 0;
  for (uint64_t i=0; i<children.size(); ++i) {
    int64_t child_n = (
      along == ALONG_ROW ? get_n_cols(like(0, i)) : get_n_rows(like(i, 0))
    );
    children[i] = IndexRange(child_start, child_n);
    child_start += children[i].n;
  }
  return children;
}

} // namespace hicma
