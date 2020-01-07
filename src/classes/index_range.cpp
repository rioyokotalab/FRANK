#include "hicma/classes/index_range.h"

namespace hicma
{

IndexRange::IndexRange() = default;

IndexRange::~IndexRange() = default;

IndexRange::IndexRange(const IndexRange&) = default;

IndexRange& IndexRange::operator=(const IndexRange&) = default;

IndexRange::IndexRange(IndexRange&&) = default;

IndexRange& IndexRange::operator=(IndexRange&&) = default;

IndexRange::IndexRange(int start, int length)
: start(start), length(length) {}

std::vector<IndexRange> IndexRange::split(int n_splits) {
  std::vector<IndexRange> children(n_splits);
  for (int i=0; i<n_splits; ++i) {
    int length_child = length/n_splits;
    if (i == n_splits-1)
      length_child = length - (length/n_splits) * (n_splits-1);
    int start_child = start + length/n_splits * i;
    children[i] = IndexRange(start_child, length_child);
  }
  return children;
}

bool IndexRange::is_subrange(const IndexRange& range) const {
  bool out = range.start >= start;
  out &= range.start < start + length;
  out &= range.start + range.length <= start + length;
  return out;
}

} // namespace hicma
