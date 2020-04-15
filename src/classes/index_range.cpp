#include "hicma/classes/index_range.h"

#include <cassert>
#include <cstdint>


namespace hicma
{

IndexRange::IndexRange(
  int64_t start, int64_t length
) : start(start), length(length) {}


IndexRange& IndexRange::operator[](int64_t i) {
  return children[i];
}

const IndexRange& IndexRange::operator[](int64_t i) const {
  return children[i];
}

void IndexRange::split(int64_t n_splits) {
  assert(children.empty());
  children.reserve(n_splits);
  for (int64_t i=0; i<n_splits; ++i) {
    int64_t length_child = length/n_splits;
    if (i == n_splits-1)
      length_child = length - (length/n_splits) * (n_splits-1);
    int64_t start_child = length/n_splits * i;
    children.push_back(IndexRange(start_child, length_child));
  }
}

} // namespace hicma
