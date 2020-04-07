#include "hicma/classes/index_range.h"

#include <cassert>
#include <vector>


namespace hicma
{

IndexRange::IndexRange(int start, int length) : start(start), length(length) {}


IndexRange& IndexRange::operator[](int i) {
  // TODO Change int to size_t and get rid of the casting issue
  assert(static_cast<unsigned int>(i) < children.size());
  return children[i];
}

const IndexRange& IndexRange::operator[](int i) const {
  assert(static_cast<unsigned int>(i) < children.size());
  return children[i];
}

void IndexRange::split(int n_splits) {
  assert(children.empty());
  children.reserve(n_splits);
  for (int i=0; i<n_splits; ++i) {
    int length_child = length/n_splits;
    if (i == n_splits-1)
      length_child = length - (length/n_splits) * (n_splits-1);
    int start_child = length/n_splits * i;
    children.push_back(IndexRange(start_child, length_child));
  }
}

} // namespace hicma
