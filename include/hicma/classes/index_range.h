#ifndef hicma_classes_index_range_h
#define hicma_classes_index_range_h

#include <vector>

namespace hicma
{

class IndexRange {
public:
  int start = 0;
  int length = 0;

  // Special member functions
  IndexRange() = default;

  ~IndexRange() = default;

  IndexRange(const IndexRange&) = default;

  IndexRange& operator=(const IndexRange&) = default;

  IndexRange(IndexRange&&) = default;

  IndexRange& operator=(IndexRange&&) = default;

  // Additional constructors
  IndexRange(int start, int length);

  // Additional methods
  std::vector<IndexRange> split(int n_splits);

  bool is_subrange(const IndexRange& range) const;
};

} // namespace hicma

#endif // hicma_classes_index_range_h
