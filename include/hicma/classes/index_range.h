#ifndef hicma_classes_index_range_h
#define hicma_classes_index_range_h

#include <vector>

namespace hicma
{

class IndexRange {
public:
  int start;
  int length;

  // Special member functions
  IndexRange();

  ~IndexRange();

  IndexRange(const IndexRange&);

  IndexRange& operator=(const IndexRange&);

  IndexRange(IndexRange&&);

  IndexRange& operator=(IndexRange&&);

  // Additional constructors
  IndexRange(int start, int length);

  // Additional methods
  std::vector<IndexRange> split(int n_splits);

  bool is_subrange(const IndexRange& range) const;
};

} // namespace hicma

#endif // hicma_classes_index_range_h
