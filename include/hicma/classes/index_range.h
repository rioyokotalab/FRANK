#ifndef hicma_classes_index_range_h
#define hicma_classes_index_range_h

#include <vector>


namespace hicma
{

class IndexRange {
 private:
  std::vector<IndexRange> children;
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

  // Indexing
  IndexRange& operator[](int i);
  const IndexRange& operator[](int i) const;

  // Additional methods
  void split(int n_splits);
};

} // namespace hicma

#endif // hicma_classes_index_range_h
