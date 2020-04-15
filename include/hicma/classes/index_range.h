#ifndef hicma_classes_index_range_h
#define hicma_classes_index_range_h

#include <cstdint>
#include <vector>


namespace hicma
{

class IndexRange {
 private:
  std::vector<IndexRange> children;
 public:
  int64_t start = 0;
  int64_t length = 0;

  // Special member functions
  IndexRange() = default;

  ~IndexRange() = default;

  IndexRange(const IndexRange&) = default;

  IndexRange& operator=(const IndexRange&) = default;

  IndexRange(IndexRange&&) = default;

  IndexRange& operator=(IndexRange&&) = default;

  // Additional constructors
  IndexRange(int64_t start, int64_t length);

  // Indexing
  IndexRange& operator[](int64_t i);
  const IndexRange& operator[](int64_t i) const;

  // Additional methods
  void split(int64_t n_splits);
};

} // namespace hicma

#endif // hicma_classes_index_range_h
