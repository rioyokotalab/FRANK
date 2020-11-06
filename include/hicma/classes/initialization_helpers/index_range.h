#ifndef hicma_classes_initialization_helpers_index_range_h
#define hicma_classes_initialization_helpers_index_range_h

#include <cstdint>
#include <tuple>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Hierarchical;

enum { ALONG_ROW, ALONG_COL };

class IndexRange {
 public:
  int64_t start;
  int64_t n;

  // Special member functions
  IndexRange() = default;

  ~IndexRange() = default;

  IndexRange(const IndexRange& A) = default;

  IndexRange& operator=(const IndexRange& A) = default;

  IndexRange(IndexRange&& A) = default;

  IndexRange& operator=(IndexRange&& A) = default;

  // Constructors
  IndexRange(int64_t start, int64_t n);

  IndexRange(std::tuple<int64_t, int64_t> A);

  // Utility methods
  std::vector<IndexRange> split(int64_t n_splits) const;

  std::vector<IndexRange> split_at(int64_t index) const;

  std::vector<IndexRange> split_like(const Hierarchical& A, int along) const;
};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_index_range_h
