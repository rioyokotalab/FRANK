/**
 * @file index_range.h
 * @brief Include the `IndexTree` class.
 *
 * @copyright Copyright (c) 2020
 */
#ifndef hicma_classes_initialization_helpers_index_range_h
#define hicma_classes_initialization_helpers_index_range_h

#include <cstdint>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

template<typename T>
class Hierarchical;

/**
 * @brief Choice along which axis to perform some operations on `IndexTree`
 */
enum SplitAxis {
  /**
   * @brief Perform operation along the rows.
   */
  ALONG_ROW,
  /**
   * @brief Perform operation along the column.
   */
  ALONG_COL
};

/**
 * @brief Simple class to describe a range of indices
 *
 * This class stores the beginning and end of an index range. Indices are not
 * stored individually.
 */
class IndexRange {
 public:
  /**
   * @brief Start of the index range
   */
  int64_t start;
  /**
   * @brief Length of the index range
   */
  int64_t n;

  // Special member functions
  IndexRange() = default;

  ~IndexRange() = default;

  IndexRange(const IndexRange& A) = default;

  IndexRange& operator=(const IndexRange& A) = default;

  IndexRange(IndexRange&& A) = default;

  IndexRange& operator=(IndexRange&& A) = default;

  /**
   * @brief Construct a new `IndexRange` object
   *
   * @param start
   * Starting index of the `IndexRange` to be constructed.
   * @param n
   * Length of the `IndexRange` to be constructed.
   */
  IndexRange(int64_t start, int64_t n);

  /**
   * @brief Split the `IndexRange` into \p n_splits equal parts
   *
   * @param n_splits
   * Number of splits to split the `IndexRange` into.
   * @return std::vector<IndexRange>
   * Vector containing the `IndexRange`s resulting from the split in order.
   *
   * If the length of the `IndexRange` is not divisible by \p n_splits, the last
   * `IndexRange` returned is shorter than the others and ends on the end of the
   * parent `IndexRange`.
   */
  std::vector<IndexRange> split(int64_t n_splits) const;

  /**
   * @brief Split the `IndexRange` in two at a certain \p index.
   *
   * @param index
   * Index at which to split the `IndexRange`.
   * @return std::vector<IndexRange>
   * Vector containing the `IndexRange`s resulting from the split in order.
   */
  std::vector<IndexRange> split_at(int64_t index) const;

  /**
   * @brief Split the `IndexRange` along a specified axis to match the structure
   * of \p A
   *
   * @param A
   * `Hierarchical` matrix whose immediate substructure along rows or columns is
   * to be matched by the `IndexRange`s resulting from the split.
   * @param along
   * Either `::ALONG_ROW` or `::ALONG_COL`, deciding whether the
   * subranges are to match the structure of \p A along the rows or columns.
   * @return std::vector<IndexRange>
   * Vector containing the `IndexRange`s resulting from the split in order.
   *
   * Note that this function is not recursive. Only the first level of \p A is
   * mimicked in the resulting `IndexRange`s.
   */
  template<typename T>
  std::vector<IndexRange> split_like(const Hierarchical<T>& A, int along) const;
};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_index_range_h
