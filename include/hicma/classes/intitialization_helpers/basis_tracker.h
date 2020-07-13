#ifndef hicma_classes_initialization_helpers_basis_tracker_h
#define hicma_classes_initialization_helpers_basis_tracker_h

#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/classes/intitialization_helpers/index_range.h"

#include <cstdint>
#include <functional>
#include <set>
#include <tuple>
#include <vector>


namespace hicma
{

class BasisKey {
 public:
  const double* data_ptr;
  std::array<int64_t, 2> dim;

  BasisKey(const Matrix&);

  BasisKey(const MatrixProxy&);
};

} // namespace hicma


namespace std {
  template <>
  struct hash<hicma::BasisKey> {
    size_t operator()(const hicma::BasisKey& key) const;
  };

  template <>
  struct hash<hicma::IndexRange> {
    size_t operator()(const hicma::IndexRange& key) const;
  };

  template <>
  struct less<hicma::IndexRange> {
    bool operator()(
      const hicma::IndexRange& a, const hicma::IndexRange& b
    ) const;
  };
}

namespace hicma
{

class ClusterTree;

bool operator==(const BasisKey& A, const BasisKey& B);

bool operator==(const IndexRange& A, const IndexRange& B);

template<class T>
class BasisTracker {
 private:
  std::unordered_map<T, MatrixProxy> bases;
 public:
  bool has_basis(const T& key) const {
    return (bases.find(key) != bases.end());
  }

  const MatrixProxy& operator[](const T& key) const {
    return bases[key];
  }

  MatrixProxy& operator[](const T& key) {
    return bases[key];
  }
};

class NestedTracker {
 public:
  std::vector<NestedTracker> children;
  IndexRange index_range;
  std::set<IndexRange> associated_ranges;

  // Special member functions
  NestedTracker() = default;

  virtual ~NestedTracker() = default;

  NestedTracker(const NestedTracker& A) = default;

  NestedTracker& operator=(const NestedTracker& A) = default;

  NestedTracker(NestedTracker&& A) = default;

  NestedTracker& operator=(NestedTracker&& A) = default;

  // Additional constructors
  NestedTracker(const IndexRange& index_range);

  // Utility methods
  void register_range(
    const IndexRange& main_range, const IndexRange& associated_range
  );

  void add_associated_range(const IndexRange& associated_range);

  bool contains(const IndexRange& range) const;

  bool is_exactly(const IndexRange& range) const;

  void complete_index_range();

 private:
  void sort_children();
};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_basis_tracker_h
