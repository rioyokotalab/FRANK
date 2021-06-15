#ifndef hicma_classes_initialization_helpers_basis_tracker_h
#define hicma_classes_initialization_helpers_basis_tracker_h

#include "hicma/classes/matrix.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <tuple>
#include <vector>


namespace hicma
{

class Dense;

class BasisKey {
 public:
  const double* data_ptr;
  // Hold a shared version of the origin of the key. This way we can avoid the
  // memory from being released as long as the key is in use. Important since we
  // use the pointer to hash and compare the keys.
  Dense D;
  std::array<int64_t, 2> dim;

  // Special member functions
  BasisKey() = default;

  virtual ~BasisKey() = default;

  BasisKey(const BasisKey& A);

  BasisKey& operator=(const BasisKey& A);

  BasisKey(BasisKey&& A) = default;

  BasisKey& operator=(BasisKey&& A) = default;

  BasisKey(const Dense&);
};

bool matrix_is_tracked(std::string tracker, const Dense& key);

void register_matrix(
  std::string tracker, const Dense& key, Dense&& content=Dense()
);

Dense& get_tracked_content(std::string tracker, const Dense& key);

bool matrix_is_tracked(
  std::string tracker, const Dense& key1, const Dense& key2
);

void register_matrix(
  std::string tracker,
  const Dense& key1, const Dense& key2,
  Dense&& content=Dense()
);

Dense& get_tracked_content(
  std::string tracker, const Dense& key1, const Dense& key2
);

void clear_tracker(std::string tracker);

void clear_trackers();

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

template<class Key, class Content = Dense>
class BasisTracker {
 private:
  std::unordered_map<Key, Content> map;
 public:
  bool has_key(const Key& key) const { return map.find(key) != map.end(); }

  const Content& operator[](const Key& key) const { return map[key]; }

  Content& operator[](const Key& key) { return map[key]; }

  void clear() { map.clear(); }
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
