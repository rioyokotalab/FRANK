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


/**
 * @brief Namespace of the C++ standard library
 *
 * Some template specializations are added to utilize standard library
 * containers with HiCMA custom classes.
 */
namespace std {
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


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class ClusterTree;
class Dense;

bool matrix_is_tracked(std::string tracker, const Dense& A);

void register_matrix(
  std::string tracker, const Dense& A, Dense&& content=Dense()
);

Dense& get_tracked_content(std::string tracker, const Dense& A);

bool matrix_is_tracked(std::string tracker, const Dense& A, const Dense& B);

void register_matrix(
  std::string tracker, const Dense& A, const Dense& B, Dense&& content=Dense()
);

Dense& get_tracked_content(std::string tracker, const Dense& A, const Dense& B);

void clear_tracker(std::string tracker);

void clear_trackers();

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
