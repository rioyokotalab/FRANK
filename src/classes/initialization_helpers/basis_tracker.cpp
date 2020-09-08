#include "hicma/classes/initialization_helpers/basis_tracker.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/timer.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>


namespace std
{

size_t hash<hicma::BasisKey>::operator()(const hicma::BasisKey& key) const {
  return (
    ((
      hash<const double*>()(key.data_ptr)
      ^ (hash<int64_t>()(key.dim[0]) << 1)
    ) >> 1)
    ^ (hash<int64_t>()(key.dim[1]) << 1)
  );
}

size_t hash<hicma::IndexRange>::operator()(const hicma::IndexRange& key) const {
  return (hash<int64_t>()(key.start) ^ (hash<int64_t>()(key.n) << 1));
}

bool less<hicma::IndexRange>::operator()(
  const hicma::IndexRange& a, const hicma::IndexRange& b
) const {
  // No empty ranges
  assert(a.n != 0);
  assert(b.n != 0);
  // No half-overlapping ranges
  assert(!((a.start < b.start) && (a.start+a.n > b.start)));
  bool out = false;
  // a is earlier separate range
  out |= (a.start + a.n <= b.start);
  // Same start, but b is smaller part of a
  out |= ((a.start == b.start) && (a.n > b.n));
  // b starts later, but is smaller part of a
  out |= ((a.start < b.start) && (a.start + a.n >= b.start + b.n));
  return out;
}

} // namespace std


namespace hicma
{

BasisKey::BasisKey(const BasisKey& A)
: data_ptr(A.data_ptr), D(share_basis(A.D)), dim(A.dim) {}

BasisKey& BasisKey::operator=(const BasisKey& A) {
  data_ptr = A.data_ptr;
  D = share_basis(A.D);
  dim = A.dim;
  return *this;
}

BasisKey::BasisKey(const Dense& A)
: data_ptr(&A), D(A.share()), dim(A.dim) {}

bool operator==(const BasisKey& A, const BasisKey& B) {
  return (A.data_ptr == B.data_ptr) && (A.dim == B.dim);
}

bool operator==(const IndexRange& A, const IndexRange& B) {
  return (A.start == B.start) && (A.n == B.n);
}

std::map<std::string, BasisTracker<BasisKey>> single_trackers;

bool matrix_is_tracked(std::string tracker, const Dense& key) {
  if (single_trackers.find(tracker) == single_trackers.end()) {
    return false;
  } else {
    return single_trackers[tracker].has_key(key);
  }
}

void register_matrix(
  std::string tracker, const Dense& key, Dense&& content
) {
  if (single_trackers.find(tracker) == single_trackers.end()) {
    single_trackers[tracker] = BasisTracker<BasisKey>();
  }
  single_trackers[tracker][key] = std::move(content);
}

Dense& get_tracked_content(std::string tracker, const Dense& key) {
  return single_trackers[tracker][key];
}

std::map<
  std::string, BasisTracker<BasisKey, BasisTracker<BasisKey>>
> double_trackers;

bool matrix_is_tracked(
  std::string tracker, const Dense& key1, const Dense& key2
) {
  if (double_trackers.find(tracker) == double_trackers.end()) {
    return false;
  } else {
    if (!double_trackers[tracker].has_key(key1)) return false;
    if (!double_trackers[tracker][key1].has_key(key2)) return false;
    return true;
  }
}

void register_matrix(
  std::string tracker, const Dense& key1, const Dense& key2, Dense&& content
) {
  if (double_trackers.find(tracker) == double_trackers.end()) {
    double_trackers[tracker] = BasisTracker<BasisKey, BasisTracker<BasisKey>>();
  }
  double_trackers[tracker][key1][key2] = std::move(content);
}

Dense& get_tracked_content(
  std::string tracker, const Dense& key1, const Dense& key2
) {
  return double_trackers[tracker][key1][key2];
}

void clear_tracker(std::string tracker) { single_trackers.erase(tracker); }

void clear_trackers() {
  single_trackers.clear();
  double_trackers.clear();
}

BasisTracker<BasisKey, BasisTracker<BasisKey>> concatenated_bases;

NestedTracker::NestedTracker(const IndexRange& index_range)
: index_range(index_range) {}

void NestedTracker::register_range(
  const IndexRange& main_range, const IndexRange& associated_range
) {
  // NOTE This function assumes that there are now overlapping ranges!
  for (NestedTracker& child : children) {
    if (child.is_exactly(main_range)) {
      child.add_associated_range(associated_range);
      return;
    } else if (child.contains(main_range)) {
      child.register_range(main_range, associated_range);
      return;
    }
  }
  // Following part is only reached if main_range was not contained in any
  // children. Thus, main_range is added as a new child
  NestedTracker new_child(main_range);
  new_child.add_associated_range(associated_range);
  // Also add associated range of parent to new child
  for (const IndexRange& parent_range : associated_ranges) {
    new_child.add_associated_range(parent_range);
  }
  // Find previously registered ranges that are subrange of the new range added
  // here and move them to the new range
  for (decltype(children)::iterator it=children.begin(); it !=children.end();) {
    if (new_child.contains((*it).index_range)) {
      NestedTracker child_of_new = std::move(*it);
      // Add associated range of new parent
      child_of_new.add_associated_range(associated_range);
      new_child.children.push_back(std::move(child_of_new));
      it = children.erase(it);
    } else {
      ++it;
    }
  }
  children.push_back(new_child);
  // Keep the children sorted in order of start index
  sort_children();
}

void NestedTracker::add_associated_range(const IndexRange& associated_range) {
  // We need index ranges in order. std::set::insert does this
  associated_ranges.insert(associated_range);
  // If the parent block is admissible, this is also true for subblocks. Thus,
  // children also need this range.
  for (NestedTracker& child : children) {
    child.add_associated_range(associated_range);
  }
}

bool NestedTracker::contains(const IndexRange& range) const {
  bool compatible_start = index_range.start <= range.start;
  bool compatible_end = (
    (range.start + range.n) <= (index_range.start + index_range.n)
  );
  return compatible_start && compatible_end;
}

bool NestedTracker::is_exactly(const IndexRange& range) const {
  return (index_range.start == range.start) && (index_range.n == range.n);
}

void NestedTracker::complete_index_range() {
  assert(!children.empty());
  std::vector<NestedTracker> to_add;
  int64_t previous_range_end = index_range.start;
  for (uint64_t i=0; i<children.size(); ++i) {
    const IndexRange& child_range = children[i].index_range;
    if (child_range.start > previous_range_end) {
      // NOTE There should be no missing index ranges!
      abort();
      NestedTracker new_child(IndexRange(
        previous_range_end, child_range.start - previous_range_end
      ));
      // Use same associated range as parent
      new_child.associated_ranges = associated_ranges;
      to_add.push_back(std::move(new_child));
    }
    previous_range_end = child_range.start + child_range.n;
  }
  // Check final range
  if (previous_range_end < index_range.start + index_range.n) {
    NestedTracker last_child(IndexRange(
      previous_range_end, index_range.start + index_range.n - previous_range_end
    ));
    // Use same associated range as parent
    last_child.associated_ranges = associated_ranges;
    to_add.push_back(std::move(last_child));
  }
  // Add new children to end of children vector
  children.insert(children.end(), to_add.begin(), to_add.end());
  // Keep the children sorted in order of start index
  sort_children();
}

void NestedTracker::sort_children() {
  std::sort(
    children.begin(), children.end(),
    [](const NestedTracker& a, const NestedTracker& b) {
      return a.index_range.start < b.index_range.start;
    }
  );
}

} // namespace hicma
