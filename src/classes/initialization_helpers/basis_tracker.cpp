#include "hicma/classes/intitialization_helpers/basis_tracker.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/classes/intitialization_helpers/index_range.h"
#include "hicma/util/omm_error_handler.h"

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
  return (a.n <= b.n) && (a.start < b.start);
}

} // namespace std


namespace hicma
{

declare_method(void, init_basis_key, (BasisKey&, virtual_<const Matrix&>))

BasisKey::BasisKey(const MatrixProxy& A) {
  init_basis_key(*this, A);
}

define_method(void, init_basis_key, (BasisKey& key, const Dense& A)) {
  key.data_ptr = &A;
  key.dim = A.dim;
}

define_method(void, init_basis_key, (BasisKey& key, const SharedBasis& A)) {
  init_basis_key(key, A.transfer_mat());
}

define_method(void, init_basis_key, (BasisKey&, const Matrix& A)) {
  omm_error_handler("init_basis_key", {A}, __FILE__, __LINE__);
  std::abort();
}

bool operator==(const BasisKey& A, const BasisKey& B) {
  // return true;
  return (A.data_ptr == B.data_ptr) && (A.dim == B.dim);
}

bool operator==(const IndexRange& A, const IndexRange& B) {
  return (A.start == B.start) && (A.n == B.n);
}


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
