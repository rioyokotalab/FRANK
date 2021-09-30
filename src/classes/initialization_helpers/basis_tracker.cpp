#include "hicma/classes/initialization_helpers/basis_tracker.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <functional>
#include <unordered_map>
#include <utility>


namespace std
{

size_t hash<hicma::IndexRange>::operator()(const hicma::IndexRange& key) const {
  return (hash<int64_t>()(key.start) ^ (hash<int64_t>()(key.n) << 1));
}

} // namespace std


namespace hicma
{

bool operator==(const IndexRange& A, const IndexRange& B) {
  return (A.start == B.start) && (A.n == B.n);
}

std::unordered_map<std::string, BasisTracker<uint64_t>> single_trackers;

bool matrix_is_tracked(std::string tracker, const Dense& A) {
  if (single_trackers.find(tracker) == single_trackers.end()) {
    return false;
  } else {
    return single_trackers[tracker].has_key(A.id());
  }
}

void register_matrix(
  std::string tracker, const Dense& A, Dense&& content
) {
  if (single_trackers.find(tracker) == single_trackers.end()) {
    single_trackers[tracker] = BasisTracker<uint64_t>();
  }
  single_trackers[tracker][A.id()] = std::move(content);
}

Dense& get_tracked_content(std::string tracker, const Dense& A) {
  return single_trackers[tracker][A.id()];
}

std::unordered_map<
  std::string, BasisTracker<uint64_t, BasisTracker<uint64_t>>
> double_trackers;

bool matrix_is_tracked(
  std::string tracker, const Dense& A, const Dense& B
) {
  if (double_trackers.find(tracker) == double_trackers.end()) {
    return false;
  } else {
    if (!double_trackers[tracker].has_key(A.id())) return false;
    if (!double_trackers[tracker][A.id()].has_key(B.id())) return false;
    return true;
  }
}

void register_matrix(
  std::string tracker, const Dense& A, const Dense& B, Dense&& content
) {
  if (double_trackers.find(tracker) == double_trackers.end()) {
    double_trackers[tracker] = BasisTracker<uint64_t, BasisTracker<uint64_t>>();
  }
  double_trackers[tracker][A.id()][B.id()] = std::move(content);
}

Dense& get_tracked_content(
  std::string tracker, const Dense& A, const Dense& B
) {
  return double_trackers[tracker][A.id()][B.id()];
}

void clear_tracker(std::string tracker) { single_trackers.erase(tracker); }

void clear_trackers() {
  single_trackers.clear();
  double_trackers.clear();
}

} // namespace hicma
