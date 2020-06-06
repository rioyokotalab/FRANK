#include "hicma/classes/intitialization_helpers/basis_tracker.h"

#include "hicma/classes/shared_basis.h"
#include "hicma/classes/intitialization_helpers/index_range.h"

#include <cstdint>
#include <functional>


namespace std
{

size_t hash<hicma::IndexRange>::operator()(const hicma::IndexRange& k) const {
  return (hash<int64_t>()(k.start) ^ (hash<int64_t>()(k.n) << 1));
}

} // namespace std


namespace hicma
{

bool operator==(const IndexRange& A, const IndexRange& B) {
  return (A.start == B.start) && (A.n == B.n);
}

} // namespace hicma
