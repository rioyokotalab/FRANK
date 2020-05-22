#include "hicma/classes/intitialization_helpers/basis_copy_tracker.h"

#include "hicma/classes/basis.h"
#include "hicma/classes/low_rank.h"

#include <memory>


namespace hicma
{

std::shared_ptr<Dense> BasisCopyTracker::copy_col_basis(const LowRank& A) {
  if(copied_col_bases.find(A._U) == copied_col_bases.end()) {
    copied_col_bases[A._U] = std::make_shared<Dense>(*A._U);
  }
  return copied_col_bases.at(A._U);
}

std::shared_ptr<Dense> BasisCopyTracker::copy_row_basis(const LowRank& A) {
  if(copied_row_bases.find(A._V) == copied_row_bases.end()) {
    copied_row_bases[A._V] = std::make_shared<Dense>(*A._V);
  }
  return copied_row_bases.at(A._V);
}

} // namespace hicma
