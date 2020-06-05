#include "hicma/classes/intitialization_helpers/basis_copy_tracker.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/shared_basis.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstddef>
#include <functional>
#include <memory>


namespace hicma
{

typedef std::unordered_map<std::shared_ptr<Matrix>, MatrixProxy> BasisMap;
declare_method(
  MatrixProxy, tracked_copy_omm, (virtual_<const Matrix&>, BasisMap&)
)

LowRank BasisCopyTracker::tracked_copy(const LowRank& A) {
  return LowRank(
    tracked_copy_omm(A.U, copied_col_bases),
    A.S,
    tracked_copy_omm(A.V, copied_row_bases),
    true
  );
}

define_method(MatrixProxy, tracked_copy_omm, (const Dense& A, BasisMap&)) {
  return A;
}

define_method(
  MatrixProxy, tracked_copy_omm, (const SharedBasis& A, BasisMap& map)
) {
  if(map.find(A.get_ptr()) == map.end()) {
    map[A.get_ptr()] = A;
  }
  return share_basis(map.at(A.get_ptr()));
}

define_method(
  MatrixProxy, tracked_copy_omm, (const Matrix& A, BasisMap&)
) {
  omm_error_handler("tracked_copy", {A}, __FILE__, __LINE__);
  std::abort();
}

MatrixProxy BasisCopyTracker::tracked_copy(const MatrixProxy& A) {
  return tracked_copy_omm(A, copied_col_bases);
}

declare_method(
  bool, has_basis_omm, (virtual_<const Matrix&>, const BasisMap&)
)

define_method(
  bool, has_basis_omm, (const SharedBasis& A, const BasisMap& map)
) {
  return map.find(A.get_ptr()) != map.end();
}

define_method(bool, has_basis_omm, (const Matrix&, const BasisMap&)) {
  // TODO Might need to find a way to track regular Dense along with SharedBasis
  // for some cases. Example trsm, where we use this to check whether a trsm was
  // already applied. If we don't check for shared Dense, trsm might get applied
  // multiple times
  return false;
}

bool BasisCopyTracker::has_col_basis(const Matrix& A) const {
  return has_basis_omm(A, copied_col_bases);
}

bool BasisCopyTracker::has_row_basis(const Matrix& A) const {
  return has_basis_omm(A, copied_row_bases);
}

declare_method(
  void, register_basis_omm, (virtual_<const Matrix&>, BasisMap&)
)

define_method(
  void, register_basis_omm, (const SharedBasis& A, BasisMap& map)
) {
  map[A.get_ptr()] = MatrixProxy();
}

define_method(void, register_basis_omm, (const Matrix&, BasisMap&)) {
  // TODO Might need to find a way to track regular Dense along with SharedBasis
  // for some cases. Example trsm, where we use this to check whether a trsm was
  // already applied. If we don't check for shared Dense, trsm might get applied
  // multiple times.
  // For now: Do nothing
}

void BasisCopyTracker::register_col_basis(const Matrix& A) {
  register_basis_omm(A, copied_col_bases);
}

void BasisCopyTracker::register_row_basis(const Matrix& A) {
  register_basis_omm(A, copied_row_bases);
}

} // namespace hicma
