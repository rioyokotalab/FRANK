#include "hicma/classes/intitialization_helpers/basis_copy_tracker.h"
#include "hicma/extension_headers/classes.h"

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
    tracked_copy_omm(A._U, copied_col_bases),
    A.S(),
    tracked_copy_omm(A._V, copied_row_bases),
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

define_method(MatrixProxy, share_basis, (const SharedBasis& A)) {
  return A.share();
}

define_method(MatrixProxy, share_basis, (const Dense& A)) {
  return Dense(A, A.dim[0], A.dim[1], 0, 0);
}

define_method(MatrixProxy, share_basis, (const Matrix& A)) {
  omm_error_handler("share_basis", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
