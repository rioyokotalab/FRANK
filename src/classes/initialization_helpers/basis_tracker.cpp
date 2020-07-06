#include "hicma/classes/intitialization_helpers/basis_tracker.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/classes/intitialization_helpers/index_range.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

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
  init_basis_key(key, *A.get_ptr());
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

} // namespace hicma
