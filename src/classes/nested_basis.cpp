#include "hicma/classes/nested_basis.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>


namespace hicma
{

NestedBasis::NestedBasis(
  const MatrixProxy& sub_bases, const Dense& trans_mat, bool col_basis
) : sub_bases(share_basis(sub_bases)),
    translation(trans_mat.share()),
    col_basis(col_basis)
{}

bool NestedBasis::is_col_basis() const { return col_basis; }

bool NestedBasis::is_row_basis() const { return !col_basis; }

declare_method(
  bool, is_shared_omm, (virtual_<const Matrix&>, virtual_<const Matrix&>)
)

bool is_shared(const Matrix& A, const Matrix& B) {
  return is_shared_omm(A, B);
}

define_method(bool, is_shared_omm, (const Dense& A, const Dense& B)) {
  return A.is_shared_with(B);
}

define_method(
  bool, is_shared_omm, (const NestedBasis& A, const NestedBasis& B)
) {
  bool shared = A.translation.is_shared_with(B.translation);
  shared &= is_shared(A.sub_bases, B.sub_bases);
  shared &= (A.is_col_basis() == B.is_col_basis());
  return shared;
}

define_method(bool, is_shared_omm, (const NestedBasis&, const Dense&)) {
  return false;
}

define_method(bool, is_shared_omm, (const Dense&, const NestedBasis&)) {
  return false;
}

define_method(
  bool, is_shared_omm, (const Hierarchical& A, const Hierarchical& B)
) {
  assert(A.dim == B.dim);
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      if (!is_shared(A(i, j), B(i, j))) return false;
    }
  }
  return true;
}

define_method(bool, is_shared_omm, (const Matrix& A, const Matrix& B)) {
  omm_error_handler("is_shared", {A, B}, __FILE__, __LINE__);
  std::abort();
}

MatrixProxy share_basis(const Matrix& A) {
  return share_basis_omm(A);
}

define_method(MatrixProxy, share_basis_omm, (const Dense& A)) {
  // TODO Having this work for Dense might not be desirable (see is_shared check
  // above)
  return A.share();
}

define_method(MatrixProxy, share_basis_omm, (const NestedBasis& A)) {
  return NestedBasis(A.sub_bases, A.translation, A.is_col_basis());
}

define_method(MatrixProxy, share_basis_omm, (const Hierarchical& A)) {
  Hierarchical new_shared(A.dim[0], A.dim[1]);
  for (int64_t i=0; i<A.dim[0]; ++i) {
    for (int64_t j=0; j<A.dim[1]; ++j) {
      new_shared(i, j) = share_basis(A(i, j));
    }
  }
  return new_shared;
}

define_method(MatrixProxy, share_basis_omm, (const Matrix& A)) {
  omm_error_handler("share_basis", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
