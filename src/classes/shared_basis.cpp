#include "hicma/classes/shared_basis.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <memory>


namespace hicma
{

SharedBasis::SharedBasis(const SharedBasis& A)
: Matrix(A), representation(std::make_shared<Dense>(*A.representation)) {}

SharedBasis& SharedBasis::operator=(const SharedBasis& A) {
  Matrix::operator=(A);
  representation = std::make_shared<Dense>(*A.representation);
  return *this;
}

SharedBasis::SharedBasis(Dense&& A)
: representation(std::make_shared<Dense>(std::move(A))) {}

SharedBasis::SharedBasis(std::shared_ptr<Dense> A) : representation(A) {}

SharedBasis SharedBasis::share() const { return SharedBasis(representation); }

std::shared_ptr<Dense> SharedBasis::get_ptr() const { return representation; }

declare_method(
  bool, is_shared_omm, (virtual_<const Matrix&>, virtual_<const Matrix&>)
)

bool is_shared(const Matrix& A, const Matrix& B) {
  return is_shared_omm(A, B);
}

define_method(
  bool, is_shared_omm, (const SharedBasis& A, const SharedBasis& B)
) {
  return A.get_ptr() == B.get_ptr();
}

define_method(bool, is_shared_omm, (const Dense&, const Dense&)) {
  // TODO Might need to find a way to check for regular Dense as well. In LR
  // addition, this could potentiall save a lot of time. For now though, such
  // cases should not happen.
  return false;
}

define_method(bool, is_shared_omm, (const Matrix& A, const Matrix& B)) {
  omm_error_handler("is_shared", {A, B}, __FILE__, __LINE__);
  std::abort();
}

MatrixProxy share_basis(const Matrix& A) {
  return share_basis_omm(A);
}

define_method(MatrixProxy, share_basis_omm, (const SharedBasis& A)) {
  return A.share();
}

define_method(MatrixProxy, share_basis_omm, (const Dense& A)) {
  return Dense(A, A.dim[0], A.dim[1], 0, 0);
}

define_method(MatrixProxy, share_basis_omm, (const Matrix& A)) {
  omm_error_handler("share_basis", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
