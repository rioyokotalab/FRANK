#include "hicma/classes/nested_basis.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>


namespace hicma
{

SharedBasis::SharedBasis(const SharedBasis& A)
: Matrix(A), transfer_matrix(std::make_shared<Dense>(*A.transfer_matrix)),
  sub_bases(A.sub_bases), col_basis(A.col_basis)
{}

SharedBasis& SharedBasis::operator=(const SharedBasis& A) {
  Matrix::operator=(A);
  transfer_matrix = std::make_shared<Dense>(*A.transfer_matrix);
  sub_bases = A.sub_bases;
  col_basis = A.col_basis;
  return *this;
}

SharedBasis::SharedBasis(
  Dense&& A, std::vector<MatrixProxy>& sub_bases,bool is_col_basis
) : transfer_matrix(std::make_shared<Dense>(std::move(A))),
    sub_bases(std::move(sub_bases)), col_basis(is_col_basis)
{}

MatrixProxy& SharedBasis::operator[](int64_t i) {
  return sub_bases[i];
}

const MatrixProxy& SharedBasis::operator[](int64_t i) const {
  return sub_bases[i];
}

int64_t SharedBasis::num_child_basis() const { return sub_bases.size(); }

SharedBasis SharedBasis::share() const {
  SharedBasis new_shared;
  new_shared.transfer_matrix = transfer_matrix;
  new_shared.sub_bases = std::vector<MatrixProxy>(num_child_basis());
  for (int64_t i=0; i<new_shared.num_child_basis(); ++i) {
    new_shared[i] = share_basis((*this)[i]);
  }
  new_shared.col_basis = col_basis;
  return new_shared;
}

bool SharedBasis::is_shared(const SharedBasis& A) const {
  bool shared = transfer_matrix == A.transfer_matrix;
  shared &= num_child_basis() == A.num_child_basis();
  if (shared) {
    for (int64_t i=0; i<num_child_basis(); ++i) {
      shared &= hicma::is_shared((*this)[i], A[i]);
    }
  }
  return shared;
}

Dense& SharedBasis::transfer_mat() { return *transfer_matrix; }

const Dense& SharedBasis::transfer_mat() const { return *transfer_matrix; }

bool SharedBasis::is_col_basis() const { return col_basis; }

bool SharedBasis::is_row_basis() const { return !col_basis; }

declare_method(
  bool, is_shared_omm, (virtual_<const Matrix&>, virtual_<const Matrix&>)
)

bool is_shared(const Matrix& A, const Matrix& B) {
  return is_shared_omm(A, B);
}

define_method(
  bool, is_shared_omm, (const SharedBasis& A, const SharedBasis& B)
) {
  return A.is_shared(B);
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
  // TODO Having this work for Dense might not be desirable (see is_shared check
  // above)
  return Dense(A, A.dim[0], A.dim[1], 0, 0);
}

define_method(MatrixProxy, share_basis_omm, (const Matrix& A)) {
  omm_error_handler("share_basis", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
