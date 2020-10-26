#include "hicma/operations/misc.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/pre_scheduler.h"
#include "hicma/operations/arithmetic.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>


namespace hicma
{

declare_method(
  void, recompress_col_omm,
  (virtual_<Matrix&>, virtual_<const Matrix&>, Dense&, const Dense&)
)

void recompress_col(Matrix& AU, const Matrix& BU, Dense& AS, const Dense& BS) {
  recompress_col_omm(AU, BU, AS, BS);
}

define_method(
  void, recompress_col_omm,
  (Dense& AU, const Dense& BU, Dense& AS, const Dense& BS)
) {
  add_recompress_col_task(AU, BU, AS, BS);
}

define_method(
  void, recompress_col_omm,
  (Matrix& AU, const Matrix& BU, Dense&, const Dense&)
) {
  omm_error_handler("recompress_col", {AU, BU}, __FILE__, __LINE__);
  std::abort();
}

declare_method(
  void, recompress_row_omm,
  (virtual_<Matrix&>, virtual_<const Matrix&>, Dense&, const Dense&)
)

void recompress_row(Matrix& AV, const Matrix& BV, Dense& AS, const Dense& BS) {
  recompress_row_omm(AV, BV, AS, BS);
}

define_method(
  void, recompress_row_omm,
  (Dense& AV, const Dense& BV, Dense& AS, const Dense& BS)
) {
  add_recompress_row_task(AV, BV, AS, BS);
}

define_method(
  void, recompress_row_omm,
  (Matrix& AV, const Matrix& BV, Dense&, const Dense&)
) {
  omm_error_handler("recompress_row", {AV, BV}, __FILE__, __LINE__);
  std::abort();
}

declare_method(Dense, get_trans_or_S, (virtual_<const Matrix&>))

define_method(Dense, get_trans_or_S, (const NestedBasis& A)) {
  return A.translation.share();
}

define_method(Dense, get_trans_or_S, (const LowRank& A)) {
  return A.S.share();
}

define_method(Dense, get_trans_or_S, (const Matrix& A)) {
  omm_error_handler("get_trans_or_S", {A}, __FILE__, __LINE__);
  std::abort();
}

std::vector<Dense> gather_trans_mats(const Hierarchical& sub_bases) {
  assert(!(sub_bases.dim[0] > 1 && sub_bases.dim[1] > 1));
  std::vector<Dense> out(sub_bases.dim[0]*sub_bases.dim[1]);
  for (uint64_t i=0; i<out.size(); ++i) {
    out[i] = get_trans_or_S(sub_bases[i]);
  }
  return out;
}

declare_method(
  void, reform_basis_omm, (virtual_<Matrix&>, virtual_<Matrix&>, Dense&)
)

define_method(
  void, reform_basis_omm,
  (NestedBasis& old_basis, NestedBasis& new_basis, Dense& new_trans)
) {
  old_basis.sub_bases = std::move(new_basis.sub_bases);
  old_basis.translation = std::move(new_trans);
}

define_method(
  void, reform_basis_omm,
  (NestedBasis& old_basis, LowRank& new_basis, Dense& new_trans)
) {
  if (old_basis.is_col_basis()) {
    old_basis.sub_bases = std::move(new_basis.U);
  } else {
    old_basis.sub_bases = std::move(new_basis.V);
  }
  old_basis.translation = std::move(new_trans);
}

define_method(
  void, reform_basis_omm,
  (Matrix& new_basis, Matrix& old_basis, Dense& new_trans)
) {
  omm_error_handler(
    "reform_basis_omm", {new_basis, old_basis, new_trans}, __FILE__, __LINE__
  );
  std::abort();
}

void reform_basis(
  Hierarchical& bases, Hierarchical& split, std::vector<Dense>& trans_mats
) {
  assert(!(bases.dim[0] > 1 && bases.dim[1] > 1));
  for (int64_t i=0; i<std::max(bases.dim[0], bases.dim[1]); ++i) {
    reform_basis_omm(bases[i], split[i], trans_mats[i]);
  }
}

void recombine_col(Hierarchical& split, MatrixProxy& U, Dense& S) {
  // Assumes U is actually Hierarchical (should always be the case). This allows
  // us to skip one layer of OMM functions (actually an OMM is used implicitly).
  Hierarchical U_orig(std::move(U));
  std::vector<Dense> trans_orig = gather_trans_mats(U_orig);
  std::vector<Dense> trans_mats = gather_trans_mats(split);
  add_recombine_col_task(trans_orig, S, trans_mats);
  reform_basis(U_orig, split, trans_mats);
  U = std::move(U_orig);
}

void recombine_row(Hierarchical& split, MatrixProxy& V, Dense& S) {
  // Assumes U is actually Hierarchical (should always be the case). This allows
  // us to skip one layer of OMM functions (actually an OMM is used implicitly).
  Hierarchical V_orig(std::move(V));
  std::vector<Dense> trans_orig = gather_trans_mats(V_orig);
  std::vector<Dense> trans_mats = gather_trans_mats(split);
  add_recombine_row_task(trans_orig, S, trans_mats);
  reform_basis(V_orig, split, trans_mats);
  V = std::move(V_orig);
}

} // namespace hicma
