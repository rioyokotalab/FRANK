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
  Dense newU(AU.dim[0], AU.dim[1]);
  Dense newS(AS.dim[0], AS.dim[1]);
  add_recompress_col_task(newU, newS, AU, BU, AS, BS);
  AU = std::move(newU);
  AS = std::move(newS);
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
  Dense newV(AV.dim[0], AV.dim[1]);
  Dense newS(AS.dim[0], AS.dim[1]);
  add_recompress_row_task(newV, newS, AV, BV, AS, BS);
  AV = std::move(newV);
  AS = std::move(newS);
}

define_method(
  void, recompress_row_omm,
  (Matrix& AV, const Matrix& BV, Dense&, const Dense&)
) {
  omm_error_handler("recompress_row", {AV, BV}, __FILE__, __LINE__);
  std::abort();
}

declare_method(
  Dense, add_recombine_col,
  (virtual_<const Matrix&>, virtual_<Matrix&>, const Dense&)
)

define_method(
  Dense, add_recombine_col,
  (const NestedBasis& split, NestedBasis& basis_orig, const Dense& S_orig)
) {
  Dense new_trans, new_S;
  std::tie(new_trans, new_S) = add_recombine_col_task(
    basis_orig.translation, S_orig, split.translation
  );
  basis_orig = NestedBasis(split.sub_bases, new_trans, true);
  return new_S;
}

define_method(
  Dense, add_recombine_col,
  (const LowRank& split, NestedBasis& basis_orig, const Dense& S_orig)
) {
  Dense new_trans, new_S;
  std::tie(new_trans, new_S) = add_recombine_col_task(
    basis_orig.translation, S_orig, split.S
  );
  basis_orig = NestedBasis(split.U, new_trans, true);
  return new_S;
}

define_method(
  Dense, add_recombine_col,
  (const Matrix& split, Matrix& original, const Dense&)
) {
  omm_error_handler("add_recombine_col", {split, original}, __FILE__, __LINE__);
  std::abort();
}

void recombine_col(Hierarchical& split, MatrixProxy& U, Dense& S) {
  // Assumes U is actually Hierarchical (should always be the case). This allows
  // us to skip one layer of OMM functions (actually an OMM is used implicitly).
  Hierarchical U_orig(std::move(U));
  Dense new_S;
  for (int64_t i=0; i<split.dim[0]; ++i) {
    Dense block_S = add_recombine_col(split[i], U_orig[i], S);
    if (i == 0) {
      new_S = std::move(block_S);
    } else {
      assert(is_shared(block_S, new_S));
    }
  }
  U = std::move(U_orig);
  S = std::move(new_S);
}

declare_method(
  Dense, add_recombine_row,
  (virtual_<const Matrix&>, virtual_<Matrix&>, const Dense&)
)

define_method(
  Dense, add_recombine_row,
  (const NestedBasis& split, NestedBasis& basis_orig, const Dense& S_orig)
) {
  Dense new_trans, new_S;
  std::tie(new_trans, new_S) = add_recombine_row_task(
    basis_orig.translation, S_orig, split.translation
  );
  basis_orig = NestedBasis(split.sub_bases, new_trans, false);
  return new_S;
}

define_method(
  Dense, add_recombine_row,
  (const LowRank& split, NestedBasis& basis_orig, const Dense& S_orig)
) {
  Dense new_trans, new_S;
  std::tie(new_trans, new_S) = add_recombine_row_task(
    basis_orig.translation, S_orig, split.S
  );
  basis_orig = NestedBasis(split.V, new_trans, false);
  return new_S;
}

define_method(
  Dense, add_recombine_row,
  (const Matrix& split, Matrix& original, const Dense&)
) {
  omm_error_handler("add_recombine_row", {split, original}, __FILE__, __LINE__);
  std::abort();
}

void recombine_row(Hierarchical& split, MatrixProxy& V, Dense& S) {
  // Assumes U is actually Hierarchical (should always be the case). This allows
  // us to skip one layer of OMM functions (actually an OMM is used implicitly).
  Hierarchical V_orig(std::move(V));
  Dense new_S;
  for (int64_t j=0; j<split.dim[1]; ++j) {
    Dense block_ = add_recombine_row(split[j], V_orig[j], S);
    if (j == 0) {
      new_S = std::move(block_);
    } else {
      assert(is_shared(block_, new_S));
    }
  }
  V = std::move(V_orig);
  S = std::move(new_S);
}

} // namespace hicma
