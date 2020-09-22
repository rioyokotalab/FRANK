#include "hicma/operations/misc.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/util/omm_error_handler.h"
#include "hicma/util/pre_scheduler.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <utility>


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
  (NestedBasis& AU, const NestedBasis& BU, Dense& AS, const Dense& BS)
) {
  recompress_col(AU.transfer_matrix, BU.transfer_matrix, AS, BS);
}

define_method(
  void, recompress_col_omm,
  (NestedBasis& AU, const Dense& BU, Dense& AS, const Dense& BS)
) {
  recompress_col(AU.transfer_matrix, BU, AS, BS);
}

define_method(
  void, recompress_col_omm,
  (Dense& AU, const NestedBasis& BU, Dense& AS, const Dense& BS)
) {
  recompress_col(AU, BU.transfer_matrix, AS, BS);
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
  omm_error_handler("recompress", {AU, BU}, __FILE__, __LINE__);
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
  (NestedBasis& AV, const NestedBasis& BV, Dense& AS, const Dense& BS)
) {
  recompress_row(AV.transfer_matrix, BV.transfer_matrix, AS, BS);
}

define_method(
  void, recompress_row_omm,
  (NestedBasis& AV, const Dense& BV, Dense& AS, const Dense& BS)
) {
  recompress_row(AV.transfer_matrix, BV, AS, BS);
}

define_method(
  void, recompress_row_omm,
  (Dense& AV, const NestedBasis& BV, Dense& AS, const Dense& BS)
) {
  recompress_row(AV, BV.transfer_matrix, AS, BS);
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
  omm_error_handler("recompress", {AV, BV}, __FILE__, __LINE__);
}

} // namespace hicma
