#ifndef hicma_extension_headers_operations_h
#define hicma_extension_headers_operations_h

#include "hicma/classes/dense.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/extension_headers/tuple_types.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

// Arithmetic
declare_method(
  Matrix&, addition_omm,
  (virtual_<Matrix&>, virtual_<const Matrix&>)
)

declare_method(
  MatrixProxy, addition_omm,
  (virtual_<const Matrix&>, virtual_<const Matrix&>)
)

declare_method(
  MatrixProxy, subtraction_omm,
  (virtual_<const Matrix&>, virtual_<const Matrix&>)
)

declare_method(
  Matrix&, multiplication_omm,
  (virtual_<Matrix&>, double)
)

// BLAS
declare_method(
  void, gemm_omm,
  (
    virtual_<const Matrix&>, virtual_<const Matrix&>, virtual_<Matrix&>,
    double, double
  )
)

declare_method(
  Dense, gemm_omm,
  (
    virtual_<const Matrix&>, virtual_<const Matrix&>,
    double, bool, bool
  )
)

declare_method(
  void, trmm_omm,
  (
    virtual_<const Matrix&>, virtual_<Matrix&>,
    const char&, const char&, const char&, const char&,
    double
  )
)

declare_method(
  void, trsm_omm,
  (virtual_<const Matrix&>, virtual_<Matrix&>, int, int)
)

// LAPACK
declare_method(DenseIndexSetPair, geqp3_omm, (virtual_<Matrix&>))

declare_method(
  void, geqrt_omm,
  (virtual_<Matrix&>, virtual_<Matrix&>)
)

declare_method(
  MatrixPair, getrf_omm,
  (virtual_<Matrix&>)
)

declare_method(
  DenseIndexSetPair, one_sided_id_omm, (virtual_<Matrix&>, int64_t)
)

declare_method(DenseTriplet, id_omm, (virtual_<Matrix&>, int64_t))

declare_method(
  void, larfb_omm,
  (virtual_<const Matrix&>, virtual_<const Matrix&>, virtual_<Matrix&>, bool)
)

declare_method(
  void, qr_omm,
  (virtual_<Matrix&>, virtual_<Matrix&>, virtual_<Matrix&>)
)
declare_method(
  DensePair, make_left_orthogonal_omm,
  (virtual_<const Matrix&>)
)
declare_method(
  void, update_splitted_size_omm,
  (virtual_<const Matrix&>, int64_t&, int64_t&)
)
declare_method(
  MatrixProxy, split_by_column_omm,
  (virtual_<const Matrix&>, virtual_<Matrix&>, int64_t&)
)
declare_method(
  MatrixProxy, concat_columns_omm,
  (
    virtual_<const Matrix&>, virtual_<const Matrix&>, virtual_<const Matrix&>,
    int64_t&
  )
)
declare_method(
  void, orthogonalize_block_col_omm,
  (int64_t, virtual_<const Matrix&>, virtual_<Matrix&>, virtual_<Matrix&>)
)

declare_method(void, zero_lowtri_omm, (virtual_<Matrix&>))
declare_method(void, zero_whole_omm, (virtual_<Matrix&>))

declare_method(
  void, rq_omm,
  (virtual_<Matrix&>, virtual_<Matrix&>, virtual_<Matrix&>)
)

declare_method(
  void, tpmqrt_omm,
  (
    virtual_<const Matrix&>, virtual_<const Matrix&>,
    virtual_<Matrix&>, virtual_<Matrix&>,
    bool
  )
)

declare_method(
  void, tpqrt_omm,
  (virtual_<Matrix&>, virtual_<Matrix&>, virtual_<Matrix&>)
)

declare_method(int64_t, get_n_rows_omm, (virtual_<const Matrix&>))

declare_method(int64_t, get_n_cols_omm, (virtual_<const Matrix&>))

declare_method(double, norm_omm, (virtual_<const Matrix&>))

declare_method(void, transpose_omm, (virtual_<Matrix&>))

} // namespace hicma

#endif // hicma_extension_headers_operations_h
