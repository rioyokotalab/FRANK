#include "hicma/classes/no_copy_split.h"
#include "hicma/extension_headers/classes.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/shared_basis.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"
#include "hicma/operations/misc.h"
#include "hicma/util/omm_error_handler.h"

#include "yorel/yomm2/cute.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>


namespace hicma
{

NoCopySplit::NoCopySplit(
  const Matrix& A, int64_t n_row_blocks, int64_t n_col_blocks
) : Hierarchical(n_row_blocks, n_col_blocks) {
  ClusterTree node(get_n_rows(A), get_n_cols(A), dim[0], dim[1]);
  for (ClusterTree& child_node : node) {
    (*this)[child_node] = get_part(A, child_node);
  }
}

NoCopySplit::NoCopySplit(const Matrix& A, const Hierarchical& like)
: Hierarchical(like.dim[0], like.dim[1]) {
  assert(get_n_rows(A) == get_n_rows(like));
  assert(get_n_cols(A) == get_n_cols(like));
  ClusterTree node(like);
  for (ClusterTree& child_node : node) {
    (*this)[child_node] = get_part(A, child_node);
  }
}

define_method(
  MatrixProxy, get_part_omm,
  (
    const Dense& A,
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    bool copy
  )
) {
  return Dense(A, n_rows, n_cols, row_start, col_start, copy);
}

define_method(
  MatrixProxy, get_part_omm,
  (
    const SharedBasis& A,
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    bool copy
  )
) {
  return get_part(*A.get_ptr(), n_rows, n_cols, row_start, col_start, copy);
}

define_method(
  MatrixProxy, get_part_omm,
  (
    const LowRank& A,
    int64_t n_rows, int64_t n_cols, int64_t row_start, int64_t col_start,
    bool copy
  )
) {
  return LowRank(A, n_rows, n_cols, row_start, col_start, copy);
}

define_method(
  MatrixProxy, get_part_omm,
  (
    const Matrix& A,
    [[maybe_unused]] int64_t n_rows, [[maybe_unused]] int64_t n_cols,
    [[maybe_unused]] int64_t row_start, [[maybe_unused]] int64_t col_start,
    [[maybe_unused]] bool copy
  )
) {
  omm_error_handler("get_part", {A}, __FILE__, __LINE__);
  std::abort();
}

} // namespace hicma
