#ifndef hicma_extension_headers_classes_h
#define hicma_extension_headers_classes_h

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <memory>


namespace hicma
{

declare_method(std::unique_ptr<Matrix>, clone, (virtual_<const Matrix&>))

declare_method(std::unique_ptr<Matrix>, move_clone, (virtual_<Matrix&&>))

declare_method(MatrixProxy, shallow_copy_omm, (virtual_<const Matrix&>))

declare_method(
  MatrixProxy, split_omm,
  (
    virtual_<const Matrix&>,
    const std::vector<IndexRange>&, const std::vector<IndexRange>&,
    bool
  )
)

// TODO overload of OMM function to allow for default arguments
// row_start and colum_start are only used in case of 2 dense matrices
void fill_dense_from(const Matrix& A, Matrix &B);

declare_method(
  void, fill_dense_from, (virtual_<const Matrix&>, virtual_<Matrix&>, int64_t, int64_t)
)

} // namespace hicma

#endif // hicma_extension_headers_classes_h
