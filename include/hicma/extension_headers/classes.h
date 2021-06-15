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
  Hierarchical, split_omm,
  (
    virtual_<const Matrix&>,
    const std::vector<IndexRange>&, const std::vector<IndexRange>&,
    bool
  )
)

declare_method(
  void, fill_dense_from, (virtual_<const Matrix&>, virtual_<Matrix&>)
)

} // namespace hicma

#endif // hicma_extension_headers_classes_h
