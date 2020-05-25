#ifndef hicma_extension_headers_classes_h
#define hicma_extension_headers_classes_h

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/matrix.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/no_copy_split.h"
#include "hicma/classes/intitialization_helpers/cluster_tree.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cstdint>
#include <memory>


namespace hicma
{

declare_method(std::unique_ptr<Matrix>, clone, (virtual_<const Matrix&>))

declare_method(std::unique_ptr<Matrix>, move_clone, (virtual_<Matrix&&>))

declare_method(
  void, split_into_hierarchical,
  (Hierarchical&, virtual_<const Matrix &>, const ClusterTree&)
)

declare_method(
  NoCopySplit, make_no_copy_split, (virtual_<Matrix&>, int64_t, int64_t)
)

declare_method(
  MatrixProxy, make_view, (const ClusterTree&, virtual_<const Matrix&>)
)

declare_method(
  void, fill_dense_from, (virtual_<const Matrix&>, virtual_<Matrix&>)
)

} // namespace hicma

#endif // hicma_extension_headers_classes_h
