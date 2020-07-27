#ifndef hicma_extension_headers_util_h
#define hicma_extension_headers_util_h

#include "hicma/classes/matrix.h"
#include "hicma/classes/initialization_helpers/basis_tracker.h"
#include "hicma/extension_headers/tuple_types.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <string>


namespace hicma
{

declare_method(
  unsigned long, get_memory_usage_omm,
  (virtual_<const Matrix&>, BasisTracker<BasisKey>&, bool)
)

declare_method(
  DoublePair, collect_diff_norm_omm,
  (virtual_<const Matrix&>, virtual_<const Matrix&>)
)

declare_method(void, print_omm, (virtual_<const Matrix&>))

declare_method(std::string, type_omm, (virtual_<const Matrix&>))

} // namespace hicma

#endif // hicma_extension_headers_util_h
