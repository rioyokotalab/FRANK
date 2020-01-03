#ifndef hicma_operations_LAPACK_svd_h
#define hicma_operations_LAPACK_svd_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

#include <tuple>

namespace hicma
{

class Dense;

std::tuple<Dense, Dense, Dense> svd(Dense& A);

std::tuple<Dense, Dense, Dense> sdd(Dense& A);

Dense get_singular_values(Dense& A);

} // namespace hicma

#endif // hicma_operations_LAPACK_svd_h
